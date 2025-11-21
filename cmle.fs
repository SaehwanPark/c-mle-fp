// ============================================================================
// Constrained Maximum Likelihood Estimation (cMLE) for Risk Prediction
// Based on: Cao et al. (2024) Lifetime Data Analysis
// ============================================================================

open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions
open MathNet.Numerics.Optimization

// ============================================================================
// Logging
// ============================================================================

let log_info (msg: string) =
    printfn "[INFO] %s" msg

// ============================================================================
// Result computation expression
// ============================================================================

type ResultBuilder() =
  member _.Bind(m: Result<'a, 'e>, f: 'a -> Result<'b, 'e>) = Result.bind f m
  member _.Return(x: 'a) : Result<'a, 'e> = Ok x
  member _.ReturnFrom(m: Result<'a, 'e>) = m
  member _.Zero() : Result<unit, 'e> = Ok ()
  member _.Combine(r1: Result<unit, 'e>, r2: Result<'a, 'e>) =
    match r1 with
    | Ok () -> r2
    | Error e -> Error e

let result = ResultBuilder()

// ============================================================================
// Domain types
// ============================================================================

type Interval = {
  lower: float
  upper: float
} with
  member this.contains(value: float) =
    value > this.lower && value <= this.upper

type TrainingData = {
  x: Matrix<float>
  z: Vector<float>
  y: Vector<float>
} with
  member this.n_samples = this.y.Count
  member this.n_features_x = this.x.ColumnCount

type ConditionalDensity = {
  tau_mean: Vector<float>   // coefficients for log(Z) | X
  sigma: float
}

type Parameters = {
  beta_0: float
  beta_x: Vector<float>
  beta_z: float
  density: ConditionalDensity
}

/// Holds the structure for the integration "noise"
/// This allows us to use Common Random Numbers (CRN) for deterministic optimization
type IntegrationContext = {
  /// One vector of Uniform(0,1) noise per row in the target distribution
  noise_vectors: Vector<float> list
  /// Number of samples per integration point
  n_integration_samples: int
}

type CalibrationTarget = {
  intervals: Interval list
  expected_risks: float list
  tolerance: float
  /// Empirical or tabulated distribution of X in the *target* population
  /// Each element is (X, weight)
  x_distribution: (Vector<float> * float) list
}

// ============================================================================
// Math helpers
// ============================================================================

module MathHelpers =

  let sigmoid (eta: float) : float =
    let clipped = Math.Max(-50.0, Math.Min(50.0, eta))
    1.0 / (1.0 + Math.Exp(-clipped))

  /// Log-PDF of Truncated Log-Normal density
  /// log Z ~ N(mu, sigma^2), truncated to (-inf, 0], so Z in (0, 1]
  let log_pdf_truncated
    (density: ConditionalDensity)
    (z: float)
    (x: Vector<float>)
    : float =

    let eps = 1e-10
    let z_clamped = z |> max eps |> min (1.0 - eps)

    let x_aug =
      seq { yield 1.0; yield! x }
      |> Vector.Build.DenseOfEnumerable

    let mu = density.tau_mean * x_aug
    let log_z = Math.Log z_clamped
    
    // pdf(log Z) under N(mu, sigma)
    let pdf_val = Normal.PDF(mu, density.sigma, log_z)
    
    // CDF at truncation point (log(1) = 0)
    let cdf_upper = Normal.CDF(mu, density.sigma, 0.0)

    if cdf_upper <= 1e-15 || Double.IsNaN pdf_val then
      -1e10
    else
      // log(pdf_trunc) = log(pdf_normal) - log(cdf_upper) - log(z) (jacobian)
      Math.Log pdf_val - Math.Log cdf_upper - Math.Log z_clamped

  /// Deterministic Inverse Transform Sampling for Truncated Log-Normal.
  /// Used for integration during optimization to ensure smooth gradients.
  /// u_noise: Pre-generated Uniform(0,1) scalars.
  let sample_truncated_deterministic
    (density: ConditionalDensity)
    (x: Vector<float>)
    (u_noise: Vector<float>) 
    : Vector<float> =

    // 1. Calculate mu for this specific x
    let x_aug =
      seq { yield 1.0; yield! x }
      |> Vector.Build.DenseOfEnumerable
    
    let mu = density.tau_mean * x_aug
    let sigma = density.sigma

    // 2. Calculate the CDF value at the truncation point (log(1) = 0.0)
    // This represents the total probability mass of the non-truncated part
    let p_max = Normal.CDF(mu, sigma, 0.0)

    // 3. Inverse Transform:
    // Map u ~ [0,1] to p ~ [0, p_max]
    // Then find the quantile corresponding to p
    u_noise.Map (fun u ->
      let p_scaled = u * p_max
      
      // Handle numerical edge cases where p is effectively 0
      let p_safe = max p_scaled 1e-15
      
      // Find y such that CDF(y) = p_safe
      let log_z = Normal.InvCDF(mu, sigma, p_safe)
      
      // Transform back to Z space: z = exp(log_z)
      // Clamp to avoid 0.0 or 1.0 exactly
      Math.Exp(log_z) |> max 1e-10 |> min (1.0 - 1e-10)
    )

// ============================================================================
// Context Management
// ============================================================================

module ContextOps = 
  
  /// Pre-calculates the random noise vectors for the target distribution integration.
  /// This fixes the "randomness" for the duration of the optimization (CRN).
  let create_integration_context 
    (target_x: (Vector<float> * float) list) 
    (n_samples: int) 
    (seed: int) 
    : IntegrationContext =
    
    let rng = Random(seed)
    let noise = 
      target_x 
      |> List.map (fun _ -> 
          // Generate Fixed Uniform Noise U(0,1)
          let samples = Array.init n_samples (fun _ -> rng.NextDouble())
          Vector.Build.DenseOfArray samples
      )
        
    { noise_vectors = noise; n_integration_samples = n_samples }

// ============================================================================
// Fitting: density + likelihood
// ============================================================================

module Fitting =

  let fit_conditional_density
    (x: Matrix<float>)
    (z: Vector<float>)
    : Result<ConditionalDensity, string> =
    try
      // Guard against log(0)
      let z_safe = z.Map (fun zi -> max zi 1e-10)
      let log_z = z_safe.Map (fun v -> Math.Log v)

      // Augment X with intercept
      let ones = Vector.Build.Dense(x.RowCount, 1.0)
      let x_aug = x.InsertColumn(0, ones)

      // OLS for log Z | X
      let tau_mean = x_aug.Solve log_z
      let preds = x_aug * tau_mean
      let residuals = log_z - preds

      let sigma_raw =
        residuals
        |> Seq.toArray
        |> Array.map (fun r -> r * r)
        |> Array.average
        |> Math.Sqrt

      let sigma = max sigma_raw 0.01
      log_info (sprintf "Fitted Z|X density: sigma = %.4f" sigma)
      Ok { tau_mean = tau_mean; sigma = sigma }
    with ex ->
      Error (sprintf "Failed to fit conditional density: %s" ex.Message)

  let predict_risk (p: Parameters) (x: Vector<float>) (z: float) : float =
    let eta = (p.beta_x * x) + (p.beta_z * z) + p.beta_0
    MathHelpers.sigmoid eta

  let compute_log_likelihood (params': Parameters) (data: TrainingData) : float =
    try
      let eta = (data.x * params'.beta_x) + (data.z * params'.beta_z) + params'.beta_0

      // Logistic contribution
      let log_lik_logistic =
        Seq.zip (data.y :> seq<float>) (eta :> seq<float>)
        |> Seq.sumBy (fun (y, e) ->
          let e_clipped = e |> max -50.0 |> min 50.0
          // Stable computation: y*eta - log(1 + exp(eta))
          y * e_clipped - Math.Log(1.0 + Math.Exp e_clipped)
        )

      // Density contribution
      let density_contrib =
        Seq.zip (data.x.EnumerateRows()) (data.z :> seq<float>)
        |> Seq.sumBy (fun (xi, zi) ->
          MathHelpers.log_pdf_truncated params'.density zi xi
        )

      log_lik_logistic + density_contrib
    with _ -> -1e10

// ============================================================================
// Constraints & Penalty Logic
// ============================================================================

module Constraints =

  /// Computes risk using the DETERMINISTIC sampler
  let compute_expected_risk_in_interval_deterministic
    (params': Parameters)
    (interval: Interval)
    (x_dist: (Vector<float> * float) list)
    (context: IntegrationContext)
    (base_model: Vector<float> -> float)
    : float =

    // Zip the data with the pre-generated noise
    List.zip x_dist context.noise_vectors
    |> List.filter (fun ((x, _), _) -> interval.contains (base_model x))
    |> function
      | [] -> 0.0
      | items ->
        items
        |> List.map (fun ((xi, weight), u_vec) ->
          // Deterministic sampling
          let z_samples =
            MathHelpers.sample_truncated_deterministic
              params'.density xi u_vec

          let avg_risk =
            z_samples.ToArray()
            |> Array.averageBy (fun zi ->
                Fitting.predict_risk params' xi zi
              )

          avg_risk * weight, weight
        )
        |> List.reduce (fun (r1, w1) (r2, w2) -> (r1 + r2, w1 + w2))
        |> fun (total_risk, total_weight) ->
          if total_weight = 0.0 then 0.0
          else total_risk / total_weight

  /// Returns: (sum_of_squared_violations, max_absolute_violation)
  let calculate_violations
    (params': Parameters)
    (calib: CalibrationTarget)
    (context: IntegrationContext)
    (base_model: Vector<float> -> float)
    : float * float =
    
    let violations = 
      calib.intervals
      |> List.indexed
      |> List.map (fun (i, interval) ->
        let expected =
          compute_expected_risk_in_interval_deterministic
            params' interval calib.x_distribution context base_model

        let target = calib.expected_risks.[i]
        let tol = calib.tolerance

        // Constraint: target * (1-tol) <= expected <= target * (1+tol)
        // Reference [cite: 99] (Eq 3 in paper)
        let upper_viol = max 0.0 (expected - (1.0 + tol) * target)
        let lower_viol = max 0.0 ((1.0 - tol) * target - expected)

        // Squared penalty for the objective, max_abs for convergence check
        let sq_penalty = (upper_viol * upper_viol) + (lower_viol * lower_viol)
        let max_abs = max upper_viol lower_viol
        
        sq_penalty, max_abs
      )
        
    let total_sq_penalty = violations |> List.sumBy fst
    let max_violation = violations |> List.map snd |> List.max
    
    total_sq_penalty, max_violation

// ============================================================================
// Parameter vectorization
// ============================================================================

module ParameterOps =

  let to_vector (p: Parameters) : Vector<float> =
    let log_sigma = Math.Log p.density.sigma
    seq {
      yield p.beta_0
      yield! p.beta_x
      yield p.beta_z
      yield! p.density.tau_mean
      yield log_sigma
    }
    |> Vector.Build.DenseOfEnumerable

  let from_vector
    (v: Vector<float>)
    (n_feat_x: int)
    (n_tau: int)
    : Parameters =

    let idx_beta0 = 0
    let idx_beta_x = 1
    let idx_beta_z = idx_beta_x + n_feat_x
    let idx_tau = idx_beta_z + 1
    let idx_log_sigma = idx_tau + n_tau

    let beta_0 = v.[idx_beta0]
    let beta_x = v.SubVector(idx_beta_x, n_feat_x)
    let beta_z = v.[idx_beta_z]
    let tau_mean = v.SubVector(idx_tau, n_tau)
    let log_sigma = v.[idx_log_sigma]
    let sigma = log_sigma |> Math.Exp |> max 0.01

    { beta_0 = beta_0; beta_x = beta_x; beta_z = beta_z;
      density = { tau_mean = tau_mean; sigma = sigma } }

// ============================================================================
// Optimization
// ============================================================================

module Optimization =

  let initialize_parameters (data: TrainingData) (density: ConditionalDensity) : Parameters =
    { beta_0 = 0.0; beta_x = Vector.Build.Dense(data.n_features_x, 0.0);
      beta_z = 0.0; density = density }

  // Central difference gradient
  let private numerical_gradient (f: float[] -> float) (x: float[]) : float[] =
    let n = x.Length
    let h = 1e-5
    let grad = Array.zeroCreate n
    for i in 0 .. n - 1 do
      let original = x.[i]
      x.[i] <- original + h
      let f_plus = f x
      x.[i] <- original - h
      let f_minus = f x
      grad.[i] <- (f_plus - f_minus) / (2.0 * h)
      x.[i] <- original
    grad

  let optimize_with_scheduling
      (data: TrainingData)
      (calib: CalibrationTarget)
      (base_model: Vector<float> -> float)
      (initial: Parameters)
      : Result<Parameters, string> =

      // 1. Freeze noise for deterministic integration (CRN)
      // [cite: 139] (Paper implies integration over X and Z, we use MC for Z)
      let context = ContextOps.create_integration_context calib.x_distribution 100 42
      
      // Penalty Schedule Settings
      let max_penalty_weight = 1e6
      let penalty_multiplier = 10.0
      let violation_tolerance = 1e-4 // Convergence criteria

      // Inner Loop: BFGS with fixed rho
      let run_bfgs (start_params: Parameters) (rho: float) =
        let objective (v: Vector<float>) : float =
          try
            let p = ParameterOps.from_vector v data.n_features_x initial.density.tau_mean.Count
            let nll = -Fitting.compute_log_likelihood p data
            let penalty_sq, _ = Constraints.calculate_violations p calib context base_model
            // [cite: 84, 85] Maximize Likelihood subject to constraints -> Minimize NLL + Penalty
            nll + (rho * penalty_sq)
          with _ -> Double.MaxValue

        let start_vec = ParameterOps.to_vector start_params
        
        // Deterministic gradient allows BFGS to work
        let gradient (v: Vector<float>) : Vector<float> =
          let f_arr (arr: float[]) = arr |> Vector.Build.DenseOfArray |> objective
          let grad_arr = numerical_gradient f_arr (v.ToArray())
          Vector.Build.DenseOfArray grad_arr

        let obj_func = ObjectiveFunction.Gradient(objective, gradient)
        let minimizer = BfgsMinimizer(1e-5, 1e-5, 1e-5, 500)
        let res = minimizer.FindMinimum(obj_func, start_vec)
        
        ParameterOps.from_vector res.MinimizingPoint data.n_features_x initial.density.tau_mean.Count

      // Outer Loop: Penalty Scheduling
      let rec schedule_loop (current_params: Parameters) (rho: float) (iteration: int) =
        if rho > max_penalty_weight then
          log_info "Max penalty weight reached. Returning best effort."
          current_params
        else
          log_info (sprintf "Schedule Iteration %d: rho = %.1f" iteration rho)
          let next_params = run_bfgs current_params rho
          
          let _, max_viol = 
            Constraints.calculate_violations next_params calib context base_model
          
          log_info (sprintf "  -> Max Constraint Violation: %.6f" max_viol)

          if max_viol < violation_tolerance then
            log_info "Constraints satisfied within tolerance."
            next_params
          else
            // Warm start next iteration
            schedule_loop next_params (rho * penalty_multiplier) (iteration + 1)

      try
        // Start with a modest weight
        let final_params = schedule_loop initial 1.0 1
        Ok final_params
      with ex ->
        Error (sprintf "Optimization failed: %s" ex.Message)

// ============================================================================
// Public API
// ============================================================================

let fit_constrained_model
  (data: TrainingData)
  (calib: CalibrationTarget)
  (base_model: Vector<float> -> float)
  : Result<Parameters, string> =

  result {
    // 1. Fit Z | X density (initial guess) [cite: 80]
    let! density = Fitting.fit_conditional_density data.x data.z

    // 2. Initialize parameters
    let initial = Optimization.initialize_parameters data density

    // 3. Optimize with penalty scheduling [cite: 114] (Approximating KKT solution)
    let! final_params = 
        Optimization.optimize_with_scheduling data calib base_model initial

    return final_params
  }

// ============================================================================
// Simulation (for verification)
// ============================================================================

let simulate_target_population (n_target: int) (p: int) (rng: Random) 
  : Matrix<float> * Vector<float> * Vector<float> =

  let beta_0_true = -2.0
  let beta_x_true = Vector.Build.DenseOfArray [| 0.5; -0.3; 0.2; 0.1 |]
  let beta_z_true = 1.0
  let tau_true = Vector.Build.DenseOfArray [| 0.0; 0.2; -0.1; 0.1; 0.3 |]
  let density_true = { tau_mean = tau_true; sigma = 0.4 }

  // X generation
  let x_target =
    Matrix.Build.Dense(n_target, p, (fun _ j ->
      let u = rng.NextDouble()
      if j % 2 = 0 then if u < 0.3 then 0.0 elif u < 0.7 then 1.0 else 2.0
      else Normal.Sample(rng, 0.0, 1.0)
    ))

  // Z generation using clean sampling (not deterministic context here, this is ground truth)
  // Using a simpler sampling for data gen than the constrained one
  let z_target =
    x_target.EnumerateRows()
    |> Seq.map (fun xi ->
      let x_aug = seq {1.0; yield! xi} |> Vector.Build.DenseOfEnumerable
      let mu = tau_true * x_aug
      let raw = Normal.Sample(rng, mu, 0.4)
      // Truncation simulation (naive rejection for ground truth generation is fine)
      if raw > 0.0 then Math.Exp(0.0) else Math.Exp(raw) // simplified
    )
    |> Vector.Build.DenseOfEnumerable

  // Y generation
  let eta = (x_target * beta_x_true) + (z_target * beta_z_true) + beta_0_true
  let y_target =
    eta.Map (fun e ->
      if rng.NextDouble() < MathHelpers.sigmoid e then 1.0 else 0.0
    )

  x_target, z_target, y_target

let build_base_model (beta_0: float) (beta_x: Vector<float>) =
  fun (x: Vector<float>) -> MathHelpers.sigmoid (beta_0 + (beta_x * x))

let build_calibration_target (x: Matrix<float>) (base_model: Vector<float>->float) =
  let base_risks = x.EnumerateRows() |> Seq.map base_model |> Seq.toArray
  let sorted = Array.copy base_risks
  Array.Sort sorted
  let n = sorted.Length
  
  // Define quartiles as intervals [cite: 90]
  let intervals = [
    { lower = -1.0; upper = sorted.[n/4] }
    { lower = sorted.[n/4]; upper = sorted.[n/2] }
    { lower = sorted.[n/2]; upper = sorted.[3*n/4] }
    { lower = sorted.[3*n/4]; upper = 2.0 }
  ]
  
  // P_r^e (Expected risk in target population) [cite: 95]
  let expected_risks =
    intervals |> List.map (fun iv ->
      let in_bin = base_risks |> Array.filter iv.contains
      if in_bin.Length = 0 then 0.0 else Array.average in_bin
    )

  let x_dist = [ for r in x.EnumerateRows() -> (r, 1.0) ]
  
  { intervals = intervals; expected_risks = expected_risks; 
    tolerance = 0.1; x_distribution = x_dist }

let run_example () =
  let rng = Random(42)
  let n_target, n_source, p = 1000, 500, 4
  
  let x_t, z_t, y_t = simulate_target_population n_target p rng
  // Base model (mimicking BCRAT) [cite: 69]
  let base_model = build_base_model -2.0 (Vector.Build.DenseOfArray [|0.5;-0.3;0.2;0.1|])
  let calib = build_calibration_target x_t base_model
  
  // Source data (subset)
  let source_data = { 
      x = x_t.SubMatrix(0, n_source, 0, p); 
      z = z_t.SubVector(0, n_source); 
      y = y_t.SubVector(0, n_source) 
  }

  match fit_constrained_model source_data calib base_model with
  | Ok res -> 
      printfn "Success! Beta_Z: %.4f (True ~1.0)" res.beta_z
      printfn "Density Sigma: %.4f" res.density.sigma
  | Error e -> printfn "Error: %s" e

// run_example()