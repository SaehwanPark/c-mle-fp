module CMLE.Vectorized

open System
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions
open MathNet.Numerics.Optimization
open MathNet.Numerics.Random // Added for MersenneTwister

// ============================================================================
// Configuration & Logging
// ============================================================================

// Optional: Uncomment to force managed provider if hardware differences 
// (AVX on Linux vs NEON on Mac) cause slight floating point drift.
// Control.UseManaged() 

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
  tau_mean: Vector<float>
  sigma: float
}

type Parameters = {
  beta_0: float
  beta_x: Vector<float>
  beta_z: float
  density: ConditionalDensity
}

type IntegrationContext = {
  // CHANGED: Flattened noise into a Matrix for vectorization.
  // Rows = Observations, Cols = Integration Samples
  noise_matrix: Matrix<float>
  n_integration_samples: int
}

type CalibrationTarget = {
  intervals: Interval list
  expected_risks: float list
  tolerance: float
  // We keep the raw matrix rows for index lookup, but pre-calculating
  // the matrix form in the context is preferred for speed.
  x_matrix: Matrix<float> 
  weights: Vector<float>
}

// ============================================================================
// Math helpers (Robust & Vectorized)
// ============================================================================

module MathHelpers =

  let sigmoid (eta: float) : float =
    let clipped = Math.Max(-100.0, Math.Min(100.0, eta))
    1.0 / (1.0 + Math.Exp(-clipped))

  /// Log-PDF of Truncated Log-Normal density
  let log_pdf_truncated
    (density: ConditionalDensity)
    (z: float)
    (x: Vector<float>)
    : float =

    let eps = 1e-12
    let z_clamped = z |> max eps |> min (1.0 - eps)

    let x_aug =
      seq { yield 1.0; yield! x }
      |> Vector.Build.DenseOfEnumerable

    let mu = density.tau_mean * x_aug
    let sigma = max 1e-5 density.sigma 
    let log_z = Math.Log z_clamped
    
    let pdf_val = Normal.PDF(mu, sigma, log_z)
    let cdf_upper = Normal.CDF(mu, sigma, 0.0)

    let safe_pdf = max 1e-100 pdf_val
    let safe_cdf = max 1e-100 cdf_upper

    Math.Log safe_pdf - Math.Log safe_cdf - Math.Log z_clamped

  /// NEW: Vectorized Deterministic Inverse Transform Sampling
  /// Maps pre-generated Uniform(0,1) noise matrix to the truncated distribution
  /// Returns: Matrix (N_obs x N_samples)
  let sample_truncated_matrix
    (density: ConditionalDensity)
    (x_aug: Matrix<float>)       // Matrix: N x (Features + 1)
    (u_noise: Matrix<float>)     // Matrix: N x Samples
    : Matrix<float> =

    // 1. Calculate Mean for every observation (Vectorized)
    // mu_vec[i] corresponds to row i
    let mu_vec = x_aug * density.tau_mean
    let sigma = max 1e-5 density.sigma

    // 2. Map the noise matrix to Z values
    // MathNet's MapIndexed is efficient enough here.
    // We process every cell (i, j) where i is observation, j is sample index
    u_noise.MapIndexed (fun i _ u ->
        let mu = mu_vec.[i]
        
        // CDF value at truncation point 0.0
        let p_max = Normal.CDF(mu, sigma, 0.0)
        
        // Scale u to the valid CDF range [0, p_max]
        let p_scaled = u * p_max
        let p_safe = max p_scaled 1e-12
        
        // Inverse CDF
        let log_z = Normal.InvCDF(mu, sigma, p_safe)
        
        // Transform back
        Math.Exp(log_z) |> max 1e-12 |> min (1.0 - 1e-12)
    )

// ============================================================================
// Context Management
// ============================================================================

module ContextOps = 
  
  let create_integration_context 
    (x_matrix: Matrix<float>)
    (n_samples: int) 
    (seed: int) 
    : IntegrationContext =
    
    // CHANGED: Use MersenneTwister for cross-platform consistency
    let rng = MersenneTwister(seed)
    
    let n_rows = x_matrix.RowCount

    // CHANGED: Generate one large dense matrix of noise
    let noise = Matrix.Build.Dense(n_rows, n_samples, (fun _ _ -> rng.NextDouble()))
      
    { noise_matrix = noise; n_integration_samples = n_samples }

// ============================================================================
// Fitting
// ============================================================================

module Fitting =

  let fit_conditional_density
    (x: Matrix<float>)
    (z: Vector<float>)
    : Result<ConditionalDensity, string> =
    try
      let z_safe = z.Map (fun zi -> max zi 1e-10)
      let log_z = z_safe.Map (fun v -> Math.Log v)

      let ones = Vector.Build.Dense(x.RowCount, 1.0)
      let x_aug = x.InsertColumn(0, ones)

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
    let eta = (data.x * params'.beta_x) + (data.z * params'.beta_z) + params'.beta_0

    let log_lik_logistic =
      Seq.zip (data.y :> seq<float>) (eta :> seq<float>)
      |> Seq.sumBy (fun (y, e) ->
        if e > 0.0 then
          y * e - (e + Math.Log(1.0 + Math.Exp(-e)))
        else
          y * e - Math.Log(1.0 + Math.Exp(e))
      )

    let density_contrib =
      Seq.zip (data.x.EnumerateRows()) (data.z :> seq<float>)
      |> Seq.sumBy (fun (xi, zi) ->
        MathHelpers.log_pdf_truncated params'.density zi xi
      )

    log_lik_logistic + density_contrib

// ============================================================================
// Constraints (Vectorized)
// ============================================================================

module Constraints =

  /// NEW: Fully vectorized violation calculation
  let calculate_violations
    (params': Parameters)
    (calib: CalibrationTarget)
    (context: IntegrationContext)
    (base_model_scores: Vector<float>) // Optimization: Pass pre-calculated base scores
    : float * float =
    
    // 1. Prepare X_aug for density sampling
    let ones = Vector.Build.Dense(calib.x_matrix.RowCount, 1.0)
    let x_aug = calib.x_matrix.InsertColumn(0, ones)

    // 2. Vectorized Sampling of Z (N x Samples)
    // This replaces the row-by-row loop
    let z_samples_mat = 
        MathHelpers.sample_truncated_matrix params'.density x_aug context.noise_matrix

    // 3. Calculate Risk Matrix (N x Samples)
    // Pre-calculate the linear part of X: beta_0 + X * beta_x
    let x_logits = (calib.x_matrix * params'.beta_x) + params'.beta_0

    // We iterate the matrix to apply the sigmoid.
    // Note: (x_logits.[i] + beta_z * z) is the eta
    let risk_mat = 
        z_samples_mat.MapIndexed(fun i _ z ->
            let eta = x_logits.[i] + (params'.beta_z * z)
            MathHelpers.sigmoid eta
        )
    
    // 4. Average across integration samples (Row-wise mean) to get E[Risk|X]
    // Result is Vector of length N
    let expected_risk_per_obs = 
        risk_mat.RowSums() / float context.n_integration_samples

    // 5. Calculate Violations per Interval
    let violations = 
      calib.intervals
      |> List.indexed
      |> List.map (fun (i, interval) ->
        
        // Filter indices based on base_model_scores
        // (This is still iterative but fast compared to integration)
        let indices = 
             base_model_scores 
             |> Seq.toList
             |> List.indexed 
             |> List.filter (fun (_, score) -> interval.contains score)
             |> List.map fst

        // Weighted Average of Expected Risks for this interval
        let (total_risk, total_weight) =
            indices 
            |> List.fold (fun (acc_r, acc_w) idx -> 
                let w = calib.weights.[idx]
                (acc_r + (expected_risk_per_obs.[idx] * w), acc_w + w)
            ) (0.0, 0.0)

        let expected = 
            if total_weight = 0.0 then 0.0 
            else total_risk / total_weight

        let target = calib.expected_risks.[i]
        let tol = calib.tolerance

        let upper_viol = max 0.0 (expected - (1.0 + tol) * target)
        let lower_viol = max 0.0 ((1.0 - tol) * target - expected)

        let sq_penalty = (upper_viol * upper_viol) + (lower_viol * lower_viol)
        let max_abs = max upper_viol lower_viol
       
        sq_penalty, max_abs
      )
      
    let total_sq_penalty = violations |> List.sumBy fst
    let max_violation = violations |> List.map snd |> List.max
    
    total_sq_penalty, max_violation

// ============================================================================
// Parameter Ops
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
    let sigma = Math.Exp(log_sigma) + 1e-6

    { beta_0 = beta_0; beta_x = beta_x; beta_z = beta_z;
      density = { tau_mean = tau_mean; sigma = sigma } }

// ============================================================================
// Optimization
// ============================================================================

module Optimization =

  let initialize_parameters (data: TrainingData) (density: ConditionalDensity) : Parameters =
    { beta_0 = 0.0;
      beta_x = Vector.Build.Dense(data.n_features_x, 0.0);
      beta_z = 0.0; density = density }

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

    // Freeze noise for deterministic integration (CRN)
    // CHANGED: Use the Matrix-based context creator
    let context = ContextOps.create_integration_context calib.x_matrix 100 42
    
    // Pre-calculate base model scores once to avoid re-computing in loop
    let base_scores = 
        calib.x_matrix.EnumerateRows() 
        |> Seq.map base_model 
        |> Vector.Build.DenseOfEnumerable

    let max_penalty_weight = 1e6
    let penalty_multiplier = 5.0 
    let violation_tolerance = 1e-4 

    let n_samples = float data.n_samples

    let run_bfgs (start_params: Parameters) (rho: float) =
      let objective (v: Vector<float>) : float =
        let p = ParameterOps.from_vector v data.n_features_x initial.density.tau_mean.Count
        
        let log_lik = Fitting.compute_log_likelihood p data
        let avg_nll = -(log_lik / n_samples)

        // CHANGED: Call vectorized violations
        let penalty_sq, _ = Constraints.calculate_violations p calib context base_scores
        
        avg_nll + (rho * penalty_sq)

      let start_vec = ParameterOps.to_vector start_params
      
      let gradient (v: Vector<float>) : Vector<float> =
        let f_arr (arr: float[]) = arr |> Vector.Build.DenseOfArray |> objective
        let grad_arr = numerical_gradient f_arr (v.ToArray())
        Vector.Build.DenseOfArray grad_arr

      let obj_func = ObjectiveFunction.Gradient(objective, gradient)
      let minimizer = BfgsMinimizer(1e-4, 1e-5, 1e-5, 1000)
      
      try 
        let res = minimizer.FindMinimum(obj_func, start_vec)
        ParameterOps.from_vector res.MinimizingPoint data.n_features_x initial.density.tau_mean.Count
      with _ ->
        printfn "[WARN] BFGS Step Failed. Keeping parameters."
        start_params

    let rec schedule_loop (current_params: Parameters) (rho: float) (iteration: int) =
      if rho > max_penalty_weight then
        log_info "Max penalty weight reached."
        current_params
      else
        log_info (sprintf "Schedule Iteration %d: rho = %.1f" iteration rho)
        let next_params = run_bfgs current_params rho
        
        let _, max_viol = 
          Constraints.calculate_violations next_params calib context base_scores
        
        log_info (sprintf "  -> Max Constraint Violation: %.6f" max_viol)

        if max_viol < violation_tolerance then
          log_info "Constraints satisfied."
          next_params
        else
          schedule_loop next_params (rho * penalty_multiplier) (iteration + 1)

    try
      let final_params = schedule_loop initial 0.1 1
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
    let! density = Fitting.fit_conditional_density data.x data.z
    let initial = Optimization.initialize_parameters data density
    let! final_params = 
      Optimization.optimize_with_scheduling data calib base_model initial

    return final_params
  }

// ============================================================================
// Simulation
// ============================================================================

let simulate_target_population (n_target: int) (p: int) (seed: int) 
  : Matrix<float> * Vector<float> * Vector<float> =

  // CHANGED: Use MersenneTwister for cross-platform consistency
  let rng = MersenneTwister(seed)

  let beta_0_true = -2.0
  let beta_x_true = Vector.Build.DenseOfArray [| 0.5; -0.3; 0.2; 0.1 |]
  let beta_z_true = 1.0
  let tau_true = Vector.Build.DenseOfArray [| 0.0; 0.2; -0.1; 0.1; 0.3 |]

  let x_target =
    Matrix.Build.Dense(n_target, p, (fun _ j ->
      let u = rng.NextDouble()
      if j % 2 = 0 then if u < 0.3 then 0.0 elif u < 0.7 then 1.0 else 2.0
      else Normal.Sample(rng, 0.0, 1.0)
    ))

  let z_target =
    x_target.EnumerateRows()
    |> Seq.map (fun xi ->
      let x_aug = seq {1.0; yield! xi} |> Vector.Build.DenseOfEnumerable
      let mu = tau_true * x_aug
      let raw = Normal.Sample(rng, mu, 0.4)
      if raw > 0.0 then Math.Exp(0.0) else Math.Exp(raw)
    )
    |> Vector.Build.DenseOfEnumerable

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
  
  let intervals = [
    { lower = -1.0; upper = sorted.[n/4] }
    { lower = sorted.[n/4]; upper = sorted.[n/2] }
    { lower = sorted.[n/2]; upper = sorted.[3*n/4] }
    { lower = sorted.[3*n/4]; upper = 2.0 }
  ]
  
  let expected_risks =
    intervals |> List.map (fun iv ->
      let in_bin = base_risks |> Array.filter iv.contains
      if in_bin.Length = 0 then 0.0 else Array.average in_bin
    )

  // CHANGED: Setup matrix and weights in target for vectorization
  let weights = Vector.Build.Dense(x.RowCount, 1.0)
  
  { intervals = intervals;
    expected_risks = expected_risks; 
    tolerance = 0.1; 
    x_matrix = x; 
    weights = weights }

let run_example () =
  let n_target, n_source, p = 1000, 500, 4
  
  // Pass seed explicitly
  let x_t, z_t, y_t = simulate_target_population n_target p 42
  let base_model = build_base_model -2.0 (Vector.Build.DenseOfArray [|0.5;-0.3;0.2;0.1|])
  let calib = build_calibration_target x_t base_model
  
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
