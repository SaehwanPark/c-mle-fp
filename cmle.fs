open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions
open MathNet.Numerics.Optimization

// ============================================================================
// Logging
// ============================================================================

let log_info (msg: string) =
  // Swap out for a proper logger if desired
  printfn "[INFO] %s" msg

// ============================================================================
// Result computation expression (railway-oriented)
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
  /// (lower, upper] bin
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

  /// Truncated log-normal density:
  /// log Z ~ N(mu, sigma^2), truncated to (-∞, 0], so Z ∈ (0, 1].
  let log_pdf_truncated
    (density: ConditionalDensity)
    (z: float)
    (x: Vector<float>)
    : float =

    let eps = 1e-10
    let z_clamped = z |> max eps |> min (1.0 - eps)

    let x_aug =
      seq {
        yield 1.0
        yield! x
      }
      |> Vector.Build.DenseOfEnumerable

    let mu = density.tau_mean * x_aug
    let log_z = Math.Log z_clamped
    let norm_dist = Normal(mu, density.sigma)

    // pdf(log Z) under N(mu, sigma)
    let pdf_logz = norm_dist.Density log_z
    // truncation at log Z <= 0
    let cdf_upper = norm_dist.CumulativeDistribution 0.0

    if cdf_upper <= 0.0 || Double.IsNaN cdf_upper || Double.IsNaN pdf_logz then
      -1e10
    else
      Math.Log pdf_logz
      - Math.Log cdf_upper
      - Math.Log z_clamped

  /// Sample from truncated log-normal: log Z ~ N(mu, sigma^2), log Z <= 0.
  let sample_truncated
    (density: ConditionalDensity)
    (x: Vector<float>)
    (n_samples: int)
    (rng: Random)
    : Vector<float> =

    let x_aug =
      seq {
        yield 1.0
        yield! x
      }
      |> Vector.Build.DenseOfEnumerable

    let mu = density.tau_mean * x_aug
    let norm_dist = Normal(mu, density.sigma, rng)

    let sample_one () =
      let mutable attempts = 0
      let mutable z = 1e-10
      let mutable done_ = false

      while not done_ && attempts < 1000 do
        let y = norm_dist.Sample()
        if y <= 0.0 then
          let zz = Math.Exp y |> max 1e-10 |> min (1.0 - 1e-10)
          z <- zz
          done_ <- true
        else
          attempts <- attempts + 1

      z

    Array.init n_samples (fun _ -> sample_one ())
    |> Vector.Build.DenseOfArray

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
      let z_safe =
        z.Map (fun zi -> max zi 1e-10)

      let log_z =
        z_safe.Map (fun v -> Math.Log v)

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

  /// Predict P(Y=1 | X, Z) under logistic model.
  let predict_risk
    (p: Parameters)
    (x: Vector<float>)
    (z: float)
    : float =

    let eta =
      (p.beta_x * x)
      + (p.beta_z * z)
      + p.beta_0

    MathHelpers.sigmoid eta

  /// Joint log-likelihood of Y and Z given X.
  let compute_log_likelihood
    (params': Parameters)
    (data: TrainingData)
    : float =
    try
      let eta =
        (data.x * params'.beta_x)
        + (data.z * params'.beta_z)
        + params'.beta_0

      // Logistic contribution
      let log_lik_logistic =
        Seq.zip (data.y :> seq<float>) (eta :> seq<float>)
        |> Seq.sumBy (fun (y, e) ->
          let e_clipped = e |> max -50.0 |> min 50.0
          y * e_clipped - Math.Log(1.0 + Math.Exp e_clipped)
        )

      // Density contribution
      let density_contrib =
        Seq.zip (data.x.EnumerateRows()) (data.z :> seq<float>)
        |> Seq.sumBy (fun (xi, zi) ->
            MathHelpers.log_pdf_truncated params'.density zi xi
          )

      log_lik_logistic + density_contrib
    with _ ->
      -1e10

// ============================================================================
// Constraints
// ============================================================================

module Constraints =

  let compute_expected_risk_in_interval
    (params': Parameters)
    (interval: Interval)
    (x_dist: (Vector<float> * float) list)
    (base_model: Vector<float> -> float)
    (rng: Random)
    : float =

    x_dist
    |> List.filter (fun (x, _) -> interval.contains (base_model x))
    |> function
       | [] -> 0.0
       | items ->
           items
           |> List.map (fun (xi, weight) ->
               let z_samples =
                 MathHelpers.sample_truncated
                   params'.density
                   xi
                   100
                   rng

               let avg_risk =
                 z_samples.ToArray()
                 |> Array.averageBy (fun zi ->
                     Fitting.predict_risk params' xi zi
                   )

               avg_risk * weight, weight
           )
           |> List.reduce (fun (r1, w1) (r2, w2) ->
               (r1 + r2, w1 + w2)
           )
           |> fun (total_risk, total_weight) ->
               if total_weight = 0.0 then 0.0
               else total_risk / total_weight

  let evaluate_constraints
    (params': Parameters)
    (calib: CalibrationTarget)
    (base_model: Vector<float> -> float)
    : float =

    if List.isEmpty calib.x_distribution then
      0.0
    else
      let rng = Random 42
      calib.intervals
      |> List.indexed
      |> List.sumBy (fun (i, interval) ->
          let expected =
            compute_expected_risk_in_interval
              params'
              interval
              calib.x_distribution
              base_model
              rng

          let target = calib.expected_risks.[i]
          let tol = calib.tolerance

          let upper_viol =
            max 0.0 (expected - (1.0 + tol) * target)

          let lower_viol =
            max 0.0 ((1.0 - tol) * target - expected)

          (upper_viol * upper_viol)
          + (lower_viol * lower_viol)
      )

// ============================================================================
// Parameter vectorization
// ============================================================================

module ParameterOps =

  let to_vector (p: Parameters) : Vector<float> =
    let log_sigma = Math.Log p.density.sigma
    seq {
      // logistic parameters
      yield p.beta_0
      yield! p.beta_x
      yield p.beta_z
      // density parameters
      yield! p.density.tau_mean
      yield log_sigma
    }
    |> Vector.Build.DenseOfEnumerable

  let from_vector
    (v: Vector<float>)
    (n_feat_x: int)
    (n_tau: int)
    : Parameters =

    // layout:
    // [ beta_0
    //   beta_x (n_feat_x)
    //   beta_z
    //   tau_mean (n_tau)
    //   log_sigma ]
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

    {
      beta_0 = beta_0
      beta_x = beta_x
      beta_z = beta_z
      density = { tau_mean = tau_mean; sigma = sigma }
    }

// ============================================================================
// Optimization
// ============================================================================

module Optimization =

  let initialize_parameters
    (data: TrainingData)
    (density: ConditionalDensity)
    : Parameters =

    {
      beta_0 = 0.0
      beta_x = Vector.Build.Dense(data.n_features_x, 0.0)
      beta_z = 0.0
      density = density
    }

  /// Central-difference numerical gradient on R^n.
  let private numerical_gradient
    (f: float[] -> float)
    (x: float[])
    : float[] =

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

  let optimize
    (data: TrainingData)
    (calib: CalibrationTarget)
    (base_model: Vector<float> -> float)
    (initial: Parameters)
    : Result<Parameters, string> =

    let penalty_weight = 1000.0

    let objective (v: Vector<float>) : float =
      try
        let p =
          ParameterOps.from_vector
            v
            data.n_features_x
            initial.density.tau_mean.Count

        let nll =
          Fitting.compute_log_likelihood p data
          |> (~-)

        let penalty =
          Constraints.evaluate_constraints p calib base_model

        nll + penalty_weight * penalty
      with _ ->
        Double.MaxValue

    try
      log_info "Starting BFGS optimization..."

      let start_vec = ParameterOps.to_vector initial

      let gradient (v: Vector<float>) : Vector<float> =
        let f_arr (arr: float[]) =
          arr
          |> Vector.Build.DenseOfArray
          |> objective

        let grad_arr =
          v.ToArray()
          |> numerical_gradient f_arr

        Vector.Build.DenseOfArray grad_arr

      let obj_func = ObjectiveFunction.Gradient(objective, gradient)
      let minimizer = BfgsMinimizer(1e-5, 1e-5, 1e-5, 500)
      let result = minimizer.FindMinimum(obj_func, start_vec)

      let final_params =
        ParameterOps.from_vector
          result.MinimizingPoint
          data.n_features_x
          initial.density.tau_mean.Count

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
    // 1. Fit Z | X density (initial guess)
    let! density =
      Fitting.fit_conditional_density data.x data.z

    // 2. Initialize parameters
    let initial = Optimization.initialize_parameters data density

    // 3. Optimize with penalty constraints
    let! final_params =
      Optimization.optimize data calib base_model initial

    return final_params
  }

// ============================================================================
// Example: synthetic simulation
// ============================================================================

let simulate_target_population
  (n_target: int)
  (p: int)
  (rng: Random)
  : Matrix<float> * Vector<float> * Vector<float> =

  // True parameters
  let beta_0_true = -2.0
  let beta_x_true =
    Vector.Build.DenseOfArray [| 0.5; -0.3; 0.2; 0.1 |]
  let beta_z_true = 1.0

  let tau_true =
    Vector.Build.DenseOfArray [| 0.0; 0.2; -0.1; 0.1; 0.3 |]
  let sigma_true = 0.4
  let density_true = { tau_mean = tau_true; sigma = sigma_true }

  // X in target population
  let x_target =
    Matrix.Build.Dense(n_target, p, (fun _ j ->
      let u = rng.NextDouble()
      if j % 2 = 0 then
        if u < 0.3 then 0.0
        elif u < 0.7 then 1.0
        else 2.0
      else
        Normal.Sample(rng, 0.0, 1.0)
    ))

  // Z | X
  let z_target =
    x_target.EnumerateRows()
    |> Seq.map (fun xi ->
        MathHelpers.sample_truncated density_true xi 1 rng
        |> fun v -> v.[0]
    )
    |> Vector.Build.DenseOfEnumerable

  // Y | X, Z
  let eta_target =
    (x_target * beta_x_true)
    + (z_target * beta_z_true)
    + beta_0_true

  let y_target =
    eta_target.Map (fun e ->
        let p_y = MathHelpers.sigmoid e
        if rng.NextDouble() < p_y then 1.0 else 0.0
    )

  x_target, z_target, y_target

let build_base_model
  (beta_0_true: float)
  (beta_x_true: Vector<float>)
  : Vector<float> -> float =

  fun (x: Vector<float>) ->
    let eta = beta_0_true + (beta_x_true * x)
    MathHelpers.sigmoid eta

let build_calibration_target
  (x_target: Matrix<float>)
  (base_model: Vector<float> -> float)
  (tolerance: float)
  : CalibrationTarget =

  let n_target = x_target.RowCount

  let base_risks =
    x_target.EnumerateRows()
    |> Seq.map base_model
    |> Seq.toArray

  let sorted = Array.copy base_risks
  Array.Sort sorted

  let min_r = sorted.[0]
  let max_r = sorted.[sorted.Length - 1]
  let q1 = sorted.[sorted.Length / 4]
  let q2 = sorted.[sorted.Length / 2]
  let q3 = sorted.[3 * sorted.Length / 4]

  let intervals = [
    { lower = min_r - 1e-6; upper = q1 }
    { lower = q1; upper = q2 }
    { lower = q2; upper = q3 }
    { lower = q3; upper = max_r + 1e-6 }
  ]

  let expected_risks =
    intervals
    |> List.map (fun interval ->
        base_risks
        |> Array.filter interval.contains
        |> function
           | [||] -> 0.0
           | arr  -> Array.average arr
    )

  let x_distribution =
    [ for i in 0 .. n_target - 1 ->
        let xi = x_target.Row(i)
        (xi, 1.0) ]

  {
    intervals = intervals
    expected_risks = expected_risks
    tolerance = tolerance
    x_distribution = x_distribution
  }

let sample_source_population
  (x_target: Matrix<float>)
  (z_target: Vector<float>)
  (y_target: Vector<float>)
  (n_source: int)
  : TrainingData =

  {
    x = x_target.SubMatrix(0, n_source, 0, x_target.ColumnCount)
    z = z_target.SubVector(0, n_source)
    y = y_target.SubVector(0, n_source)
  }

let run_example () =
  let rng = Random 42
  let n_target = 50_000
  let n_source = 2_000
  let p = 4

  // True parameters for base model (must match simulate_target_population)
  let beta_0_true = -2.0
  let beta_x_true =
    Vector.Build.DenseOfArray [| 0.5; -0.3; 0.2; 0.1 |]

  let x_target, z_target, y_target =
    simulate_target_population n_target p rng

  let base_model = build_base_model beta_0_true beta_x_true

  let calib_target =
    build_calibration_target x_target base_model 0.1

  let source_data =
    sample_source_population x_target z_target y_target n_source

  match fit_constrained_model source_data calib_target base_model with
  | Ok params' ->
      printfn "Constrained fit succeeded."
      printfn "beta_z (estimated): %.4f" params'.beta_z
      printfn "sigma (estimated): %.4f" params'.density.sigma
  | Error e ->
      printfn "Constrained fit failed: %s" e

// Uncomment if you want this to run automatically as a script
// run_example ()
