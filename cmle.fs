module ConstrainedML

open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions
open MathNet.Numerics.Statistics
open MathNet.Numerics.Optimization

// =============================================================================
// INFRASTRUCTURE & UTILS
// =============================================================================

type ResultBuilder() =
  member _.Bind(x, f) = Result.bind f x
  member _.Return(x) = Ok x
  member _.ReturnFrom(x) = x
  member _.Zero() = Ok ()

let result = ResultBuilder()

let log_info msg = printfn "[INFO] %s" msg
let log_error msg = printfn "[ERROR] %s" msg

// =============================================================================
// DOMAIN TYPES
// =============================================================================

type Interval = {
  lower: float
  upper: float
} with
  /// Returns true if value is in the (lower, upper] bin.
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

type CalibrationTarget = {
  intervals: Interval list
  expected_risks: float list
  tolerance: float
  /// Empirical or tabulated distribution over X in the target population.
  /// Each entry is a pair of (X, weight).
  x_distribution: (Vector<float> * float) list
}


// =============================================================================
// LOGIC: Density & Math
// =============================================================================

module MathHelpers =

  let sigmoid (eta: float) =
    let clipped = Math.Max(-50.0, Math.Min(50.0, eta))
    1.0 / (1.0 + Math.Exp(-clipped))

  let log_pdf_truncated (density: ConditionalDensity) (z: float) (x: Vector<float>) =
    // Truncated log-normal: log Z ~ N(mu, sigma) truncated above at 0,
    // so Z is supported on (0, 1]. This matches the setup in the cML paper.
    let eps = 1e-10
    let z_clamped = Math.Min(Math.Max(z, eps), 1.0 - eps)

    let x_aug =
      Seq.append [1.0] x
      |> Vector<float>.Build.DenseOfEnumerable

    let mu = density.tau_mean * x_aug
    let log_z = Math.Log(z_clamped)
    let norm_dist = Normal(mu, density.sigma)

    // pdf(log Z) under N(mu, sigma)
    let pdf_logz = norm_dist.Density(log_z)
    // truncation at log Z <= 0 -> divide by P(log Z <= 0)
    let cdf_upper = norm_dist.CumulativeDistribution(0.0)

    if cdf_upper <= 0.0 || Double.IsNaN(cdf_upper) || Double.IsNaN(pdf_logz) then
      -1e10
    else
      Math.Log(pdf_logz) - Math.Log(cdf_upper) - Math.Log(z_clamped)

  let sample_truncated (density: ConditionalDensity) (x: Vector<float>) (n_samples: int) (rng: Random) =
    // Sample from the same truncated log-normal: log Z ~ N(mu, sigma), log Z <= 0.
    let x_aug =
      Seq.append [1.0] x
      |> Vector<float>.Build.DenseOfEnumerable

    let mu = density.tau_mean * x_aug
    let norm_dist = Normal(mu, density.sigma, rng)

    let rec sample_one attempts =
      if attempts > 1000 then
        // extremely unlikely if the truncation probability is reasonable,
        // but guards against pathologies in optimization.
        1e-10
      else
        let y = norm_dist.Sample()
        if y <= 0.0 then
          let z_val = Math.Exp(y)
          Math.Max(1e-10, Math.Min(1.0 - 1e-10, z_val))
        else
          sample_one (attempts + 1)

    let samples =
      Array.init n_samples (fun _ -> sample_one 0)
    Vector<float>.Build.Dense(samples)


// =============================================================================
// LOGIC: Parameter Conversion
// =============================================================================

module ParameterOps =

  let to_vector (p: Parameters) : Vector<float> =
    let log_sigma = Math.Log(p.density.sigma)
    seq {
      // logistic regression part
      yield p.beta_0
      yield! p.beta_x
      yield p.beta_z
      // density part (tau_mean and log sigma)
      yield! p.density.tau_mean
      yield log_sigma
    }
    |> Vector<float>.Build.DenseOfEnumerable

  let from_vector (v: Vector<float>) (n_feat_x: int) (n_tau: int) : Parameters =
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
    let sigma = Math.Exp(log_sigma) |> fun s -> Math.Max(s, 0.01)

    {
      beta_0 = beta_0
      beta_x = beta_x
      beta_z = beta_z
      density = { tau_mean = tau_mean; sigma = sigma }
    }


// =============================================================================
// LOGIC: Fitting & Likelihood
// =============================================================================

module Fitting =

  let fit_conditional_density (x: Matrix<float>) (z: Vector<float>) : Result<ConditionalDensity, string> =
    try
      let z_safe = z.Map(fun zi -> Math.Max(zi, 1e-10))
      let log_z = z_safe.PointwiseLog()

      let ones = Vector<float>.Build.Dense(x.RowCount, 1.0)
      let x_aug = x.InsertColumn(0, ones)

      let tau_mean = x_aug.Solve(log_z)
      let preds = x_aug * tau_mean
      let residuals = log_z - preds
      let sigma = Math.Max(residuals.StandardDeviation(), 0.01)

      log_info (sprintf "Fitted density: sigma=%.4f" sigma)
      Ok { tau_mean = tau_mean; sigma = sigma }
    with ex ->
      Error (sprintf "Failed to fit density: %s" ex.Message)

  let compute_log_likelihood (params': Parameters) (data: TrainingData) =
    try
      let eta =
        (data.x * params'.beta_x)
        + (data.z * params'.beta_z)
        + params'.beta_0

      let log_lik_logistic =
        data.y.Map2((fun y e ->
          let e_clipped = Math.Max(-50.0, Math.Min(50.0, e))
          y * e_clipped - Math.Log(1.0 + Math.Exp(e_clipped))
        ), eta).Sum()

      let density_contrib =
        (data.x.EnumerateRows(), data.z)
        ||> Seq.zip
        |> Seq.sumBy (fun (xi, zi) -> MathHelpers.log_pdf_truncated params'.density zi xi)

      log_lik_logistic + density_contrib
    with _ ->
      -1e10

  let predict_risk (params': Parameters) (x: Vector<float>) (z: float) =
    let eta = params'.beta_0 + (params'.beta_x * x) + (params'.beta_z * z)
    MathHelpers.sigmoid eta

// =============================================================================
// LOGIC: Constraints
// =============================================================================

module Constraints =

  let compute_expected_risk_in_interval
    (params': Parameters)
    (interval: Interval)
    (x_dist: (Vector<float> * float) list)
    (base_model: Vector<float> -> float)
    (rng: Random) =

    // Keep only X in the desired risk bin according to the base model.
    let relevant_x =
      x_dist
      |> List.filter (fun (xi, _) -> interval.contains(base_model xi))

    match relevant_x with
    | [] -> 0.0
    | items ->
      let (total_risk, total_weight) =
        items
        |> List.fold (fun (acc_risk, acc_weight) (xi, weight) ->
          let z_samples = MathHelpers.sample_truncated params'.density xi 100 rng
          let avg_risk =
            z_samples.Map(fun zi -> Fitting.predict_risk params' xi zi).Mean()
          (acc_risk + (avg_risk * weight), acc_weight + weight)
        ) (0.0, 0.0)

      if total_weight = 0.0 then 0.0 else total_risk / total_weight

  let evaluate_constraints (params': Parameters) (calib: CalibrationTarget) (base_model) =
    // If we don't have any target X distribution, there is nothing to calibrate against.
    if List.isEmpty calib.x_distribution then
      0.0
    else
      let rng = Random(42)
      calib.intervals
      |> List.indexed
      |> List.sumBy (fun (i, interval) ->
        let expected = compute_expected_risk_in_interval params' interval calib.x_distribution base_model rng
        let target = calib.expected_risks.[i]
        let tol = calib.tolerance
        let upper_viol = Math.Max(0.0, expected - (1.0 + tol) * target)
        let lower_viol = Math.Max(0.0, (1.0 - tol) * target - expected)
        (upper_viol * upper_viol) + (lower_viol * lower_viol)
      )


// =============================================================================
// OPTIMIZATION (Impure Shell)
// =============================================================================

module Optimization =

  let initialize_parameters (data: TrainingData) (density: ConditionalDensity) =
    {
      beta_0 = 0.0
      beta_x = Vector<float>.Build.Dense(data.n_features_x, 0.0)
      beta_z = 0.0
      density = density
    }

  // manual central difference gradient to avoid 'Differentiate' dependency issues
  let private numerical_gradient (f: float[] -> float) (x: float[]) : float[] =
    let n = x.Length
    let h = 1e-5
    let grad = Array.zeroCreate n

    // need to mutate a copy to calculate differences
    let x_mutable = Array.copy x

    for i in 0 .. n - 1 do
      let original_val = x.[i]

      // f(x + h)
      x_mutable.[i] <- original_val + h
      let f_plus = f x_mutable

      // f(x - h)
      x_mutable.[i] <- original_val - h
      let f_minus = f x_mutable

      // centered difference
      grad.[i] <- (f_plus - f_minus) / (2.0 * h)

      // restore
      x_mutable.[i] <- original_val

    grad

  let optimize (data: TrainingData) (calib: CalibrationTarget) (base_model) (initial: Parameters) =

    let penalty_weight = 1000.0

    // objective function (Vector -> float)
    let objective (v: Vector<float>) =
      try
        let p = ParameterOps.from_vector v data.n_features_x initial.density.tau_mean.Count
        let nll = -(Fitting.compute_log_likelihood p data)
        let penalty = Constraints.evaluate_constraints p calib base_model
        nll + (penalty * penalty_weight)
      with _ ->
        Double.MaxValue

    try
      log_info "Starting BFGS Optimization..."

      let start_vec = ParameterOps.to_vector initial

      // gradient adapter
      let gradient (v: Vector<float>) : Vector<float> =
        let f_arr (arr: float[]) = objective (Vector<float>.Build.DenseOfArray(arr))

        // use local manual gradient
        let grad_arr = numerical_gradient f_arr (v.ToArray())

        Vector<float>.Build.DenseOfArray(grad_arr)

      let obj_func = ObjectiveFunction.Gradient(objective, gradient)
      let minimizer = BfgsMinimizer(1e-5, 1e-5, 1e-5, 500)
      let result = minimizer.FindMinimum(obj_func, start_vec)

      let final_params = ParameterOps.from_vector result.MinimizingPoint data.n_features_x initial.density.tau_mean.Count
      Ok final_params

    with ex ->
      Error (sprintf "Optimization failed: %s" ex.Message)


// =============================================================================
// API & WORKFLOW
// =============================================================================

let fit_constrained_model
  (data: TrainingData)
  (calib: CalibrationTarget)
  (base_model: Vector<float> -> float) =

  result {
    log_info "Starting constrained model fitting..."
    let! density = Fitting.fit_conditional_density data.x data.z
    let initial_params = Optimization.initialize_parameters data density
    let! final_params = Optimization.optimize data calib base_model initial_params
    log_info "Model fitted successfully."
    return final_params
  }

// =============================================================================
// USAGE EXAMPLE
// =============================================================================

let run_example () =
  // Synthetic example roughly mirroring the structure of the cML paper:
  // 1. Simulate a large target population (X, Z, Y)
  // 2. Build a base risk model using only X
  // 3. Construct calibration targets from the target population
  // 4. Fit a constrained model on a (possibly biased) source sample

  let rng = Random(42)
  let n_target = 50000
  let n_source = 2000
  let p = 4

  // "True" parameters for data generation
  let beta_0_true = -2.0
  let beta_x_true = Vector<float>.Build.DenseOfArray [| 0.5; -0.3; 0.2; 0.1 |]
  let beta_z_true = 1.0

  // log Z | X ~ N(tau_mean' * x_aug, sigma), truncated to log Z <= 0
  let tau_true =
    // intercept + 4 slopes
    Vector<float>.Build.DenseOfArray [| 0.0; 0.2; -0.1; 0.1; 0.3 |]
  let sigma_true = 0.4

  // 1) Simulate target X
  let x_target =
    Matrix<float>.Build.Dense(n_target, p, (fun _ j ->
      let u = rng.NextDouble()
      if j % 2 = 0 then
        // simple discrete covariate
        if u < 0.3 then 0.0
        elif u < 0.7 then 1.0
        else 2.0
      else
        // simple continuous covariate
        Normal.Sample(rng, 0.0, 1.0)
    ))

  let density_true = { tau_mean = tau_true; sigma = sigma_true }

  // 2) Simulate Z | X using the truncated log-normal helper
  let z_target =
    x_target.EnumerateRows()
    |> Seq.map (fun xi ->
      let samples = MathHelpers.sample_truncated density_true xi 1 rng
      samples.[0]
    )
    |> Vector<float>.Build.DenseOfEnumerable

  // 3) Simulate Y | X, Z from a logistic model
  let eta_target =
    (x_target * beta_x_true)
    + (z_target * beta_z_true)
    + beta_0_true

  let y_target =
    eta_target.Map(fun e ->
      let p_y = MathHelpers.sigmoid e
      if rng.NextDouble() < p_y then 1.0 else 0.0
    )

  // Base risk model using only X (as in the paper: φ(X))
  let base_model (xi: Vector<float>) =
    let eta_base = beta_0_true + (beta_x_true * xi)
    MathHelpers.sigmoid eta_base

  // Compute base risks in the target population
  let base_risks =
    x_target.EnumerateRows()
    |> Seq.map base_model
    |> Seq.toArray

  // Build calibration intervals based on quartiles of base risk
  let sorted = Array.copy base_risks
  Array.Sort(sorted)
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

  // Expected base-model risk inside each interval (target Pe_r)
  let expected_risks =
    intervals
    |> List.map (fun interval ->
      let mutable s = 0.0
      let mutable c = 0.0
      for i in 0 .. n_target - 1 do
        let r = base_risks.[i]
        if interval.contains r then
          s <- s + r
          c <- c + 1.0
      if c = 0.0 then 0.0 else s / c
    )

  // Empirical X distribution in the target population
  let x_distribution =
    [ for i in 0 .. n_target - 1 ->
        let xi = x_target.Row(i)
        (xi, 1.0) ]

  let target = {
    intervals = intervals
    expected_risks = expected_risks
    tolerance = 0.1
    x_distribution = x_distribution
  }

  // Source sample: here we just take a subset of the target population.
  // You could bias this (e.g., oversample high-risk individuals) to create
  // a distributional shift between source and target.
  let data = {
    x = x_target.SubMatrix(0, n_source, 0, p)
    z = z_target.SubVector(0, n_source)
    y = y_target.SubVector(0, n_source)
  }

  match fit_constrained_model data target base_model with
  | Ok params' ->
    printfn "Constrained model fit succeeded."
    printfn "Estimated beta_z: %f (true: %f)" params'.beta_z beta_z_true
    printfn "Estimated sigma (density): %f (true: %f)" params'.density.sigma sigma_true
  | Error e ->
    printfn "Constrained model fit failed: %s" e

run_example ()
