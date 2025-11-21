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
  x_distribution: Map<float list, float>
}

// =============================================================================
// LOGIC: Density & Math
// =============================================================================

module MathHelpers =

  let sigmoid (eta: float) =
    let clipped = Math.Max(-50.0, Math.Min(50.0, eta))
    1.0 / (1.0 + Math.Exp(-clipped))

  let log_pdf_truncated (density: ConditionalDensity) (z: float) (x: Vector<float>) =
    let z_safe = Math.Max(z, 1e-10)

    let x_aug =
      Seq.append [1.0] x
      |> Vector<float>.Build.DenseOfEnumerable

    let mu = density.tau_mean * x_aug

    let log_z = Math.Log(z_safe)
    let a_std = -mu / density.sigma

    try
      let norm_dist = Normal(mu, density.sigma)
      let pdf_val = norm_dist.Density(log_z)
      let survival = 1.0 - Normal.CDF(0.0, 1.0, a_std)

      Math.Log(pdf_val) - Math.Log(survival) - Math.Log(z_safe)
    with _ ->
      -1e10

  let sample_truncated (density: ConditionalDensity) (x: Vector<float>) (n_samples: int) (rng: Random) =
    let x_aug =
      Seq.append [1.0] x
      |> Vector<float>.Build.DenseOfEnumerable

    let mu = density.tau_mean * x_aug

    let samples =
      Array.init n_samples (fun _ ->
        let s = LogNormal.Sample(rng, mu, density.sigma)
        Math.Max(1e-10, Math.Min(1e10, s))
      )
    Vector<float>.Build.Dense(samples)

// =============================================================================
// LOGIC: Parameter Conversion
// =============================================================================

module ParameterOps =

  let to_vector (p: Parameters) : Vector<float> =
    seq {
      yield p.beta_0
      yield! p.beta_x
      yield p.beta_z
    }
    |> Vector<float>.Build.DenseOfEnumerable

  let from_vector (v: Vector<float>) (n_feat_x: int) (density: ConditionalDensity) : Parameters =
    {
      beta_0 = v.[0]
      beta_x = v.SubVector(1, n_feat_x)
      beta_z = v.[1 + n_feat_x]
      density = density
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
    (x_dist: Map<float list, float>)
    (base_model: Vector<float> -> float)
    (rng: Random) =

    let relevant_x =
      x_dist
      |> Map.toSeq
      |> Seq.map (fun (k, v) -> Vector<float>.Build.Dense(List.toArray k), v)
      |> Seq.filter (fun (xi, _) -> interval.contains(base_model xi))
      |> Seq.toList

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
        let p = ParameterOps.from_vector v data.n_features_x initial.density
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

      let final_params = ParameterOps.from_vector result.MinimizingPoint data.n_features_x initial.density
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
  let n = 1000
  let p = 4
  let rng = Random(42)

  let x = Matrix<float>.Build.Random(n, p)
  let z = Vector<float>.Build.Dense(n, fun _ -> Math.Exp(rng.NextDouble()))
  let y = Vector<float>.Build.Dense(n, fun _ -> if rng.NextDouble() > 0.9 then 1.0 else 0.0)

  let data = { x = x; z = z; y = y }

  let base_model (xi: Vector<float>) =
    1.0 / (1.0 + Math.Exp(-xi.Sum() * 0.5))

  let target = {
    intervals = [{ lower = 0.0; upper = 1.0 }]
    expected_risks = [0.1]
    tolerance = 0.1
    x_distribution = Map.empty
  }

  match fit_constrained_model data target base_model with
  | Ok params' ->
    printfn "Success! beta_z: %f" params'.beta_z
  | Error e ->
    printfn "Failure: %s" e

run_example ()