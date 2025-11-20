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

let logInfo msg = printfn "[INFO] %s" msg
let logError msg = printfn "[ERROR] %s" msg

// =============================================================================
// DOMAIN TYPES
// =============================================================================

type Interval = {
  Lower: float
  Upper: float
} with
  member this.Contains(value: float) =
    value > this.Lower && value <= this.Upper

type TrainingData = {
  X: Matrix<float>
  Z: Vector<float>
  Y: Vector<float>
} with
  member this.NSamples = this.Y.Count
  member this.NFeaturesX = this.X.ColumnCount

type ConditionalDensity = {
  TauMean: Vector<float>
  Sigma: float
}

type Parameters = {
  Beta0: float
  BetaX: Vector<float>
  BetaZ: float
  Density: ConditionalDensity
}

type CalibrationTarget = {
  Intervals: Interval list
  ExpectedRisks: float list
  Tolerance: float
  XDistribution: Map<float list, float>
}

// =============================================================================
// LOGIC: Density & Math
// =============================================================================

module MathHelpers =
  let sigmoid (eta: float) =
    let clipped = Math.Max(-50.0, Math.Min(50.0, eta))
    1.0 / (1.0 + Math.Exp(-clipped))

  let logPdfTruncated (density: ConditionalDensity) (z: float) (x: Vector<float>) =
    let zSafe = Math.Max(z, 1e-10)
    
    let xAug = 
      Seq.append [1.0] x 
      |> Vector<float>.Build.DenseOfEnumerable

    let mu = density.TauMean * xAug
    
    let logZ = Math.Log(zSafe)
    let aStd = -mu / density.Sigma
    
    try
      let normDist = Normal(mu, density.Sigma)
      let pdfVal = normDist.Density(logZ)
      let survival = 1.0 - Normal.CDF(0.0, 1.0, aStd)
      
      Math.Log(pdfVal) - Math.Log(survival) - Math.Log(zSafe)
    with _ ->
      -1e10

  let sampleTruncated (density: ConditionalDensity) (x: Vector<float>) (nSamples: int) (rng: System.Random) =
    let xAug = 
      Seq.append [1.0] x 
      |> Vector<float>.Build.DenseOfEnumerable

    let mu = density.TauMean * xAug
    
    let samples = 
      Array.init nSamples (fun _ -> 
        let s = LogNormal.Sample(rng, mu, density.Sigma)
        Math.Max(1e-10, Math.Min(1e10, s))
      )
    Vector<float>.Build.Dense(samples)

// =============================================================================
// LOGIC: Parameter Conversion
// =============================================================================

module ParameterOps =
  
  let toVector (p: Parameters) : Vector<float> =
    seq {
      yield p.Beta0
      yield! p.BetaX
      yield p.BetaZ
    }
    |> Vector<float>.Build.DenseOfEnumerable

  let fromVector (v: Vector<float>) (nFeatX: int) (density: ConditionalDensity) : Parameters =
    {
      Beta0 = v.[0]
      BetaX = v.SubVector(1, nFeatX)
      BetaZ = v.[1 + nFeatX]
      Density = density
    }

// =============================================================================
// LOGIC: Fitting & Likelihood
// =============================================================================

module Fitting = 
  
  let fitConditionalDensity (X: Matrix<float>) (Z: Vector<float>) : Result<ConditionalDensity, string> =
    try
      let ZSafe = Z.Map(fun z -> Math.Max(z, 1e-10))
      let logZ = ZSafe.PointwiseLog()
      
      let ones = Vector<float>.Build.Dense(X.RowCount, 1.0)
      let XAug = X.InsertColumn(0, ones)
      
      let tauMean = XAug.Solve(logZ)
      let preds = XAug * tauMean
      let residuals = logZ - preds
      let sigma = Math.Max(residuals.StandardDeviation(), 0.01)
      
      logInfo (sprintf "Fitted density: sigma=%.4f" sigma)
      Ok { TauMean = tauMean; Sigma = sigma }
    with ex ->
      Error (sprintf "Failed to fit density: %s" ex.Message)

  let computeLogLikelihood (params': Parameters) (data: TrainingData) =
    try
      let eta = 
        (data.X * params'.BetaX) 
        + (data.Z * params'.BetaZ) 
        + params'.Beta0
      
      let logLikLogistic = 
        data.Y.Map2((fun y e -> 
          let eClipped = Math.Max(-50.0, Math.Min(50.0, e))
          y * eClipped - Math.Log(1.0 + Math.Exp(eClipped))
        ), eta).Sum()

      let densityContrib = 
        (data.X.EnumerateRows(), data.Z)
        ||> Seq.zip
        |> Seq.sumBy (fun (x, z) -> MathHelpers.logPdfTruncated params'.Density z x)

      logLikLogistic + densityContrib
    with _ -> 
      -1e10

  let predictRisk (params': Parameters) (x: Vector<float>) (z: float) =
    let eta = params'.Beta0 + (params'.BetaX * x) + (params'.BetaZ * z)
    MathHelpers.sigmoid eta

// =============================================================================
// LOGIC: Constraints
// =============================================================================

module Constraints =
  
  let computeExpectedRiskInInterval 
    (params': Parameters) 
    (interval: Interval) 
    (xDist: Map<float list, float>) 
    (baseModel: Vector<float> -> float) 
    (rng: System.Random) =
    
    let relevantX = 
      xDist 
      |> Map.toSeq
      |> Seq.map (fun (k, v) -> Vector<float>.Build.Dense(List.toArray k), v)
      |> Seq.filter (fun (x, _) -> interval.Contains(baseModel x))
      |> Seq.toList

    match relevantX with
    | [] -> 0.0
    | items ->
      let (totalRisk, totalWeight) = 
        items 
        |> List.fold (fun (accRisk, accWeight) (x, weight) ->
          let zSamples = MathHelpers.sampleTruncated params'.Density x 100 rng
          let avgRisk = 
            zSamples.Map(fun z -> Fitting.predictRisk params' x z).Mean()
          (accRisk + (avgRisk * weight), accWeight + weight)
        ) (0.0, 0.0)
      
      if totalWeight = 0.0 then 0.0 else totalRisk / totalWeight

  let evaluateConstraints (params': Parameters) (calib: CalibrationTarget) (baseModel) =
    let rng = System.Random(42)
    calib.Intervals
    |> List.indexed
    |> List.sumBy (fun (i, interval) ->
      let expected = computeExpectedRiskInInterval params' interval calib.XDistribution baseModel rng
      let target = calib.ExpectedRisks.[i]
      let tol = calib.Tolerance
      let upperViol = Math.Max(0.0, expected - (1.0 + tol) * target)
      let lowerViol = Math.Max(0.0, (1.0 - tol) * target - expected)
      (upperViol * upperViol) + (lowerViol * lowerViol)
    )

// =============================================================================
// OPTIMIZATION (Impure Shell)
// =============================================================================

module Optimization =
  
  let initializeParameters (data: TrainingData) (density: ConditionalDensity) =
    {
      Beta0 = 0.0
      BetaX = Vector<float>.Build.Dense(data.NFeaturesX, 0.0)
      BetaZ = 0.0
      Density = density
    }

  // Manual Central Difference Gradient to avoid 'Differentiate' dependency issues
  let private numericalGradient (f: float[] -> float) (x: float[]) : float[] =
    let n = x.Length
    let h = 1e-5
    let grad = Array.zeroCreate n
    
    // Need to mutate a copy to calculate differences
    let xMutable = Array.copy x
    
    for i in 0 .. n - 1 do
      let originalVal = x.[i]
      
      // f(x + h)
      xMutable.[i] <- originalVal + h
      let fPlus = f xMutable
      
      // f(x - h)
      xMutable.[i] <- originalVal - h
      let fMinus = f xMutable
      
      // Centered difference
      grad.[i] <- (fPlus - fMinus) / (2.0 * h)
      
      // Restore
      xMutable.[i] <- originalVal
      
    grad

  let optimize (data: TrainingData) (calib: CalibrationTarget) (baseModel) (initial: Parameters) =
    
    let penaltyWeight = 1000.0
    
    // Objective Function (Vector -> float)
    let objective (v: Vector<float>) =
      try
        let p = ParameterOps.fromVector v data.NFeaturesX initial.Density
        let nll = -(Fitting.computeLogLikelihood p data)
        let penalty = Constraints.evaluateConstraints p calib baseModel
        nll + (penalty * penaltyWeight) 
      with _ -> 
        Double.MaxValue

    try
      logInfo "Starting BFGS Optimization..."
      
      let startVec = ParameterOps.toVector initial
      
      // Gradient Adapter
      let gradient (v: Vector<float>) : Vector<float> = 
        let fArr (arr: float[]) = objective (Vector<float>.Build.DenseOfArray(arr))
        
        // Use local manual gradient
        let gradArr = numericalGradient fArr (v.ToArray())
        
        Vector<float>.Build.DenseOfArray(gradArr)

      let objFunc = ObjectiveFunction.Gradient(objective, gradient)
      let minimizer = BfgsMinimizer(1e-5, 1e-5, 1e-5, 500)
      let result = minimizer.FindMinimum(objFunc, startVec)
      
      let finalParams = ParameterOps.fromVector result.MinimizingPoint data.NFeaturesX initial.Density
      Ok finalParams
      
    with ex ->
      Error (sprintf "Optimization failed: %s" ex.Message)

// =============================================================================
// API & WORKFLOW
// =============================================================================

let fitConstrainedModel 
  (data: TrainingData) 
  (calib: CalibrationTarget) 
  (baseModel: Vector<float> -> float) =
  
  result {
    logInfo "Starting constrained model fitting..."
    let! density = Fitting.fitConditionalDensity data.X data.Z
    let initialParams = Optimization.initializeParameters data density
    let! finalParams = Optimization.optimize data calib baseModel initialParams
    logInfo "Model fitted successfully."
    return finalParams
  }

// =============================================================================
// USAGE EXAMPLE
// =============================================================================

let runExample () =
  let n = 1000
  let p = 4
  let rng = System.Random(42)
  
  let X = Matrix<float>.Build.Random(n, p)
  let Z = Vector<float>.Build.Dense(n, fun _ -> Math.Exp(rng.NextDouble()))
  let Y = Vector<float>.Build.Dense(n, fun _ -> if rng.NextDouble() > 0.9 then 1.0 else 0.0)
  
  let data = { X = X; Z = Z; Y = Y }
  
  let baseModel (x: Vector<float>) = 
    1.0 / (1.0 + Math.Exp(-x.Sum() * 0.5))

  let target = {
    Intervals = [{ Lower=0.0; Upper=1.0 }]
    ExpectedRisks = [0.1]
    Tolerance = 0.1
    XDistribution = Map.empty 
  }

  match fitConstrainedModel data target baseModel with
  | Ok params' -> 
    printfn "Success! BetaZ: %f" params'.BetaZ
  | Error e -> 
    printfn "Failure: %s" e

runExample()