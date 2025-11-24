# Constrained Maximum Likelihood Estimation (cMLE) in F#

A functional, research-oriented implementation of **Constrained Maximum Likelihood Estimation (cMLE)** for calibrated risk prediction using **F#** and **MathNet.Numerics**.

This project implements and extends the method proposed in:

> **Cao, Y., Ma, W., Zhao, G., McCarthy, A. M., & Chen, J. (2024).** > *A constrained maximum likelihood approach to developing well-calibrated models for predicting binary outcomes.* > Lifetime Data Analysis, 30, 624–648.  
> https://doi.org/10.1007/s10985-024-09628-9

Relevant articles:
- [My First F# Project: Implementing Constrained Optimization from Scratch](https://debugndiscover.netlify.app/posts/first_fsharp_project_cmle/)
- [Working Toward Robustness (F# c-MLE Project Part 2)](https://debugndiscover.netlify.app/posts/working-toward-robustness/)

---

## 1. Motivation: The Problem cMLE Solves

In applied risk modeling—especially in healthcare and population sciences—we often face a fundamental problem:

> **We want to improve an existing risk model using new predictors, but our available training data is not representative of the target population.**

Concretely:

* You already have a **base model** $\phi(X) = P(Y=1 \mid X)$ that is well-calibrated in a target population $P$.
* You collect new data from a **non-representative source population** $P_S$ that includes an additional predictor $Z$.
* If you fit a standard logistic regression on $(X, Z)$ using this biased sample, your resulting model will often be **miscalibrated** in the target population—especially in the tails.

This project implements a solution to that problem.

### What Goes Wrong Without cMLE?

Standard maximum likelihood estimation assumes $P_S(X, Z, Y) \approx P(X, Z, Y)$. But in real applications, $P_S(X) \neq P(X)$ (covariate shift), outcome prevalence differs, or both. As shown in the original paper, naive models often drastically misestimate risk in high-risk subgroups.

---

## 2. Core Idea of Constrained Maximum Likelihood

The main idea is:

> **Fit your new model by maximizing likelihood, but subject to explicit population-level calibration constraints derived from a trusted base model.**

We fit:

$$
g_\theta(X, Z) = P(Y = 1 \mid X, Z)
$$

But we enforce that its predictions agree with the base model $\phi(X)$ **on average** across risk strata in the target population.

### Calibration Constraints

Let:
* $\phi(X)$ = base model (well-calibrated in target population)
* $I_r = (a_r, b_r]$ = predefined risk intervals based on $\phi(X)$
* $\pi_r$ = true average risk in interval $I_r$ (from external data or base model)

We impose, for each interval:

$$
(1 - \delta)\,\pi_r \le 
\mathbb{E}\Big[g_\theta(X, Z) \mid \phi(X)\in I_r\Big]
\le (1 + \delta)\,\pi_r
$$

This forces the new model to remain aligned with the base model’s calibration in the target population.

---

## 3. How cMLE Works (Conceptually)

### Step 1 — Joint Likelihood

We maximize a joint likelihood over $Y$ and $Z$:

$$
\mathcal{L}(\theta, \tau, \sigma) =
\sum_i \Big[\log P_\theta(Y_i \mid X_i, Z_i) 
           + \log f_{\tau,\sigma}(Z_i \mid X_i) \Big]
$$

Where:
* $P_\theta(Y \mid X, Z)$ is logistic regression.
* $f_{\tau,\sigma}(Z \mid X)$ models the conditional density of $Z \mid X$.

---

### Step 2 — Deterministic Integration (CRN)

Following Cao et al. (2024), we model:

$$
\log(Z) \sim \mathcal{N}(\tau^T[1, X], \sigma^2), 
\quad \text{truncated to } (-\infty, 0]
$$

To ensure the objective function is smooth and differentiable for BFGS optimization, we use **Common Random Numbers (CRN)** via **Inverse Transform Sampling**.

Instead of resampling $Z$ randomly at every step (which creates a noisy/non-differentiable loss surface), we:
1. Pre-generate "frozen" uniform noise vectors $U \sim \text{Uniform}(0,1)$.
2. Map $U$ to truncated log-normal samples $Z$ deterministically inside the loss function.

This ensures that $\nabla_\theta J$ exists and is numerically stable.

---

### Step 3 — Penalty Scheduling (Method of Multipliers)

The constrained optimization:

$$
\max_\theta \ \mathcal{L}(\theta) \quad \text{s.t. calibration constraints}
$$

is solved using **Penalty Scheduling**. Instead of a single fixed $\lambda$, we solve a sequence of optimization problems where the penalty weight $\rho$ increases geometrically (e.g., $0.1 \to 0.5 \to \dots \to 7800$):

$$
\text{Minimize: } -\frac{1}{N}\mathcal{L}(\theta) + \rho_k \sum_r \max(0, \text{violation}_r)^2
$$

This allows the optimizer to find the "valley" of the likelihood function first, then gradually forces the parameters into the feasible region defined by the constraints.

---

## 4. Implementation Highlights

This project emphasizes **clarity, correctness, and structure**, not just numerical results.

### 4.1 Functional Architecture

* **Functional core, imperative shell**
  * Pure core: likelihood, constraints, density models
  * Impure shell: optimization, logging, I/O
* **Railway-Oriented Programming (ROP)**
  Uses a custom `Result` CE (`result { ... }`) to manage failures explicitly.

### 4.2 Key Modules

| Module | Responsibility |
|--------|----------------|
| `MathHelpers` | Robust Sigmoid, **Inverse Transform Sampling**, Log-PDF with soft-clipping |
| `ContextOps` | **Manages "frozen" noise vectors (CRN) for deterministic integration** |
| `Fitting` | Joint log-likelihood + prediction (Normalized objective) |
| `Constraints` | Risk interval calibration using **deterministic integration context** |
| `ParameterOps` | Vectorization and reconstruction (with soft-constraints for Sigma) |
| `Optimization` | **Penalty Scheduling Loop** + BFGS minimization |

---

## 5. Vectorization & The "Lucky Noise" Phenomenon

The project contains two implementations: `cmle_original.fs` (row-wise) and `cmle_vectorized.fs` (matrix-based).

### 5.1 Performance
The vectorized implementation (`cmle_vectorized.fs`) flattens the integration context into a single matrix ($N_{obs} \times N_{samples}$). This allows the CPU/BLAS provider to compute risks for thousands of integration points via SIMD operations.
* **Speedup:** ~2x faster (e.g., 52s $\to$ 28s).
* **Consistency:** Uses `MersenneTwister` for cross-platform reproducibility (macOS/Linux/Windows).

### 5.2 The "Overfitting to Constraint" Trap
Users may notice that the original implementation often achieved **0.0000** constraint violation while maintaining high accuracy. **This was likely a statistical anomaly.**

Because the original code used `System.Random` with a small integration sample size ($N=100$), it is possible to find a specific seed where the integration noise "cancels out" perfectly inside the calibration bins. This allowed the optimizer to satisfy tight tolerances ($10^{-4}$) without sacrificing the predictive power of $Z$.

**The Reality (revealed by Vectorization):**
With a robust RNG (`MersenneTwister`) and matrix-based sampling, the "natural integration error" is visible (approx 2-4%).
* If we force the optimizer to achieve **0.00%** violation, it will "crush" the $\beta_Z$ coefficient to zero.
* Why? Because if $\beta_Z \approx 0$, the model mimics the Base Model perfectly, satisfying the constraint but destroying predictive power.

### 5.3 The Robust Solution
The vectorized implementation solves this via:
1.  **Warm Start:** Fits an unconstrained MLE first to find the "true" $\beta_Z$ (e.g., ~1.0).
2.  **Relaxed Schedule:** Caps the penalty weight (`rho`) earlier.
3.  **Statistical Tolerance:** Accepts a ~2.5% violation as natural noise, preserving the causal signal of $Z$.

**Diagnostic Table (Vectorized Run):**
```text
ITER  | RHO        | BETA_Z     | BETA_0     | NLL        | MAX_VIOL
----------------------------------------------------------------------
1     | 0.1        | 1.5640     | -2.6476    | -1.28153   | 0.170927
...
6     | 3.2        | 1.0216     | -2.7727    | -1.22727   | 0.038044
```

*Result: The model recovers the true Beta\_Z (1.0) while accepting a 3.8% calibration deviation.*

-----

## 6\. How to Use

```fsharp
let result =
  fit_constrained_model
    trainingData        // from biased source population
    calibrationTarget   // built using target population X-distribution
    baseModelFunction   // φ(X)

match result with
| Ok params ->
    printfn "Converged. Beta_z = %.4f" params.beta_z
| Error e ->
    printfn "Optimization failed: %s" e
```

### CalibrationTarget

```fsharp
type CalibrationTarget = {
  intervals: Interval list
  expected_risks: float list
  tolerance: float
  x_distribution: (Vector<float> * float) list
}
```

  * `intervals` → bins based on base model risk
  * `expected_risks` → population-level true risks per bin ($P_r^e$)
  * `x_distribution` → empirical distribution of $X$ in target population

-----

## 7\. Future Directions

  * **Automatic Differentiation:** Replacing finite differences with DiffSharp or similar to improve gradient precision at high penalty weights.
  * **Alternative Density Models:** Support for non-log-normal $f(Z \mid X)$.
  * **Comparison:** Benchmarking against post-hoc calibration methods (Platt scaling, isotonic regression).
