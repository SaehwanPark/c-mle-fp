# Constrained Maximum Likelihood Estimation (cMLE) in F#

A functional, research-oriented implementation of **Constrained Maximum Likelihood Estimation (cMLE)** for calibrated risk prediction using **F#** and **MathNet.Numerics**.

This project implements and extends the method proposed in:

> **Cao, Y., Ma, W., Zhao, G., McCarthy, A. M., & Chen, J. (2024).** > *A constrained maximum likelihood approach to developing well-calibrated models for predicting binary outcomes.* > Lifetime Data Analysis, 30, 624–648.  
> https://doi.org/10.1007/s10985-024-09628-9

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

---

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

### 4.3 Optimization Setup

* Uses `MathNet.Numerics.Optimization.BfgsMinimizer`
* **Penalty Scheduling:** Iterative loop that hardens constraints until violation tolerance ($< 10^{-4}$) is met.
* **Gradients:** Computed via central finite differences on the deterministic objective.
* **Numerical Stability:** Inputs to logs and exponentials are soft-clipped to prevent NaNs during line search.

---

## 5. Why F#?

This project is also a language experiment.

F# provides:

* Strong static typing → fewer silent modeling bugs  
* Algebraic domain modeling → constraints encoded structurally  
* Expression-oriented programming → natural for statistical pipelines  
* Functional-first architecture → better separation of concerns  

This is particularly important because cMLE is *structural*, not just numerical.

---

## 6. How to Use

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
