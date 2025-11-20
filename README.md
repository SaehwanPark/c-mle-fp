# Constrained Maximum Likelihood Estimation (c-MLE) in F#

A functional programming implementation of Constrained Maximum Likelihood Estimation (c-MLE). 

This project demonstrates how to train a logistic regression model that incorporates a new predictor ($Z$) while strictly adhering to calibration constraints derived from an existing base model (trained on $X$). It utilizes **F#** and **MathNet.Numerics** to solve the optimization problem using a functional, railway-oriented architecture.

## 1. Mathematical Formulation

The goal is to fit a new model $P(Y=1 | X, Z)$ parameterized by $\theta = (\beta_0, \beta_x, \beta_z)$ such that it maximizes the likelihood of the observed data, subject to the constraint that the new model's average risk matches the base model's risk within specific sub-populations (intervals).

### The Objective
We maximize the log-likelihood $\mathcal{L}(\theta)$:

$$
\max_{\theta} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]
$$

Where $p_i = \sigma(\beta_0 + \beta_x^T x_i + \beta_z z_i)$ and $\sigma(\cdot)$ is the sigmoid function.

### The Constraints
We impose constraints on specific risk intervals $I_k$ defined by the base model. For a given interval, the expected risk of the new model must match the expected risk of the target (base) model within a tolerance $\delta$:

$$
(1 - \delta) \pi_k \le \mathbb{E}_{X \in I_k} \left[ \mathbb{E}_{Z|X} [ P(Y=1 | X, Z) ] \right] \le (1 + \delta) \pi_k
$$

Where:
* $\pi_k$ is the expected risk of the base model in interval $k$.
* Since the constraints are defined over the population $X$ (where $Z$ might be unobserved or variable), we compute the inner expectation by integrating over $Z$ using a conditional density model $f(z|x)$.

### The Auxiliary Model
To evaluate the constraints, we model the conditional density of the new predictor $Z$ given $X$ as a **Truncated Log-Normal** distribution:

$$
\log(Z) \sim \mathcal{N}(\tau^T [1, X], \sigma^2) \quad \text{s.t.} \quad Z > 0
$$

This allows us to perform Monte Carlo integration to estimate the expected risk for any given $x$ vector during the optimization process.

---

## 2. Implementation Details

This repository uses **F#** to strictly separate domain logic (pure functions) from optimization mechanics (impure shell).

### Architecture
* **Railway-Oriented Programming (ROP):** Uses a `Result` computation expression (`result { ... }`) to handle errors and control flow without exceptions or side effects.
* **Functional Core, Impure Shell:**
    * **Core:** Likelihood calculation, probability density evaluation, and constraint checking are implemented as pure, immutable functions.
    * **Shell:** The actual optimization loop (BFGS) and logging reside at the boundaries of the application.

### Optimization Strategy
Standard gradient-based solvers (like BFGS) do not natively support complex non-linear constraints. We solve this using the **Penalty Method**:

1.  **Objective Modification:** The constrained problem is converted into an unconstrained one by adding a penalty term to the negative log-likelihood:
    $$J(\theta) = -\mathcal{L}(\theta) + \lambda \sum_{k} \text{max}(0, \text{violation}_k)^2$$
2.  **Numerical Gradient:** We implement a Finite Difference (Central Difference) gradient calculator to allow the BFGS algorithm to navigate the loss landscape without requiring analytical derivatives for the complex constraint integrals.
3.  **Solver:** We utilize `MathNet.Numerics.Optimization.BfgsMinimizer`.

## 3. Getting Started

### Prerequisites
* .NET SDK (6.0 or later recommended)
* MathNet.Numerics

### Installation
1.  Clone the repo:
    ```bash
    git clone [https://github.com/SaehwanPark/c-mle-fp.git](https://github.com/SaehwanPark/c-mle-fp.git)
    ```
2.  Install dependencies:
    ```bash
    dotnet add package MathNet.Numerics
    ```

### Usage
The main entry point is `fitConstrainedModel`. See `cmle.fs` for the complete workflow example.

```fsharp
// Example Workflow
let result = 
    fitConstrainedModel 
        trainingData 
        calibrationTarget 
        baseModelFunction

match result with
| Ok params -> printfn "Model converged. Beta_z: %f" params.BetaZ
| Error msg -> printfn "Optimization failed: %s" msg