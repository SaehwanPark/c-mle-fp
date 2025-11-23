===
Original version (cmle.fs)
===
[INFO] Fitted Z|X density: sigma = 0.1718
[INFO] Schedule Iteration 1: rho = 0.1
[INFO]   -> Max Constraint Violation: 0.162976
[INFO] Schedule Iteration 2: rho = 0.5
[INFO]   -> Max Constraint Violation: 0.105362
[INFO] Schedule Iteration 3: rho = 2.5
[INFO]   -> Max Constraint Violation: 0.042357
[INFO] Schedule Iteration 4: rho = 12.5
[INFO]   -> Max Constraint Violation: 0.013667
[INFO] Schedule Iteration 5: rho = 62.5
[INFO]   -> Max Constraint Violation: 0.004026
[INFO] Schedule Iteration 6: rho = 312.5
[INFO]   -> Max Constraint Violation: 0.000806
[INFO] Schedule Iteration 7: rho = 1562.5
[INFO]   -> Max Constraint Violation: 0.000164
[INFO] Schedule Iteration 8: rho = 7812.5
[INFO]   -> Max Constraint Violation: 0.000000
[INFO] Constraints satisfied.
Success! Beta_Z: 0.9778 (True ~1.0)
Density Sigma: 0.3661

===
Vectorized version (cmle_vectorized.fs)
===
[INFO] Fitted Z|X density: sigma = 0.1844
[INFO] Running Warm Start (Unconstrained MLE)...
[INFO] Warm Start Complete. Beta_Z: 1.6792 | Beta_0: -2.6366

ITER  | RHO        | BETA_Z     | BETA_0     | NLL        | MAX_VIOL
----------------------------------------------------------------------
1     | 0.1        | 1.5640     | -2.6476    | -1.28153   | 0.170927
2     | 0.2        | 1.4831     | -2.6572    | -1.27799   | 0.148419
3     | 0.4        | 1.3747     | -2.6747    | -1.27029   | 0.118702
4     | 0.8        | 1.2549     | -2.7057    | -1.25771   | 0.087422
5     | 1.6        | 1.1337     | -2.7420    | -1.24223   | 0.059833
6     | 3.2        | 1.0216     | -2.7727    | -1.22727   | 0.038044
[INFO] Constraints satisfied.
Success! Beta_Z: 1.0216 (True ~1.0)
Density Sigma: 0.3512