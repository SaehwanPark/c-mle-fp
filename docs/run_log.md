Original

```
❯ time dotnet run
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
dotnet run  111.43s user 2.66s system 102% cpu 1:51.66 total
```

Vectorized

