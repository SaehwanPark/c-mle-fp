open System.Diagnostics

printfn "==="
printfn "Original version (cmle.fs)\n==="
let stopwatch = Stopwatch.StartNew()
CMLE.Original.run_example()
stopwatch.Stop()
printfn $"Time elapsed: {stopwatch.Elapsed.Seconds} sec\n"

printfn "==="
printfn "Vectorized version (cmle_vectorized.fs)\n==="
stopwatch.Restart()
CMLE.Vectorized.run_example()
stopwatch.Stop()
printfn $"Time elapsed: {stopwatch.Elapsed.Seconds} sec"