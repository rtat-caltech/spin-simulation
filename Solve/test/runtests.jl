using Solve
using Test

tests = ["noise_test", "solve_test"]

for t in tests
    include("$(t).jl")
end
