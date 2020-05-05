@testset "Sanity Check" begin
    tend = 0.1
    B0 = 0.03
    nsave = 101
    phase0 = pi/4
    sol = run_simulations(tend, 1; B0=B0, B1=0.0, noiseratio=0.0, nsave=nsave)
    @test length(sol.t) == nsave
    sf = sol.u[:, nsave, 1]
    theta1 = neutrongyro*B0
    theta2 = he3gyro*B0+phase0
    sf_pred = []
end;
