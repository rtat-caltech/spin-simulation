@testset "Sanity Check" begin
    tend = 0.1
    B0 = 0.03
    nsave = 101
    phase0 = pi/4
    sol = run_simulations(tend, 1; B0=B0, B1=0.0, phase0=phase0, noiseratio=0.0, nsave=nsave)
    @test length(sol.t) == nsave
    sf = sol.u[:, nsave, 1]
    theta1 = -neutrongyro*B0*tend
    theta2 = -he3gyro*B0*tend+phase0
    sf_pred = [cos(theta1), sin(theta1), 0, cos(theta2), sin(theta2), 0]
    @test all(isapprox.(sf, sf_pred, atol=1e-9))
end;

@testset "Noise Test" begin
    tend = 0.1
    # Make a "noise" that's actually just a static B-field in the x direction
    Bnoise = 0.04
    noiseiterator = Iterators.repeated(t->Bnoise)
    
    nsave = 101
    phase0 = pi/2
    sol = run_simulations(tend, 1; B=0.0, B1=0.0, phase0=phase0, noiseiterator=noiseiterator, nsave=nsave)
    sf = sol.u[:, nsave, 1]
    theta = -he3gyro*Bnoise*tend
    sf_pred = [1, 0, 0, 0, cos(theta), sin(theta)]
    @test all(isapprox.(sf, sf_pred, atol=1e-9))
end;
