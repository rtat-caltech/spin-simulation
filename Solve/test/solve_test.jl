@testset "Sanity Check" begin
    tend = 0.1
    B0 = 0.03
    nsave = 101
    phase0 = pi/4
    sol = run_simulations(tend, 1; B0=B0, B1=0.0, initial_phases=(0, phase0), Bnoise=0.0, nsave=nsave)
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
    sol = run_simulations(tend, 1; B0=0.0, B1=0.0, initial_phases=(0, pi/2), noiseiterator=noiseiterator, nsave=nsave)
    sf = sol.u[:, nsave, 1]
    theta = -he3gyro*Bnoise*tend
    sf_pred = [1, 0, 0, 0, cos(theta), sin(theta)]
    @test all(isapprox.(sf, sf_pred, atol=1e-9))
end;

@testset "Phase Test" begin
    tend = 0.01
    sol = run_simulations(0.01, 3; initial_phases=(pi/4, pi/2), nsave=11)
    expected=[1/sqrt(2), 1/sqrt(2), 0, 0, 1, 0]
    @test all(isapprox.(expected, sol.u[:,1,1], atol=1e-10))
    @test all(isapprox.(expected, sol.u[:,1,2], atol=1e-10))    
    @test all(isapprox.(expected, sol.u[:,1,3], atol=1e-10))

    p =[ (0, pi/2) , (pi/3, -pi/4) , (pi, pi/3)]
    sol = run_simulations(0.01, 3; initial_phases=p, nsave=11)
    for (i, pair) = enumerate(p)
        phi1, phi2 = pair
        expected=[cos(phi1), sin(phi1), 0, cos(phi2), sin(phi2), 0]
        @test all(isapprox.(expected, sol.u[:,1,i], atol=1e-10))
    end
end

@testset "Sanity Check" begin
    tend = 0.1
    B0 = 0.03
    nsave = 101
    phase0 = pi/4
    sol = run_simulations(tend, 1; B0=B0, B1=0.0, initial_phases=(0, phase0), Bnoise=0.0, nsave=nsave)
    @test length(sol.t) == nsave
    sf = sol.u[:, nsave, 1]
    theta1 = -neutrongyro*B0*tend
    theta2 = -he3gyro*B0*tend+phase0
    sf_pred = [cos(theta1), sin(theta1), 0, cos(theta2), sin(theta2), 0]
    @test all(isapprox.(sf, sf_pred, atol=1e-9))
end;

@testset "Parameter Iteration" begin
    tend = 0.1
    B0 = [0.03, -0.01, 0.1]
    nsave = 101
    phase0 = pi/4
    sol = run_simulations(tend, 3; B0=B0, B1=0.0, initial_phases=(0, phase0), Bnoise=0.0, nsave=nsave)
    for i=1:3
        sf = sol.u[:, nsave, i]
        theta1 = -neutrongyro*B0[i]*tend
        theta2 = -he3gyro*B0[i]*tend+phase0
        sf_pred = [cos(theta1), sin(theta1), 0, cos(theta2), sin(theta2), 0]
        @test all(isapprox.(sf, sf_pred, atol=1e-8))
    end

    tend = pi/20 # Just wanted to pick something that's not a multiple of w
    B1 = [0.4, 0.0, -0.3]
    w = 2*pi*1000
    phase1 = pi/3
    phase2 = pi/4
    sol = run_simulations(tend, 3; B0=0.0, B1=B1, w=w, initial_phases=(phase1, phase2), Bnoise=0.0, nsave=nsave)
    for i=1:3
        sf = sol.u[:, nsave, i]
        phi1 = -neutrongyro*B1[i]*sin(w*tend)/w
        theta1 = phase1
        phi2 = -he3gyro*B1[i]*sin(w*tend)/w
        theta2 = phase2
        sf_pred = [cos(theta1), sin(theta1)*cos(phi1), sin(theta1)*sin(phi1),
                   cos(theta2), sin(theta2)*cos(phi2), sin(theta2)*sin(phi2)]
        @test all(isapprox.(sf, sf_pred, atol=1e-8))
    end
end
