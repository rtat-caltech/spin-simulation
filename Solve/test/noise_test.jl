using Statistics

function arrayapprox(a, b, atol=1e-10)
    for (x, y)=zip(a, b)
        if !isapprox(x, y, atol=atol)
            return false
        end
    end
    return true
end

@testset "Basic functionality" begin
    noiserate=1000
    tend=1
    rms=0.1
    n = NoiseIterator(rms, tend, noiserate)
    t = range(0, tend, step=1.0/noiserate)
    r = rand(1000).*tend
    prev = nothing
    for samp=Iterators.take(n, 3)
        samp_t = samp.(t)
        samp_r = samp.(r)
        @test std(samp_t) <= rms*1.05
        @test std(samp_t) >= rms*0.95
        @test std(samp_r) <= rms*1.2
        @test std(samp_r) >= rms*0.8
        @test prev==nothing || !arrayapprox(prev, samp_t)
        prev = samp_t
    end
end;

@testset "Repeating" begin
    noiserate=100
    tend=1
    rms=0.1
    rep=3
    n = NoiseIterator(rms, tend, noiserate, repeat=rep)
    t = range(0, tend, step=1.0/noiserate)
    prev = nothing
    ctr = 0
    for samp=Iterators.take(n, 10)
        samp_t = samp.(t)
        @test std(samp_t) <= rms*1.5
        @test std(samp_t) >= rms*0.5
        if ctr % rep != 0
            @test arrayapprox(prev, samp_t)
        else
            @test ctr == 0 || !arrayapprox(prev, samp_t)
        end
        prev = samp_t
        ctr += 1
    end
end;
