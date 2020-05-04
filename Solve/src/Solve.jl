# Array shape conventions:
# d = vector dimension (6, usually)
# T = number of time points (~10000, usually)
# N = number of trajectories simulated
# 1. Vector time series: (d, T)
# 2. Ensemble: (d, T, N)
module Solve
export neutrongyro, he3gyro
export SpinSolution
export run_simulations

using Utils

using DifferentialEquations
using Distributed
using Statistics
using BenchmarkTools
using StaticArrays
using DSP
using Interpolations
using SpecialFunctions

const neutrongyro = -1.83247172e4
const he3gyro = -2.037894585e4

const crit_dress_params = Dict("B0"=>0.03, "B1"=>4.0596741e-1, "w"=>6283.18530718) 
# B in Gauss, w in rad/s

struct SpinSolution
    t::Array{Float64, 1} # T
    u::Array{Float64, 3} # d x T x N
end

function makenoise(rms, t, noiserate; filtercutoff=nothing, uppercutoff=nothing, filtertype=Elliptic(7, 1, 60))
    n = ceil(Int, t*noiserate)
    tarr = range(0, t, length=n)
    pts = rms*randn(n)
    if filtercutoff == nothing || filtercutoff == 0.0
        return scale(interpolate(pts, BSpline(Cubic(Line(OnGrid())))), tarr)
    else
        if uppercutoff == nothing
            filter = digitalfilter(Highpass(filtercutoff; fs=noiserate), filtertype)
        else
            filter = digitalfilter(Bandpass(filtercutoff, uppercutoff; fs=noiserate), filtertype)
        end
        return scale(interpolate(filt(filter, pts), BSpline(Cubic(Line(OnGrid())))), tarr)     
    end
end

function dS_dual(B0, B1, w, gyro1, gyro2; noise_itp=t->0, freq_mod=t->0, amp_mod=t->1)
    function dS(du, u, p, t)
        By = amp_mod(t)*B1*cos(w*t+freq_mod(t))+noise_itp(t)
        du[1] = gyro1 * (-u[3]*By)
        du[2] = gyro1 * (u[3]*B0)
        du[3] = gyro1 * (u[1]*By - u[2]*B0)
        du[4] = gyro2 * (-u[6]*By)
        du[5] = gyro2 * (u[6]*B0)
        du[6] = gyro2 * (u[4]*By - u[5]*B0)
    end
end

function run_simulations(tend, n; 
        B0=crit_dress_params["B0"], 
        B1=crit_dress_params["B1"],
        w=crit_dress_params["w"],
        noiseratio=1e-4,
        phase0=0,
        noiserate=5000, 
        nsave=0,
        saveat=[],
        saveinplane=false,
        downsample=1,
        phaseonly=false,
        filtercutoff=nothing,
        uppercutoff=nothing,
        filtertype=Elliptic(7, 1, 60),
        amp_mod=t->1,
        freq_mod=t->0)
    if filtercutoff == 0.0
        filtercutoff = nothing
    end
    Bnoise = B1*noiseratio
    u0nh = [0.0, 0.0, 1.0, 0.0, sin(phase0), cos(phase0)]
    tspan = (0.0, tend)
    
    if nsave == 0 && length(saveat) == 0 && !saveinplane
        error("Either nsave, saveat, or saveinplane must be specified.")
        return nothing
    end
    if nsave != 0
        saveat = range(tspan[1], tspan[2], length=nsave)
    end
    
    function prob_func(prob,i,repeat)
        # To create different noise instances for each run in a nonstochastic integration
        new_noise = makenoise(Bnoise, tspan[2], noiserate; 
            filtercutoff=filtercutoff, uppercutoff=uppercutoff, filtertype=filtertype)
        dS = dS_dual(B0, B1, w, neutrongyro, he3gyro; noise_itp=new_noise, amp_mod=amp_mod, freq_mod=freq_mod)
        ODEProblem(dS,u0nh,tspan)
    end
    function just_data_please(sol, i)
        (sol.t, hcat(sol.u...)), false
    end
    function phase_interpolate(t, phase)
        interpolate((t,), phase, Gridded(Linear()))(saveat)
    end

    prob = prob_func(nothing, 0, 0)
    cb = saveinplane ? ContinuousCallback((u,t,integ)->u[1],integ->nothing,save_positions=(true,false)) : nothing
    output_func = phaseonly ? (sol, i)->((sol.t, planephase(sol; smoothing=false)), false) : just_data_please
    ensembleprob = EnsembleProblem(prob, prob_func=prob_func, output_func=output_func)
    solution = solve(ensembleprob, Tsit5(), EnsembleThreads(), saveat=saveat, dense=false, callback=cb, 
        trajectories=n, abstol=1e-10, reltol=1e-10, maxiters=1e8, save_everystep=false)
    #u_sim = phaseonly ? transpose([phase_interpolate(pair[1], pair[2]) for pair=solution]) : [pair[2] for pair=solution]
    u_sim = phaseonly ? transpose([pair[2] for pair=solution]) : [pair[2] for pair=solution]
    if length(saveat) == 0
        saveat = solution[1][1]
    end
    return SpinSolution(saveat[1:downsample:end], cat(u_sim..., dims=3)[:,1:downsample:end,:])
end
end