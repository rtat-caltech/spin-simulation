# Array shape conventions:
# d = vector dimension (6, usually)
# T = number of time points (~10000, usually)
# N = number of trajectories simulated
# 1. Vector time series: (d, T)
# 2. Ensemble: (d, T, N)
module Solve
export neutrongyro, he3gyro, crit_params
export run_simulations
export NoiseIterator
export makenoise
export daq_noise_iterator

using Utils

using DifferentialEquations
using Distributed
using Statistics
using BenchmarkTools
using StaticArrays
using DSP
using Interpolations
using SpecialFunctions
using IterTools
using MAT

const neutrongyro = -1.83247172e4
const he3gyro = -2.037894585e4

#const crit_params = Dict("B0"=>0.03, "B1"=>4.0596741e-1, "w"=>6283.18530718)
#const crit_params = Dict("B0"=>0.073, "B1"=>3.9739024591536093e-1, "w"=>6283.18530718)
const crit_params = Dict("B0"=>0.050, "B1"=>4.0289106454828055e-1, "w"=>6283.18530718)
# B in Gauss, w in rad/s

include("noisegeneration.jl")

function dS_dual(B0, B1::Real, w, gyro1, gyro2; noise_itp=t->0)
    dS_dual(B0, t->B1*cos(w*t), w, gyro1, gyro2; noise_itp=noise_itp)
end

function dS_dual(B0, B1, w, gyro1, gyro2; noise_itp=t->0)
    Bxfunc = t -> B1(t)+noise_itp(t)
    function dS(du, u, p, t)
        Bx = Bxfunc(t)
        By = 0
        Bz = B0
        du[1] = gyro1 * (u[2]*Bz - u[3]*By)
        du[2] = gyro1 * (u[3]*Bx - u[1]*Bz)
        du[3] = gyro1 * (u[1]*By - u[2]*Bx)
        du[4] = gyro2 * (u[5]*Bz - u[6]*By)
        du[5] = gyro2 * (u[6]*Bx - u[4]*Bz)
        du[6] = gyro2 * (u[4]*By - u[5]*Bx)
    end
end

function dS_signal(B0, B1::Real, w, gyro1, gyro2; noise_itp=t->0)
    dS_signal(B0, t->B1*cos(w*t), w, gyro1, gyro2; noise_itp=noise_itp)
end

function dS_signal(B0, B1, w, gyro1, gyro2; noise_itp=t->0)
    Bxfunc = t -> B1(t)+noise_itp(t)
    function dS(du, u, p, t)
        Bx = Bxfunc(t)
        By = 0
        Bz = B0
        du[1] = gyro1 * (u[2]*Bz - u[3]*By)
        du[2] = gyro1 * (u[3]*Bx - u[1]*Bz)
        du[3] = gyro1 * (u[1]*By - u[2]*Bx)
        du[4] = gyro2 * (u[5]*Bz - u[6]*By)
        du[5] = gyro2 * (u[6]*Bx - u[4]*Bz)
        du[6] = gyro2 * (u[4]*By - u[5]*Bx)
        anorm = u[1]^2 + u[2]^2 + u[3]^2
        bnorm = u[4]^2 + u[5]^2 + u[6]^2
        du[7] = (u[1]*u[4]+u[2]*u[5]+u[3]*u[6])/sqrt(anorm*bnorm)
    end
end

function state_from_phases(phi1, phi2)
    [cos(phi1), sin(phi1), 0.0, cos(phi2), sin(phi2), 0.0]
end

function make_stateful(arg, type)
    if arg isa type
        return Iterators.Stateful(Iterators.repeated(arg))
    else
        return Iterators.Stateful(arg)
    end
end

function preprocess(solu, downsample)
    return normalize(hcat(solu...)[:,1:downsample:end])
end

function run_simulations(tend, n; 
                         B0=crit_params["B0"], 
                         B1=crit_params["B1"],
                         B1_func=nothing,
                         w=crit_params["w"],
                         Bnoise=crit_params["B1"]*1e-4,
                         lowercutoff=0.0,
                         uppercutoff=0.0,
                         filtertype=Elliptic(7, 1, 60),
                         noiseiterator=nothing,
                         noiserate=5000,
                         initial_phases=(0.0,0.0),
                         nsave=0,
                         saveat=[],
                         saveinplane=false,
                         downsample=1,
                         output_type="bloch",
                         compute_signal=false)

    tspan = (0.0, tend)
    
    if nsave == 0 && length(saveat) == 0 && !saveinplane
        error("Either nsave, saveat, or saveinplane must be specified.")
        return nothing
    end
    if nsave != 0
        saveat = range(tspan[1], tspan[2], length=nsave)
    end
    B0_iterator = make_stateful(B0, Real)
    B1_iterator = make_stateful(B1, Real)
    phaseiterator = make_stateful(initial_phases, Tuple)
    
    if noiseiterator == nothing
        noiseiterator = NoiseIterator(Bnoise, tspan[2]-tspan[1], noiserate; lowercutoff=lowercutoff, uppercutoff=uppercutoff, filtertype=filtertype)
    end
    noiseiterator = Iterators.Stateful(noiseiterator)
    
    info_array = []
    function prob_func(prob,i,repeat)
        B0 = popfirst!(B0_iterator)
        B1 = popfirst!(B1_iterator)
        new_noise = popfirst!(noiseiterator)
        B1_param = B1_func == nothing ? B1 : t->B1*B1_func(t)
        if compute_signal
            dS = dS_signal(B0, B1_param, w, neutrongyro, he3gyro; noise_itp=new_noise)
            u0nh = push!(state_from_phases(popfirst!(phaseiterator)...), 0)
        else
            dS = dS_dual(B0, B1_param, w, neutrongyro, he3gyro; noise_itp=new_noise)
            u0nh = state_from_phases(popfirst!(phaseiterator)...)            
        end
        ODEProblem(dS,u0nh,tspan)
    end

    #prob = prob_func(nothing, 0, 0)
    prob = nothing
    cb = saveinplane ? ContinuousCallback((u,t,integ)->u[3],integ->nothing,save_positions=(true,false)) : nothing

    if output_type == "bloch"
        output_func = (sol, i)->((sol.t, preprocess(sol.u, downsample)), false)
    elseif output_type == "phase"
        output_func = (sol, i)->((sol.t, planephase(preprocess(sol.u, downsample))), false)
    elseif output_type == "signal"
        output_func = (sol, i)->((sol.t, signal(preprocess(sol.u, downsample))), false)
    elseif output_type == "signal_int"
        if !compute_signal
            throw(ArgumentError("In order to compute integrated signal, compute_signal must be set to true"))
        end
        output_func = (sol, i)->((sol.t, preprocess(sol.u, downsample)[7,:,:]), false)
    end

    ensembleprob = EnsembleProblem(prob, prob_func=prob_func, output_func=output_func)
    solution = solve(ensembleprob, Tsit5(), EnsembleThreads(), saveat=saveat, dense=false, callback=cb, 
                     trajectories=n, abstol=1e-10, reltol=1e-10, maxiters=1e8, save_everystep=false)
    
    u_sim = output_type == "bloch" ? [pair[2] for pair=solution] : transpose([pair[2] for pair=solution])
    if length(saveat) == 0
        saveat = solution[1][1]
    end
    
    return SpinSolution(saveat[1:downsample:end], cat(u_sim..., dims=3))
end
end
