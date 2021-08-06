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
export filterednoise, spatialnoise, daqnoise

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
using Base.Iterators
using MAT

const neutrongyro = -1.83247172e4
const he3gyro = -2.037894585e4

#const crit_params = Dict("B0"=>0.03, "B1"=>4.0596741e-1, "w"=>6283.18530718)
#const crit_params = Dict("B0"=>0.073, "B1"=>3.9739024591536093e-1, "w"=>6283.18530718)
const crit_params = Dict("B0"=>0.050, "B1"=>4.0289106454828055e-1, "w"=>6283.18530718)
# B in Gauss, w in rad/s

include("noisegeneration.jl")

function dS_general(B0, B1, w, gyro1, gyro2, B1func, Bxfunc, Byfunc, Bzfunc)
    function dS(du, u, p, t)
        Bx = Bxfunc(t) + B1*cos(w*t) + B1func(t)
        By = Byfunc(t)
        Bz = Bzfunc(t) + B0
        du[1] = gyro1 * (u[2]*Bz - u[3]*By)
        du[2] = gyro1 * (u[3]*Bx - u[1]*Bz)
        du[3] = gyro1 * (u[1]*By - u[2]*Bx)
        du[4] = gyro2 * (u[5]*Bz - u[6]*By)
        du[5] = gyro2 * (u[6]*Bx - u[4]*Bz)
        du[6] = gyro2 * (u[4]*By - u[5]*Bx)
    end
end

function dS_general_signal(B0, B1, w, gyro1, gyro2, B1func, Bxfunc, Byfunc, Bzfunc)
    function dS(du, u, p, t)
        Bx = Bxfunc(t) + B1*cos(w*t) + B1func(t)
        By = Byfunc(t)
        Bz = Bzfunc(t) + B0
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


function state_from_phases(phi1, phi2, theta1, theta2)
    [cos(phi1)*sin(theta1), sin(phi1)*sin(theta1), cos(theta1),
     cos(phi2)*sin(theta2), sin(phi2)*sin(theta2), cos(theta2)]
end

function make_stateful(arg, type)
    if arg isa type
        return Iterators.Stateful(repeated(arg))
    else
        return Iterators.Stateful(arg)
    end
end

function postprocess(solu)
    return normalize(hcat(solu...))
end

function run_simulations(tend, n;
                         B0=crit_params["B0"], 
                         B1=crit_params["B1"],
                         w=crit_params["w"],
                         B1func=t->0,
                         Bxfuncs=repeated(t->0),
                         Byfuncs=repeated(t->0),
                         Bzfuncs=repeated(t->0),
                         Bfuncs=nothing,
                         initial_phases=(0.0,0.0),
                         initial_latitudes=(pi/2, pi/2),
                         nsave=0,
                         saveat=[],
                         saveinplane=false,
                         output_type="bloch",
                         compute_signal=false,
                         solver_alg=Vern9())

    tspan = (0.0, tend)
    
    if nsave == 0 && length(saveat) == 0 && !saveinplane
        error("Either nsave, saveat, or saveinplane must be specified.")
        return nothing
    end
    if nsave != 0
        saveat = range(tspan[1], tspan[2], length=nsave)
    end

    Bxfuncs = Iterators.Stateful(Bxfuncs)
    Byfuncs = Iterators.Stateful(Byfuncs)
    Bzfuncs = Iterators.Stateful(Bzfuncs)
    if Bfuncs != nothing
        Bfuncs = Iterators.Stateful(Bfuncs)
    end
    
    function prob_func(prob,i,repeat)
        if Bfuncs == nothing
            Bxfunc = popfirst!(Bxfuncs)
            Byfunc = popfirst!(Byfuncs)
            Bzfunc = popfirst!(Bzfuncs)
        else
            Bxfunc, Byfunc, Bzfunc = popfirst!(Bfuncs)
        end
        if compute_signal
            dS = dS_general_signal(B0, B1, w, neutrongyro, he3gyro, B1func, Bxfunc, Byfunc, Bzfunc)
            u0nh = push!(state_from_phases(initial_phases..., initial_latitudes...), 0)
        else
            dS = dS_general(B0, B1, w, neutrongyro, he3gyro, B1func, Bxfunc, Byfunc, Bzfunc)
            u0nh = state_from_phases(initial_phases..., initial_latitudes...)
        end
        ODEProblem(dS,u0nh,tspan)
    end

    prob = nothing
    cb = saveinplane ? ContinuousCallback((u,t,integ)->u[3],integ->nothing,save_positions=(true,false)) : nothing
    
    if output_type == "bloch"
        output_func = (sol, i)->((sol.t, postprocess(sol.u)), false)
    elseif output_type == "phase"
        output_func = (sol, i)->((sol.t, planephase(postprocess(sol.u))), false)
    elseif output_type == "signal"
        output_func = (sol, i)->((sol.t, signal(postprocess(sol.u))), false)
    elseif output_type == "signal_int"
        if !compute_signal
            throw(ArgumentError("In order to compute integrated signal, compute_signal must be set to true"))
        end
        output_func = (sol, i)->((sol.t, postprocess(sol.u)[7,:,:]), false)
    end

    ensembleprob = EnsembleProblem(prob, prob_func=prob_func, output_func=output_func)
    solution = solve(ensembleprob, solver_alg, EnsembleThreads(), saveat=saveat, dense=false, callback=cb, 
                     trajectories=n, abstol=1e-10, reltol=1e-10, maxiters=1e8, save_everystep=false)
    
    u_sim = output_type == "bloch" ? [pair[2] for pair=solution] : transpose([pair[2] for pair=solution])
    if length(saveat) == 0
        saveat = solution[1][1]
    end
    return SpinSolution(saveat, cat(u_sim..., dims=3))
end
end
