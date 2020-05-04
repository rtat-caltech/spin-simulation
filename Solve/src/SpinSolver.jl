# Array shape conventions:
# d = vector dimension (6, usually)
# T = number of time points (~10000, usually)
# N = number of trajectories simulated
# 1. Vector time series: (d, T)
# 2. Ensemble: (d, T, N)

using DifferentialEquations
using DiffEqBase.EnsembleAnalysis
using Plots
pyplot()
using LinearAlgebra
using Distributed
using Statistics
using BenchmarkTools
using StaticArrays
using DSP
using Interpolations
using JLD2
using LaTeXStrings
using Dates
using SpecialFunctions
using FFTW

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

function dS_func(B0, B1, w, gyro)
    function dS(du, u, p, t)
        By = B1*sin(w*t)
        du[1] = gyro * (-u[3]*By)
        du[2] = gyro * (u[3]*B0)
        du[3] = gyro * (u[1]*By - u[2]*B0)
    end
end

function dW_func(noiserms, gyro, bandwidth)
    g = gyro * noiserms/sqrt(bandwidth)
    function dW(du, u, p, t)
        du[1] = u[3] * -g
        du[2] = 0
        du[3] = u[1] * g
    end
end

function dS_dual(B0, B1, w, gyro1, gyro2)
    function dS(du, u, p, t)
        By = B1*cos(w*t)
        du[1] = gyro1 * (-u[3]*By)
        du[2] = gyro1 * (u[3]*B0)
        du[3] = gyro1 * (u[1]*By - u[2]*B0)
        du[4] = gyro2 * (-u[6]*By)
        du[5] = gyro2 * (u[6]*B0)
        du[6] = gyro2 * (u[4]*By - u[5]*B0)
    end
end

function B_test(B1, w; noise_itp=t->0, freq_mod=t->0, amp_mod=t->1)
    function By(t)
        amp_mod(t)*B1*cos(w*t+freq_mod(t))+noise_itp(t)
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

function dW_dual(noiserms, gyro1, gyro2, bandwidth)
    g1 = gyro1 * noiserms/sqrt(bandwidth)
    g2 = gyro2 * noiserms/sqrt(bandwidth)
    function dW(du, u, p, t)
        du[1] = u[3] * -g1
        du[2] = 0
        du[3] = u[1] * g1
        du[4] = u[6] * -g2
        du[5] = 0
        du[6] = u[4] * g2
    end
end

B_func = B_test(1, 2*pi; freq_mod=t->3*pi*sin(2*pi*0.1*t))
tt = 0:1e-2:30
plot(tt, [B_func(t) for t=tt])

function phase(a, b; dims=1)
    anorm = sqrt.(sum(a.*a, dims=dims))
    bnorm = sqrt.(sum(b.*b, dims=dims))
    acos.(clamp.(sum(a .* b, dims=dims)./(anorm .* bnorm), -1.0, 1.0))
end

function planephase(a; smoothing=true)
    if length(size(a)) == 2
        ph = [atan(a[3,i], a[2,i]) for i=1:size(a)[2]]
        if smoothing
            # assumes phase increases with time.
            tol = .5
            total_phase = 0.
            for i in 2:length(ph)
                diff = ph[i] + total_phase - ph[i-1]
                if abs(diff) > tol
                    if abs(diff + 2*pi) < tol
                        total_phase += 2*pi
                    elseif abs(diff - 2*pi) < tol
                        total_phase -= 2*pi
                    end
                end
                ph[i] += total_phase
            end
            return ph
        else
            return ph
        end
    elseif length(size(a)) == 3
        x = [planephase(a[:,:,i]; smoothing=smoothing) for i=1:size(a, 3)]
        return hcat(x...)
    end
end

function compare_phase(no_noise, ensemble; precomputed=false)
    if precomputed
        if length(size(no_noise)) == 2
            ensemble_phase = ensemble[1,:,:]
            no_noise_phase = no_noise[1,:]
        else
            ensemble_phase = ensemble
            no_noise_phase = no_noise
        end
    else
        ensemble_phase = planephase(ensemble)
        no_noise_phase = planephase(no_noise)
    end
    phase_diff = ensemble_phase .- no_noise_phase
    phase_diff = asin.(sin.(phase_diff)) # Get rid of additive multiples of 2pi
    if length(size(phase_diff)) == 2
        return mean(phase_diff, dims=2)[:,1], std(phase_diff, dims=2)[:,1]
    else
        return phase_diff, nothing
    end
end

function smooth(x, n; smoothtype="mean")
    if smoothtype == "mean"
        return [sum(@view x[i:(i+n-1)])/n for i in 1:(length(x)-(n-1))]
    elseif smoothtype == "max"
        return [maximum(@view x[i:(i+n-1)]) for i in 1:(length(x)-(n-1))]
    else
        error("Smoothing type $smoothtype not recognized.")
    end
end

function run_simulations(tend, n; 
        B0=crit_dress_params["B0"], 
        B1=crit_dress_params["B1"],
        w=crit_dress_params["w"],
        noiseratio=1e-4,
        phase=0,
        noiserate=5000, 
        nsave=0,
        saveat=[],
        saveinplane=false,
        downsample=1,
        stochastic=true,
        phaseonly=false,
        filtercutoff=nothing,
        uppercutoff=nothing,
        filtertype=Elliptic(7, 1, 60),
        amp_mod=t->1,
        freq_mod=t->0)
    if filtercutoff != nothing && filtercutoff > 0.0 && stochastic
        error("Solving filterd noise with SDE is not possible at this time.")
    end
    if filtercutoff == 0.0
        filtercutoff = nothing
    end
    Bnoise = B1*noiseratio
    u0nh = [0.0, 0.0, 1.0, 0.0, sin(phase), cos(phase)]
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
    if stochastic
        W = WienerProcess(0.0,0.0,0.0)
        dSnh = dS_dual(B0, B1, w, neutrongyro, he3gyro)
        dWnh = dW_dual(Bnoise, neutrongyro, he3gyro, noiserate)
        prob = SDEProblem(dSnh,dWnh,u0nh,tspan, noise=W)
        ensembleprob = EnsembleProblem(prob, output_func=(sol, i)->(sol, false))
        return SpinSolution(saveat, solve(ensembleprob, SRIW1(), EnsembleThreads(), saveat=saveat, dense=false,
            trajectories=n, abstol=1e-3, reltol=1e-3, maxiters=1e8))
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
function extract_noise(sol::SpinSolution; noisetype="phase", nsmooth=1, smoothtype="mean")
    t = sol.t
    noise = nothing
    if noisetype == "phase"
        noise = std(phase(sol.u[1:3,:,:], sol.u[4:6,:,:])[1,:,:], dims=2)
    elseif noisetype == "x"
        noise = std(sol.u[1,:,:] - sol.u[4,:,:], dims=2)[:,1]
    elseif noisetype == "y"
        noise = std(sol.u[2,:,:] - sol.u[5,:,:], dims=2)[:,1]
    elseif noisetype == "z"
        noise = std(sol.u[3,:,:] - sol.u[6,:,:], dims=2)[:,1]
    elseif noisetype == "nspread"
        x = sol.u[1,:,:]
        y = sol.u[2,:,:]
        z = sol.u[3,:,:]
        noise = sqrt.(var(x, dims=2) + var(y, dims=2) + var(z, dims=2))[:,1]
    elseif noisetype == "npolarization"
        av = mean(sol.u[1:3,:,:], dims=3)
        noise = [norm(av[:,i]) for i=1:size(av, 2)]
    elseif noisetype == "hepolarization"
        av = mean(sol.u[4:6,:,:], dims=3)
        noise = [norm(av[:,i]) for i=1:size(av, 2)]
    else
        error("Unknown noisetype $noisetype")
    end

    if nsmooth == 1
        return t, noise
    else
        return t[1:length(t)-(nsmooth-1)], smooth(noise, nsmooth, smoothtype=smoothtype)
    end
end

function file_paths(save_dir, sub_dir; data_fname="data.jld2", metadata_fname="metadata.jld2")
    # data path and metadata path, respectively
    joinpath(save_dir, sub_dir, data_fname), joinpath(save_dir, sub_dir, metadata_fname)
end

sim_time = 1
phaseonly = false
downsample = 1
saveinplane = false
nsave = 10001
w_mod = 7.539822368615504e+02
amp_mod = 4961.20311855
w = crit_dress_params["w"]
no_noise_sol = run_simulations(sim_time, 1; 
    B1=4.520940619540292e-01,
    stochastic=false, 
    phaseonly=phaseonly, 
    noiseratio=0, 
    saveinplane=saveinplane, 
    downsample=downsample, 
    nsave=nsave,
    freq_mod=t->amp_mod*sin(w_mod*t)/(w_mod))
plot(no_noise_sol.t[1:100], no_noise_sol.u[3,1:100,1])
plot!(no_noise_sol.t[1:100], no_noise_sol.u[6,1:100,1])

save_dir = "simulations_mod_3"
datefmt = "mm-dd-HH-MM-SS-sss"
sim_time = 1
n_runs = 100
phaseonly = false
phase0 = 0
downsample = 1
saveinplane = false
nsave = 5001
w_mod = 7.539822368615504e+02
B1_mod=4.520940619540292e-01
phi1 = 0
phi2 = 0
#Amp/wfm*(1.0-cos(wfm*time+phi2))+w1*time+phi1
for i=1:1
    no_noise_sol = run_simulations(sim_time, 1; stochastic=false, phaseonly=phaseonly, noiseratio=0, saveinplane=saveinplane, downsample=downsample, nsave=nsave)
    f = 100.0
    amp_mod = 4961.20311855
    for (phi1, phi2)=Iterators.product([0, pi/2], [0, pi/4, pi/2])
        sol = run_simulations(sim_time, n_runs; stochastic=false, B1=B1_mod, phase=phase0, phaseonly=phaseonly, 
            filtercutoff=f, filtertype=Elliptic(7, 1, 60), 
            nsave=nsave, saveinplane=saveinplane,
            freq_mod=t->amp_mod*(1-cos(w_mod*t + phi2))/(w_mod) + phi1)
        timestamp = Dates.now()
        mod_info = Dict("phi1"=>phi1, "phi2"=>phi2, "amp"=>amp_mod, "w"=>w_mod)
        metadata = Dict("time"=>sim_time, 
            "cutoff"=>f, 
            "uppercutoff"=>nothing, 
            "phaseonly"=>phaseonly, 
            "downsample"=>downsample, 
            "timestamp"=>timestamp,
            "freq_mod"=>mod_info)
        save_str = Dates.format(timestamp, datefmt)
        sub_dir = "data-$(save_str)"
        sub_path = joinpath(save_dir, "data-$(save_str)")
        mkdir(sub_path)
        dp, mp = file_paths(save_dir, sub_dir)
        @save dp no_noise_sol sol 
        @save mp metadata
    end
end

function predict_phase(t; 
        B0=crit_dress_params["B0"],
        B1=crit_dress_params["B1"],
        w=crit_dress_params["w"],
        noiseratio=1e-4, 
        noiserate=5000, 
        filtercutoff=0.0)
    
    wcutoff = filtercutoff * 2 * pi
    wn = abs(neutrongyro * B0)
    w3 = abs(he3gyro * B0)
    w0 = besselj(0, neutrongyro * B1/w) * wn
    J1_n = besselj(1, neutrongyro * B1/w)
    J1_3 = besselj(1, he3gyro * B1/w)
    
    sigma = B1 * noiseratio/sqrt(noiserate)
    
    sigma_w0 = wcutoff < w0 ? abs(neutrongyro - he3gyro) * sigma/sqrt(2) : 0
    sigma_rf = true ? abs(neutrongyro * J1_n * wn - he3gyro * J1_3 * w3) * sqrt(2) * sigma/w : 0
    
    return sqrt(sigma_w0^2 + sigma_rf^2) .* sqrt.(t)
end

function predict_shift(t, gamma; B1=4.0596741e-1, noiseratio=1e-4, noiserate=5000, filtercutoff=0.0)
    f_res = 60.
    if filtercutoff == 0.0
        return 0.0 .* t
    else
        sigma = B1 * noiseratio/sqrt(noiserate)
        return -gamma^2/(4*pi) * sigma^2 * log((filtercutoff+f_res)/(filtercutoff-f_res)) .* t
    end
end

function data_iterator(save_dir)
    directories = filter(d->isdir(joinpath(save_dir, d)),readdir(save_dir))
    # returns in the order data was created
    function sort_fn(dir)
        _, mp = file_paths(save_dir, dir)
        @load mp metadata
        metadata["timestamp"]
    end
    [file_paths(save_dir, d) for d=sort(directories, by=sort_fn)]
end
function aggregate_data(save_dir; 
        selector=m->true, 
        key_func=(m,n,s)->0, 
        aggregator=(r,m,n,s)->(r==nothing ? [s] : append!(r,s)), 
        postprocess=r->r)
    result = Dict()
    for (dp,mp)=data_iterator(save_dir)
        @load mp metadata
        if !selector(metadata)
            continue
        end
        #display("Accessing $(dp)...")
        @load dp no_noise_sol sol
        key = key_func(metadata, no_noise_sol, sol)
        if !haskey(result, key)
            result[key] = nothing
        end
        result[key] = aggregator(result[key], metadata, no_noise_sol, sol)
    end
    for key in keys(result)
        result[key] = postprocess(key, result[key])
    end
    result
end

function phase_shift_aggregator(r, metadata, no_noise_sol, sol)
    nsmooth=10
    phases = cat([smooth(compare_phase(no_noise_sol.u[:,:,1], sol.u[:,:,i], precomputed=phaseonly)[1], nsmooth) for i=1:size(sol.u, 3)]..., dims=2)
    m = mean(phases, dims=2)
    v = var(phases, dims=2)./size(phases, 2)
    if r == nothing
        return no_noise_sol.t[1:end-(nsmooth-1)], m, v, 0
    else
        t, cum_sum, cum_var, N = r
        return t, cum_sum .+ m, cum_var .+ v, N + 1
    end
end

function phase_shift_postprocess(key, r)
    t, cum_sum, cum_var, N = r
    avg = cum_sum./N
    uncertainty = (sqrt.(cum_var))./N
    plot!(t, avg, ribbon=uncertainty, label=key)
    avg[end], uncertainty[end]
end

plot()
res = aggregate_data(save_dir;
        selector=m->!m["phaseonly"] && m["downsample"]==10,
        key_func=(m,n,s)->m["cutoff"],
        aggregator=phase_shift_aggregator,
        postprocess=phase_shift_postprocess)

plot!()

cutoffs = sort([k for k=keys(res)])
shifts = [res[f][1] for f=cutoffs]
uncertainties = [res[f][2] for f=cutoffs]
scatter(cutoffs, shifts, yerror=uncertainties, yformatter=:scientific, label="simulation")

cc = range(75, 200, length=100)
ss = [predict_shift(10., neutrongyro; filtercutoff=f) for f=cc]
plot!(cc, ss, label="prediction")
yaxis!("Phase Shift after 10s [radians]")
xaxis!("Highpass Filter Cutoff [Hz]")

function polarization_aggregator(r, metadata, no_noise_sol, sol)
    t, phase_noise = extract_noise(sol; noisetype="npolarization", nsmooth=1, smoothtype="mean")
    ii = 1
    plot!(t[ii:end], phase_noise[ii:end], label=string(metadata["freq_mod"]["phi1"], ", ", metadata["freq_mod"]["phi2"]))
    return t[1:10:end], phase_noise[1:10:end], size(sol.u, 3)
end

function polarization_postprocess(key, r)
    return r
end

plot()
res = aggregate_data("simulations_mod_2";
        selector=m->!m["phaseonly"],
        key_func=(m,n,s)->(m["freq_mod"]["phi1"], m["freq_mod"]["phi2"]),
        aggregator=polarization_aggregator,
        postprocess=polarization_postprocess)
plot!(title="modulated")

function phase_noise_aggregator(r, metadata, no_noise_sol, sol)
    t, phase_noise = extract_noise(sol; noisetype="phase", nsmooth=1, smoothtype="mean")
    #ii = 1
    #plot!(t[ii:end], phase_noise[ii:end], label=string(metadata["cutoff"]))
    return t[1:10:end], phase_noise[1:10:end], size(sol.u, 3)
end

function phase_noise_postprocess(key, r)
    return r
end

plot()
res1 = aggregate_data("simulations_2";
        selector=m->!m["phaseonly"],
        key_func=(m,n,s)->m["cutoff"],
        aggregator=phase_noise_aggregator,
        postprocess=phase_noise_postprocess)
plot!(title="modulated")

res1 = nothing

res_merge = res1
cutoffs = [key for key=keys(res_merge)]
noises = [res_merge[key][2][end] for key=keys(res_merge)]
uncertainties = noises./sqrt(res_merge[cutoffs[1]][3])
scatter(cutoffs, noises, yerror=uncertainties, yscale=:log10, label="simulation")
cc = range(0, 1500, length=100)
nn = [predict_phase(1.; filtercutoff=f) for f=cc]
plot!(cc, nn, label="prediction")
yaxis!("Phase Noise after 1s [radians]")
xaxis!("Highpass Filter Cutoff [Hz]")

predict_phase(1, filtercutoff=0)

predict_phase(1, filtercutoff=100.0)


