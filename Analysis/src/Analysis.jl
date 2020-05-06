module Analysis
export compare_phase, extract_noise, predict_phase, predict_shift, aggregate_data
using Utils
using Solve: SpinSolution

using JLD2
using Statistics

function compare_phase(no_noise, ensemble)
    if length(size(no_noise)) == 1
        ensemble_phase = ensemble
        no_noise_phase = no_noise
    elseif size(no_noise, 1) == 1
        ensemble_phase = ensemble[1,:,:]
        no_noise_phase = no_noise[1,:]
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

function data_iterator(save_dir)
    directories = filter(d->isdir(joinpath(save_dir, d)),readdir(save_dir))
    # returns in the order data was created
    function sort_fn(dir)
        _, mp = file_paths(joinpath(save_dir, dir))
        @load mp metadata
        metadata["timestamp"]
    end
    [file_paths(joinpath(save_dir, d)) for d=sort(directories, by=sort_fn)]
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

end
