module Analysis
export compare_phase, extract_noise, extract_signal
export predict_signal_std, predict_phase, predict_shift, predict_polarization
export aggregate_data, read_metadata, get_random_metadata
using Utils
using Solve

using JLD2
using Statistics
using SpecialFunctions
using LinearAlgebra

function compare_phase(no_noise, ensemble; axis=3)
    if length(size(no_noise)) == 1
        ensemble_phase = ensemble
        no_noise_phase = no_noise
    elseif size(no_noise, 1) == 1
        ensemble_phase = ensemble[1,:,:]
        no_noise_phase = no_noise[1,:]
    else
        ensemble_phase = planephase(ensemble; axis=axis)
        no_noise_phase = planephase(no_noise; axis=axis)
    end
    phase_diff = ensemble_phase .- no_noise_phase
    phase_diff = asin.(sin.(phase_diff)) # Get rid of additive multiples of 2pi
    if length(size(phase_diff)) == 2
        return mean(phase_diff, dims=2)[:,1], std(phase_diff, dims=2)[:,1]
    else
        return phase_diff, nothing
    end
end

function extract_signal(sol::SpinSolution)
    return signal(sol.u)
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
        noise = (var(x, dims=2) + var(y, dims=2) + var(z, dims=2))[:,1]
    elseif noisetype == "signal"
        # smooth before taking variance
        noise = var(smooth_2d(extract_signal(sol), nsmooth), dims=2)
        return sol.t[1:length(t)-(nsmooth-1)], noise
    elseif noisetype == "meandot"
        noise = mean(sum(sol.u[1:3,:,:] .* sol.u[4:6,:,:], dims=1)[1,:,:], dims=2)[:,1]
    elseif noisetype == "diffspread"
        diff = sol.u[1:3,:,:] .- sol.u[4:6,:,:]
        noise = sqrt.(var(diff[1,:,:], dims=2) + var(diff[2,:,:], dims=2) + var(diff[3,:,:], dims=2))
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
        postprocess=(k,r)->r)
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

    keyit = all(isa.(keys(result), Number)) ? sort(collect(keys(result))) : keys(result)

    for key in sort(collect(keys(result)))
        result[key] = postprocess(key, result[key])
    end
    result
end

function predict_signal_std(t;
                            theta=pi/2,
                            B0=crit_params["B0"],
                            B1=crit_params["B1"],
                            w=crit_params["w"],
                            Bnoise=crit_params["B1"]*1e-4,
                            noiserate=5000,
                            filtercutoff=0.0)
    wcutoff = filtercutoff * 2 * pi
    wn = abs(neutrongyro * B0)
    w3 = abs(he3gyro * B0)
    w0 = besselj(0, neutrongyro * B1/w) * wn
    J1_n = besselj(1, neutrongyro * B1/w)
    J1_3 = besselj(1, he3gyro * B1/w)
    J2_n = besselj(2, neutrongyro * B1/w)
    J2_3 = besselj(2, he3gyro * B1/w)
    J3_n = besselj(3, neutrongyro * B1/w)
    J3_3 = besselj(3, he3gyro * B1/w)
    J4_n = besselj(4, neutrongyro * B1/w)
    J4_3 = besselj(4, he3gyro * B1/w)

    # Square root of PSD of noise field
    sigma = Bnoise/sqrt(noiserate)
    
    sigma_w0 = wcutoff < w0 ? 0 : 0
    sigma_rf = wcutoff < 1*w ? sqrt(2)*abs(neutrongyro * J1_n * wn - he3gyro * J1_3 * w3) * sigma/w * abs(sin(theta)) : 0
    sigma_2w = wcutoff < 2*w ? 0 : 0
    sigma_3w = 1/3 * abs(neutrongyro * J3_n * wn - he3gyro * J3_3 * w3) * sigma/w

    return sqrt(sigma_w0^2 + sigma_rf^2 + sigma_2w^2 + sigma_3w^2) .* sqrt.(t)
end

function predict_phase(t; 
        B0=crit_params["B0"],
        B1=crit_params["B1"],
        w=crit_params["w"],
        Bnoise=crit_params["B1"]*1e-4,
        noiserate=5000, 
        filtercutoff=0.0)
    
    wcutoff = filtercutoff * 2 * pi
    wn = abs(neutrongyro * B0)
    w3 = abs(he3gyro * B0)
    w0 = besselj(0, neutrongyro * B1/w) * wn
    J1_n = besselj(1, neutrongyro * B1/w)
    J1_3 = besselj(1, he3gyro * B1/w)
    J2_n = besselj(2, neutrongyro * B1/w)
    J2_3 = besselj(2, he3gyro * B1/w)
    J3_n = besselj(3, neutrongyro * B1/w)
    J3_3 = besselj(3, he3gyro * B1/w)
    J4_n = besselj(4, neutrongyro * B1/w)
    J4_3 = besselj(4, he3gyro * B1/w)

    # Square root of PSD of noise field
    sigma = Bnoise/sqrt(noiserate)
    
    sigma_w0 = wcutoff < w0 ? 1/2 * abs(neutrongyro - he3gyro) * sigma : 0
    sigma_rf = wcutoff < w ? abs(neutrongyro * J1_n * wn - he3gyro * J1_3 * w3)  * sigma/w : 0
    sigma_2w = wcutoff < 2*w ? sqrt(2)/4 * abs(neutrongyro * J2_n * wn - he3gyro * J2_3 * w3) * sigma/w : 0
    sigma_3w = wcutoff < 3*w ? 1/3 * abs(neutrongyro * J3_n * wn - he3gyro * J3_3 * w3)  * sigma/w : 0
    sigma_4w = sqrt(2)/8 * abs(neutrongyro * J4_n * wn - he3gyro * J4_3 * w3)  * sigma/w    

    return sqrt(sigma_w0^2 + sigma_rf^2 + sigma_2w^2 + sigma_3w^2 + sigma_4w^2) .* sqrt.(t) .* sqrt(2)
end

function predict_shift(t, gamma;
                       B0=crit_params["B0"],
                       B1=crit_params["B1"],
                       w=crit_params["w"],
                       Bnoise=crit_params["B1"]*1e-4,
                       noiserate=5000,
                       filtercutoff=0.0,
                       noiseratecorrection=true)
    f_res_0 = abs(gamma * B0)/(2*pi) * besselj(0, gamma * B1/w)
    f_res = dressed_gamma(B0,B1,w,gamma,0,0)*B0/(2*pi)
    if filtercutoff == 0.0
        return 0.0 .* t
    else
        sigma = Bnoise/sqrt(noiserate)
        cutoff_factor = log((filtercutoff+f_res)/(filtercutoff-f_res))
        if noiseratecorrection
            cutoff_factor -= log((noiserate/2 + f_res)/(noiserate/2 - f_res))
        end
        return -gamma^2/(4*pi) * sigma^2 * cutoff_factor .* t
    end
end

function predict_polarization(t, gamma;
                              B0=crit_params["B0"],
                              B1=crit_params["B1"],
                              w=crit_params["w"],
                              Bnoise=crit_params["B1"]*1e-4,
                              noiserate=5000,
                              filtercutoff=0.0,
                              noiseratecorrection=true)
    wcutoff = filtercutoff * 2 * pi
    wn = abs(gamma * B0)
    xn = gamma * B1/w
    w0 = besselj(0, xn) * wn

    # Square root of PSD of noise field
    sigma = Bnoise/sqrt(noiserate)

    rev_t = sigma^2/2 # re(v)/t
    
    v_w0 = wcutoff < w0 ? 1/2 * gamma^2 * rev_t : 0
    v_rf = wcutoff < w ? 2 * (gamma * besselj(1, xn) * wn/w)^2 * rev_t : 0
    v_2w = wcutoff < 2*w ? 2 * 1/8 * (gamma * besselj(2, xn) * wn/w)^2 * rev_t : 0 # 2w+w_0' and 2w-w_0'
    v_3w = wcutoff < 3*w ? 1 * (gamma * besselj(3, xn) * wn/w)^2 * rev_t : 0
    v_4w = 1 * (gamma * besselj(4, xn) * wn/w)^2 * rev_t

    1 - (v_w0 + v_rf + v_2w + v_3w + v_4w) .* t
end

function read_metadata(save_dir)
    # Ignores all data, just prints metadata
    aggregate_data(save_dir, selector=m->(println(m)==nothing && false))
    return nothing
end

function get_random_metadata(save_dir)
    # Convenience function, for when you want one metadata dictionary
    # but don't care which
    for (dp,mp)=data_iterator(save_dir)
        @load mp metadata
        return metadata
    end
end
end
