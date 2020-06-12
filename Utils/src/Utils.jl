module Utils
export phase, planephase, smooth, file_paths, save_data, load_data

using Dates
using JLD2

function smooth(x, n; smoothtype="mean")
    if smoothtype == "mean"
        return [sum(@view x[i:(i+n-1)])/n for i in 1:(length(x)-(n-1))]
    elseif smoothtype == "max"
        return [maximum(@view x[i:(i+n-1)]) for i in 1:(length(x)-(n-1))]
    else
        error("Smoothing type $smoothtype not recognized.")
    end
end

function phase(a, b; dims=1)
    anorm = sqrt.(sum(a.*a, dims=dims))
    bnorm = sqrt.(sum(b.*b, dims=dims))
    acos.(clamp.(sum(a .* b, dims=dims)./(anorm .* bnorm), -1.0, 1.0))
end

function planephase(a; axis=3)
    # axis: the axis about which the spins precess (1=x, 2=y, 3=z)
    if length(size(a)) == 2
        q = (axis % 3) + 1
        r = ((axis + 1) % 3) + 1
        return [atan(a[r,i], a[q,i]) for i=1:size(a)[2]]
    elseif length(size(a)) == 3
        x = [planephase(a[:,:,i]) for i=1:size(a, 3)]
        return hcat(x...)
    end
end

function file_paths(save_path; data_fname="data.jld2", metadata_fname="metadata.jld2")
    # returns data path and metadata path, respectively
    joinpath(save_path, data_fname), joinpath(save_path, metadata_fname)
end

function save_data(no_noise_sol, sol, metadata, save_dir; datefmt="mm-dd-HH-MM-SS-sss")
    timestamp = Dates.now()
    metadata["timestamp"] = timestamp
    save_str = Dates.format(timestamp, datefmt)
    sub_dir = "data-$(save_str)"
    save_path = joinpath(save_dir, sub_dir)
    mkdir(save_path)
    dp, mp = file_paths(save_path)
    @save dp no_noise_sol sol
    @save mp metadata
    return save_path
end

function load_data(save_path)
    dp, mp = file_paths(save_path)
    @load dp no_noise_sol sol
    @load mp metadata
    return no_noise_sol, sol, metadata
end
end
