module Utils
export phase, planephase, smooth, file_paths, save_data, load_data, dressed_gamma

using Dates
using JLD2
using LinearAlgebra

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

function dressed_gamma(B0,Bdress,wdress,gamma,E0,dn; option=false)
    #This functions calculates the effective gyromagnetic ratio in an strong RF
    #field with angular frequency wdress
    #with a uniform B0 & E field as a perturbation. wdress is an angular frequency.  
    #B0 & Bdress are in Gauss; gamma is the gyromagnetic ratio
    #dn is EDM in ecm; E0 is an E field in V/cm.

    Bd=Bdress
    wd=wdress
    hbar=6.582122E-16; #hbar in eVs
    #gamma=20378.9;     #gryomagnetic ratio rad/G
    gammaE=2*dn/hbar;  #gryoelectric ratio rad/(V/cm)  
    #wd=6000
    #B0=0.03
    #E0=-75E3;  #kV/cm
    #Bd=.37505920;        #Best number for 6000 according to simulations!
    x=gamma*Bd/wd
    y=gammaE*E0/wd+gamma*B0/wd

    N = 20
    Hsd = nothing
    if option
        v0 = collect(Iterators.flatten(zip(1:N, 1:N))) # [y, y, 2y, 2y, 3y, 3y, ...]
        v1 = ((1:2*N-1) .% 2) .* y/2 # alternates [y/2, 0, y/2, ...]
        v2 = (((1:2*N-2) .% 2) .* x/2) .- x/4 # alternates [+x/4, -x/4, +, - ...]
        Hsd = diagm(0=>v0, 1=>v1, -1=>v1, 2=>v2, -2=>v2)
    else
        #Hsdd1=N+y/2:-1:1+y/2
        #Hsdd2=N-y/2:-1:1-y/2
        Hsdd1=(1:N).+y/2
        Hsdd2=(1:N).-y/2

        Hsdd=sort(-vcat(Hsdd1,Hsdd2))
        Hsdd=-Hsdd
        Hsdp=diagm(Hsdd)
        Hsd=Hsdp

        for i = 1:size(Hsdp,1)
            for j = 1:size(Hsdp,2)
                if (i+j-5)%(4)==0 && abs(i-j)<=4
                    Hsd[i,j]=x/4
                end
            end
        end
    end

    delE=diff(eigen(Hsd).values)
    delE[convert(Int, size(Hsd, 1)/2-1)]*wd/B0
end
end
