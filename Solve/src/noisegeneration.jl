using Parameters
using DSP

@with_kw struct NoiseIterator
    Bnoise::Float64
    duration::Float64
    noiserate::Float64
    filtercutoff::Float64
    uppercutoff::Float64
    filtertype
    repeat::Int = 1
end

function Base.iterate(iter::NoiseIterator, state=(0, nothing))
    if state[1] % iter.repeat == 0
        noise = makenoise(iter.Bnoise, iter.duration, iter.noiserate; 
            filtercutoff=iter.filtercutoff, uppercutoff=iter.uppercutoff, filtertype=iter.filtertype)
        return noise, (state[1]+1, noise)
    else
        return state[2], (state[1]+1, state[2])
    end
end

function NoiseIterator(rms, t, noiserate; filtercutoff=0.0, uppercutoff=0.0, filtertype=Elliptic(7, 1, 60), repeat=1)
    return NoiseIterator(rms, t, noiserate, filtercutoff, uppercutoff, filtertype, repeat)
end

function makenoise(rms, t, noiserate; filtercutoff=0.0, uppercutoff=0.0, filtertype=Elliptic(7, 1, 60))
    dt = 1.0/noiserate
    tarr = range(0, t+dt, step=dt)
    pts = rms*randn(length(tarr))
    if filtercutoff == 0.0
        return scale(interpolate(pts, BSpline(Cubic(Line(OnGrid())))), tarr)
    else
        if uppercutoff == 0.0
            filter = digitalfilter(Highpass(filtercutoff; fs=noiserate), filtertype)
        else
            filter = digitalfilter(Bandpass(filtercutoff, uppercutoff; fs=noiserate), filtertype)
        end
        return scale(interpolate(filt(filter, pts), BSpline(Cubic(Line(OnGrid())))), tarr)     
    end
end
