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

# Functions to read waveforms from files
daq_rate = 46875
function fit_sine(x, t, w)
    a1 = 2*sum(x .* sin.(w*t))/length(x)
    a2 = 2*sum(x .* cos.(w*t))/length(x)
    (a1, a2)
end

function trim_cycles(data, w, fs)
    # Trims the end of the data so that
    # The data contains an integer number of cycles
    # of angular frequency w
    #T = 2*pi/w # Period
    T = 2*pi/w
    tf = (length(data)-1)/fs
    data[1:round(Int, (tf - (tf % T))*fs)]
end

function mat_has_key(f, key)
    # Checks if .mat file f has key
    file = matopen(f)
    has_key = exists(file, key)
    close(file)
    has_key
end

function search_for_waveforms(directory)
    # All .mat files that have a "data" key.
    paths = [joinpath(directory, f) for f in readdir(directory)]
    [f for f in paths if occursin(r"\.mat$", f) && mat_has_key(f, "data")]
end

function compute_normalization(fnames; w=crit_params["w"], trim=1, fs=daq_rate)
    # Computes the average amplitude of all waveforms in the files given
    total = 0.
    itrim = round(Int, trim*fs)
    for fname in fnames
        file = matopen(fname)
        data = read(file, "data")[:,1]
        data = data[itrim:end-itrim]
        data = trim_cycles(data, w, fs)
        t = (0:length(data)-1)/fs
        x = data .- mean(data)
        a1, a2 = fit_sine(cumsum(x), t, w)
        amp = sqrt(a1^2 + a2^2)
        total += amp*(w/fs)
    end
    total/length(fnames)
end

function preprocess(data;
                    w=crit_params["w"],
                    filtercutoff=0.0, 
                    filtertype=Elliptic(7, 1, 60),
                    trim=1,
                    fs=daq_rate)
    # Filter out DC offsets and drifts
    if filtercutoff != 0.0
        filter = digitalfilter(Highpass(filtercutoff; fs=fs), filtertype)
        #filter = digitalfilter(Bandpass(filtercutoff, 1500; fs=fs), filtertype)
        data = filt(filter, data)/abs(freqz(filter, w/(2*pi), fs))
    end
    # Cut out glitches
    res = data[round(Int, trim*fs):end-round(Int, trim*fs)]
    # Try to get an integer number of cycles
    trim_cycles(res, w, fs)
end

function make_waveform(fname, duration, norm;
                       phase=pi/2,
                       B1=crit_params["B1"], 
                       w=crit_params["w"], 
                       filtercutoff=0.0, 
                       filtertype=Elliptic(7, 1, 60),
                       trim=1,
                       fs=daq_rate)
    # norm: a normalization constant computed by compute_normalization
    # w: angular frequency of waveforms
    # amp: desired amplitude
    # phase: desired starting phase
    # trim: seconds to trim from the start and end of waveforms (to get rid of glitches)
    # duration: how long the sample should be
    file = matopen(fname)
    record_len = round(Int, duration*fs)+1
    data = read(file, "data")[:,1]
    sig = cumsum(data)*w/fs
    sig = preprocess(sig;
                     w=w,
                     filtercutoff=filtercutoff,
                     filtertype=filtertype,
                     trim=trim,
                     fs=fs)

    t = (0:length(sig)-1)/fs
    a1, a2 = fit_sine(sig, t, w)
    phi0 = atan(a2, a1)
    #sig = B1*(sig .- mean(sig))/norm
    sig = B1*(sig .- mean(sig))/sqrt(a1^2 + a2^2)
    buffer = 10 # Samples
    n0 = rand(buffer+1:length(sig)-buffer-record_len+1)
    phi1 = w*(n0-1)/fs + phi0
    # Now set the phase correctly, so that
    # S(t) = sin(w*t + phase)
    dn = (fs/w) * mod2pi(phase-phi1)
    offset = floor(Int, dn)
    sig = sig[n0+offset-buffer:n0+offset+record_len+buffer]
    time = ((0:length(sig)-1) .- (dn - offset) .- buffer)/fs
    tmp = (fs/w) * phi0
    #return scale(interpolate(sig, BSpline(Cubic(Line(OnGrid())))), time)
    noise = sig .- B1*sin.(w*time .+ phase)
    noise_itp = scale(interpolate(noise, BSpline(Cubic(Line(OnGrid())))), time)
    f = t -> B1*sin(w*t + phase) + noise_itp(t)
    return f
end

function daq_noise_iterator(directory, duration; 
        B1=crit_params["B1"], 
        w=crit_params["w"], 
        filtercutoff=0.0, 
        filtertype=Elliptic(7, 1, 60),
        trim=1,
        fs=daq_rate)
    # directory: directory containing .mat files
    # duration: duration of waveforms
    # trim: seconds to trim from the start and end of waveforms (to get rid of glitches)    
    fnames = search_for_waveforms(directory)
    norm = compute_normalization(fnames; w=w, fs=fs)
    # There's probably a better way to make an iterator that goes forever
    imap(i->make_waveform(rand(fnames), duration, norm;
                          phase=pi/2,
                          B1=B1,                          
                          w=w,
                          filtercutoff=filtercutoff,
                          filtertype=filtertype,
                          trim=trim,
                          fs=fs), Iterators.countfrom())
end
