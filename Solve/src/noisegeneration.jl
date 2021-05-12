using Parameters
using DSP
using FFTW

@with_kw struct NoiseIterator
    Bnoise::Float64
    duration::Float64
    noiserate::Float64
    filtercutoff::Float64
    uppercutoff::Float64
    filtertype
    repeat::Int = 1
end

keyname = "dataBuffer";

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

function perfect_filter(sample, lower, upper, fs)
    y = rfft(sample)
    w = FFTW.rfftfreq(length(sample), fs)
    for i=1:length(y)
        # upper == 0 corresponds to a highpass filter
        if w[i] < lower || (upper != 0 && w[i] > upper)
            y[i] = 0
        end
    end
    return irfft(y, length(sample))
end

function makenoise(rms, t, noiserate; filtercutoff=0.0, uppercutoff=0.0, filtertype=Elliptic(7, 1, 60))
    dt = 1.0/noiserate
    tarr = range(0, t+dt, step=dt)
    pts = rms*randn(length(tarr))

    if filtercutoff == 0.0 && uppercutoff == 0.0
        filtered_pts = pts
    elseif filtertype isa String
        if lowercase(filtertype) != "perfect"
            throw(ErrorException("filtertype not recognized"))
        end
        filtered_pts = perfect_filter(pts, filtercutoff, uppercutoff, noiserate)
    elseif uppercutoff == 0.0
        filtered_pts = filt(digitalfilter(Highpass(filtercutoff; fs=noiserate), filtertype), pts)
    elseif filtercutoff == 0.0
        filtered_pts = filt(digitalfilter(Lowpass(uppercutoff; fs=noiserate), filtertype), pts)
    else
        filtered_pts = filt(digitalfilter(Bandpass(filtercutoff, uppercutoff; fs=noiserate), filtertype), pts)
    end
    return scale(interpolate(filtered_pts, BSpline(Cubic(Line(OnGrid())))), tarr)
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
    # All .mat files that have the correct key.
    paths = [joinpath(directory, f) for f in readdir(directory)]
    [f for f in paths if occursin(r"\.mat$", f) && mat_has_key(f, keyname)]
end

function compute_normalization(fnames; w=crit_params["w"], trim=0.5, fs=daq_rate)
    # Computes the average amplitude of all waveforms in the files given
    total = 0.
    itrim = round(Int, trim*fs)
    phase_1 = 0
    phase_2 = 0
    for fname in fnames
        file = matopen(fname)
        data = read(file, keyname)[:,1]
        data = data[itrim:end-itrim]
        data = trim_cycles(data, w, fs)
        t = (0:length(data)-1)/fs
        x = data .- mean(data)
        a1, a2 = fit_sine(cumsum(x), t, w)
        amp = sqrt(a1^2 + a2^2)
        total += amp*(w/fs)
        phase_1 += a1
        phase_2 += a2
    end
    total/length(fnames), atan(phase_2, phase_1)
end

function preprocess(data;
                    w=crit_params["w"],
                    filtercutoff=0.0, 
                    filtertype=Elliptic(7, 1, 60),
                    trim=0.5,
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

function data_from_file(fname)
    file = matopen(fname)
    read(file, keyname)[:,1]
end

function make_waveform(data, duration, norm;
                       phase=pi/2,
                       B1=crit_params["B1"],
                       w=crit_params["w"], 
                       filtercutoff=0.0, 
                       filtertype=Elliptic(7, 1, 60),
                       trim=0.5,
                       fs=daq_rate,
                       fit_time=0)
    # norm: a normalization constant computed by compute_normalization
    # w: angular frequency of waveforms
    # amp: desired amplitude
    # phase: desired starting phase
    # trim: seconds to trim from the start and end of waveforms (to get rid of glitches)
    # duration: how long the sample should be
    record_len = round(Int, duration*fs)+1
    sig = cumsum(data)*w/fs
    sig = preprocess(sig;
                     w=w,
                     filtercutoff=filtercutoff,
                     filtertype=filtertype,
                     trim=trim,
                     fs=fs)

    t = (0:length(sig)-1)/fs
    if fit_time == 0
        a1, a2 = fit_sine(sig, t, w)
    else
        # find a segment with relatively stable amplitude
        # with length fit_time
        s0 = sin(w*t) * sig
        c0 = cos(w*t) * sig
        sa1 = filt(digitalfilter(Lowpass(w*0.1; fs=fs), Butterworth(7)), s0)
        ca1 = filt(digitalfilter(Lowpass(w*0.1; fs=fs), Butterworth(7)), c0)
        amplitude = sqrt.(sa1.^2 + ca1.^2)
        seg_len = round(fit_time*fs)
        rmses = [std(amplitude[i:i+seg_len]) for i=1:seg_len]
        imin = argmin(rmses)
        a1, a2 = fit_sine(sig[imin:imin+seg_len], t, w)
    end
    phi0 = atan(a2, a1)
    sig = B1*(sig .- mean(sig))/norm
    #sig = B1*(sig .- mean(sig))/sqrt(a1^2 + a2^2)
    buffer = 10 # Samples
    #n0 = rand(buffer+1:length(sig)-buffer-record_len+1)
    n0 = buffer + 1
    phi1 = w*(n0-1)/fs + phi0
    # Now set the phase correctly, so that
    # S(t) = sin(w*t + phase)
    dn = (fs/w) * mod2pi(phase-phi1)
    offset = floor(Int, dn)
    sig = sig[n0+offset-buffer:n0+offset+record_len+buffer]
    time = ((0:length(sig)-1) .- (dn - offset) .- buffer)/fs
    return scale(interpolate(sig, BSpline(Cubic(Line(OnGrid())))), time)
end

function waveform_is_ok(data, fs)
    # Look for long stretches of small values
    glitch_duration_threshold = 1 # seconds
    glitch_amplitude_threshold = 0.1

    dn = glitch_duration_threshold*fs # samples
    count = 0
    for x=data
        if abs(x) < glitch_amplitude_threshold
            count += 1
        else
            count = 0
        end

        if count > dn
            return false
        end
    end
    return true
end

function daq_noise_iterator(directory, duration; 
        B1=crit_params["B1"], 
        w=crit_params["w"], 
        filtercutoff=0.0, 
        filtertype=Elliptic(7, 1, 60),
        trim=0.5,
        fs=daq_rate)
    # directory: directory containing .mat files
    # duration: duration of waveforms
    # trim: seconds to trim from the start and end of waveforms (to get rid of glitches)    
    fnames = search_for_waveforms(directory)
    
    norm, phase = compute_normalization(fnames; w=w, fs=fs)
    
    # There's probably a better way to make an iterator that goes forever
    #raw_data_iterator = imap(i->data_from_file(rand(fnames)), Iterators.countfrom())
    raw_data_iterator = imap(f->data_from_file(f), Iterators.cycle(fnames))
    
    filtered_data_iterator = Iterators.filter(d->waveform_is_ok(d, fs), raw_data_iterator)

    it0 = imap(d->make_waveform(d, duration, norm;
                                phase=pi/2,
                                B1=B1,                          
                                w=w,
                                filtercutoff=filtercutoff,
                                filtertype=filtertype,
                                trim=trim,
                                fs=fs), filtered_data_iterator)
    #filtered_data_iterator
    it0
end
