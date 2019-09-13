
# coding: utf-8

# # Experiments with WAV files and FFT

# Based on the blog here: http://samcarcagno.altervista.org/blog/basic-sound-processing-python/?doing_wp_cron=1544285108.7695550918579101562500

# In[116]:


import os
import sys

import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile # get the api

from pylab import*
from scipy.io import wavfile

import soundfile as sf

# Let's try getting frequency in chunks of time:

def get_fft(s1,sampFreq=44100) :
    # Do an FFT
    n = len(s1) 
    p = fft(s1) # take the fourier transform 

    nUniquePts = int(ceil((n+1)/2.0))
    p = p[0:nUniquePts]
    p = abs(p)

    p = p / float(n) # scale by the number of points so that
                     # the magnitude does not depend on the length 
                     # of the signal or on its sampling frequency  
    p = p**2  # square it to get the power 

    # multiply by two (see technical document for details)
    # odd nfft excludes Nyquist point
    if n % 2 > 0: # we've got odd number of points fft
        p[1:len(p)] = p[1:len(p)] * 2
    else:
        p[1:len(p) -1] = p[1:len(p) - 1] * 2 # we've got even number of points fft

    freqArray = arange(0, nUniquePts, 1.0) * (sampFreq / float(n));
    #plot(freqArray/1000, 10*log10(p), color='k')
    #xlabel('Frequency (kHz)')
    #ylabel('Power (dB)')

    freqs = freqArray/1000
    pows  = 10*log10(p)

    return (freqs,pows)

# Assuming 44.1 kHz sampling rate, get slices of frequency
# content in terms of power.
def get_freq_samples(amp,sampFreq=44100,samp_tot=10000,samp_div=1000) :
    
    # Pad out the length of the sample with random low volume noise,
    # if necessary.
    if len(amp)<samp_tot :
        amp = np.append(amp,np.random.rand(samp_tot-len(amp)))
    
    ns    = int(samp_tot/samp_div)
    fv = np.array([])
    pv = np.array([])
    fa = []
    pa = []
    
    # First get the dimension of the fft.
    for i in range(0,ns) :
        s_amp = amp[i*samp_div:(i+1)*samp_div]
        f,p   = get_fft(s_amp,sampFreq)
        fv = np.append(fv,f)
        pv = np.append(pv,p)
        fa.append(f)
        pa.append(p)
        
    return (fv,pv,fa,pa)

# Sample from a distribution up to a point (in time,
# frequency, etc.). Any values falling outside
# the range are set to a specified value (null_val).
def sample_dist(x,y,delta=1,max=2000,null_val=0) :
    x0=delta

    xsam = np.arange(0,max,delta)
    ysam = np.ones(int(max/delta))*null_val
    nind=-1
    for it in range(len(x)) :
        if x0>max :
            break
        xi = x[it]
        yi = y[it]
        if xi < x0 :
            continue
        nind+=1
        xsam[nind] = xi
        ysam[nind] = yi
        x0+=delta
    return (xsam,ysam)

# Maybe try binning the normalized amplitude and frequency spectra, or averaging the values over a 
# number of slices, then using those "slices" as input? Might have to scale them by the max value to get
# any kind of sensible output.
def get_binned_values(s1,nbins=10,do_fabs=True,normalize=True) :
    if do_fabs :
        s1s = fabs(s1)
    else :
        s1s = s1
    if normalize :
        s1s_scal = fabs(s1s)/float(np.max(s1s))

    # Bin over e.g. 5 bins and compute averages
    delta    = int(len(s1s_scal)/float(nbins))
    ni       = 0
    avg_vals = []
    while ni+delta < len(s1s_scal) :
        avg = np.mean(s1s_scal[ni:(ni+delta)])
        avg_vals.append(avg)
        ni  = ni+delta 
    avg = np.mean(s1s_scal[ni:-1])
    avg_vals.append(avg)

    return avg_vals

#avals = get_binned_values(pows,10,False,True)
#pp.plot(avals,".-")
#pp.show()

def process_sound(wav_file) :

    # Need to convert to 16 bits first.
    wav_fo = wav_file.replace(".wav","-2.wav")
    data, samplerate = sf.read(wav_file)

    # Need to convert to 44100 for this to work properly
    sf.write(wav_fo, data, 44100)

    sampFreq, snd = wavfile.read(wav_fo) # load the data
    #print(snd.dtype)
    #print(sampFreq)

    #print("Duration: "+str(snd.shape[0]/float(sampFreq)))
    
    dur_sam = snd.shape[0]
    # Let's keep it mono
    try :
        s1 = snd[:,0]
    except :
        s1 = snd
#        os.system('rm "'+wav_fo+'"')
#        return (0,0,0)

    # Plot the tone
    timeArray = arange(0, dur_sam, 1)
    timeArray = timeArray / float(sampFreq)
    timeArray = timeArray * 1000  #scale to milliseconds

    # plot(timeArray, s1, color='k')
    # ylabel('Amplitude')
    # xlabel('Time (ms)')

    # Do an FFT
    n = len(s1) 
    p = fft(s1) # take the fourier transform 

    nUniquePts = int(ceil((n+1)/2.0))
    p = p[0:nUniquePts]
    p = abs(p)

    p = p / float(n) # scale by the number of points so that
    # the magnitude does not depend on the length 
    # of the signal or on its sampling frequency  
    p = p**2  # square it to get the power 

    # multiply by two (see technical document for details)
    # odd nfft excludes Nyquist point
    if n % 2 > 0: # we've got odd number of points fft
        p[1:len(p)] = p[1:len(p)] * 2
    else:
        p[1:len(p) -1] = p[1:len(p) - 1] * 2 # we've got even number of points fft

    freqArray = arange(0, nUniquePts, 1.0) * (sampFreq / float(n));
    xlabel('Frequency (kHz)')
    ylabel('Power (dB)')

    freqs = freqArray/1000
    pows  = 10*log10(p)

    avg_freq = np.average(freqs,weights=pows)
    max_freq = freqs[np.argmax(pows)]

    end_ind = int(len(timeArray))
    ts      = timeArray[0:end_ind]
    s1s     = s1[0:end_ind]

    try :
        p     = np.polyfit(ts,fabs(s1s),1)
        slope = p[0]
    except :
        slope = 0

    os.system('rm "'+wav_fo+'"')

    # Try normalizing these values.
    #arrs = [max_freq/freqs[-1],avg_freq/freqs[-1],slope/s1s[s1s.argmax()]]

    arrs = []

    # Get binned distributions of amplitude and frequency and
    # use those as variables.
    #abin = get_binned_values(s1,nbins=20,do_fabs=True,normalize=True) 
    #pbin = get_binned_values(pows,nbins=20,do_fabs=False,normalize=True)

    #tbin,abin = sample_dist(timeArray,s1,1,200)
    #fbin,pbin = sample_dist(freqs,pows,0.1,20)

    #arrs.extend(abin)
    #arrs.extend(pbin)

    s1=s1.astype(float)
    for i in range(len(s1)) :
        if s1[i] == 0. :
            s1[i] = np.random.rand()
        else :
            s1[i] = float(s1[i])
    
    fv,pv,fa,pa = get_freq_samples(s1,sampFreq,20000,2000)
    arrs.extend(pv.tolist())
    
    return arrs

# pp.plot(ts,fabs(s1s))
# pp.plot(ts,np.polyval(p,ts))
# pp.show()

# print("Slope of amplitude (/ms): "+str(p[0]))

