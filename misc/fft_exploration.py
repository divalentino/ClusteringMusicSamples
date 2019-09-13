
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


# In[202]:


import soundfile as sf

#wav_file = 'ASC2_Tight_Snare_2.wav'
wav_file = 'Cymatics - Ultimate Snare 36 - G#.wav'
wav_fo   = wav_file.replace(".wav","-2.wav")

data, samplerate = sf.read(wav_file)
sf.write(wav_fo, data, samplerate)

sampFreq, snd = wavfile.read(wav_fo) # load the data
print(snd.dtype)
print(sampFreq)


# In[203]:


print("Duration: "+str(snd.shape[0]/float(sampFreq)))
dur_sam = snd.shape[0]


# In[204]:


# Let's keep it mono
s1 = snd[:,0] 


# In[205]:


# Plot the tone
timeArray = arange(0, dur_sam, 1)
timeArray = timeArray / float(sampFreq)
timeArray = timeArray * 1000  #scale to milliseconds

plot(timeArray, s1, color='k')
ylabel('Amplitude')
xlabel('Time (ms)')


# In[212]:


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
plot(freqArray/1000, 10*log10(p), color='k')
xlabel('Frequency (kHz)')
ylabel('Power (dB)')

freqs = freqArray/1000
pows  = 10*log10(p)


# In[207]:


import matplotlib.pyplot as pp

hbins=np.arange(0,20,0.1)
pp.hist(freqs,weights=pows,bins=hbins)
pp.show()


# In[220]:


avg_freq = np.average(freqs,weights=pows)
print("Average frequency (kHz): "+str(avg_freq))
print("Peak frequency (kHz): "+str(freqs[np.argmax(pows)]))


# **Question:** Can we use rate of change of amplitude w.r.t. time an an approximation for "punchiness"? Maybe Simpson's rule?

# In[211]:


end_ind = int(len(timeArray))
ts      = timeArray[0:end_ind]
s1s     = s1[0:end_ind]

p = np.polyfit(ts,fabs(s1s),1)
pp.plot(ts,fabs(s1s))
pp.plot(ts,np.polyval(p,ts))
pp.show()

print("Slope of amplitude (/ms): "+str(p[0]))

