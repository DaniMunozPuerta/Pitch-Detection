import scipy.fftpack as sf
import numpy as np
from scipy.io import wavfile
def maxFrequency(X, F_sample, Low_cutoff=80, High_cutoff= 300):
        """ Searching presence of frequencies on a real signal using FFT
        Inputs
        =======
        X: 1-D numpy array, the real time domain audio signal (single channel time series)
        Low_cutoff: float, frequency components below this frequency will not pass the filter (physical frequency in unit of Hz)
        High_cutoff: float, frequency components above this frequency will not pass the filter (physical frequency in unit of Hz)
        F_sample: float, the sampling frequency of the signal (physical frequency in unit of Hz)
        """        

        M = X.size # let M be the length of the time series
        Spectrum = sf.rfft(X, n=M) 
        [Low_cutoff, High_cutoff, F_sample] = map(float, [Low_cutoff, High_cutoff, F_sample])

        #Convert cutoff frequencies into points on spectrum
        [Low_point, High_point] = map(lambda F: F/F_sample * M, [Low_cutoff, High_cutoff])

        maximumFrequency = np.where(Spectrum == np.max(Spectrum[Low_point : High_point])) # Calculating which frequency has max power.

        return maximumFrequency

voiceVector = []
filename = "/data/fda_ue/rl001.wav"
rate, data = wavfile.read("rl001.wav")
for window in data: # Run a window of appropriate length across the audio file
    voiceVector.append (maxFrequency( window, rate))
    window+20