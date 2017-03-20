# -*- coding: utf8 -*-

"""
Simple pitch estimation
"""

from __future__ import print_function
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, medfilt
from numpy.fft import rfft



def parabolic(f, x):
    """
    Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    f is a vector and x is an index for that vector.
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def freq_from_fft(frame, fs):

    nsamples = len(frame)
    f = rfft(frame)

    i_peak = np.argmax(abs(f)) 
    i_interp = parabolic(np.log(abs(f)), i_peak)[0]

    #Compute the short-term energy of the frame
    energy = sum( [ abs(x)**2 for x in frame ] ) / nsamples 

    if energy > 10000:
        return fs * i_interp / nsamples 
    else:    
        return 0 
 
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, fs):
    order = 5   
    cutoff = 200
    b, a = butter_lowpass(cutoff, fs, order)
    y = lfilter(b, a, data)
    return y

def wav2f0(options, gui):
    with open(gui) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            filename = os.path.join(options.datadir, line + ".wav")
            f0_filename = os.path.join(options.datadir, line + ".f0")
            print("Processing:", filename, '->', f0_filename)
            rate, data = wavfile.read(filename)

            #Low pass filtering
            lp_filtered_data = butter_lowpass_filter(data,  rate)

            #Autocorrelation method
            with open(f0_filename, 'wt') as f0file:
                nsamples = len(data)
                fft_response_vector = []

                ns_windowlength = int(round((options.windowlength * rate) / 1000))
                ns_framelength = int(round((options.framelength * rate) / 1000))
                for ini in range(0, nsamples - ns_windowlength + 1, ns_framelength):

                    frame = lp_filtered_data[ini:ini+ns_windowlength]

                    f0 = freq_from_fft(frame, rate)
                    fft_response_vector.append(f0)

                #Median filter    
                median_filtered_data = medfilt(fft_response_vector,3)
                
                #Print the final values
                for i in range(0, median_filtered_data.size):
                    f1 = median_filtered_data[i]
                    print(f1, file=f0file)


def main(options, args):
    wav2f0(options, args[0])

if __name__ == "__main__":
    import optparse
    optparser = optparse.OptionParser(
        usage='%prog [OPTION]... FILELIST\n' + __doc__)
    optparser.add_option(
        '-w', '--windowlength', type='float', default=32,
        help='windows length (ms)')
    optparser.add_option(
        '-f', '--framelength', type='float', default=15,
        help='frame shift (ms)')
    optparser.add_option(
        '-d', '--datadir', type='string', default='data/fda_ue',
        help='data folder')

    options, args = optparser.parse_args()

    main(options, args)
