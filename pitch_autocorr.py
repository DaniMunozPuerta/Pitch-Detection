# -*- coding: utf8 -*-

"""
Simple pitch estimation
"""

from __future__ import print_function
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, butter, lfilter, freqz, medfilt

def autocorr_method(frame, rate):
    """Estimate pitch using autocorrelation
    """
    defvalue = (0.0, 1.0)

    # Calculate autocorrelation using scipy correlate
    frame = frame.astype(np.float)
    frame -= frame.mean()
    amax = np.abs(frame).max()
    if max > 0:
        frame /= amax
    else:
        return defvalue

    corr = correlate(frame, frame)
    # keep the positive part
    corr = corr[len(corr)/2:]

    # Find the first minimum
    dcorr = np.diff(corr)
    rmin = np.where(dcorr > 0)[0]
    if len(rmin) > 0:
        rmin1 = rmin[0]
    else:
        return defvalue

    # Find the next peak
    peak = np.argmax(corr[rmin1:]) + rmin1
    rmax = corr[peak]/corr[0]
    f0 = rate / peak

    if rmax > 0.6 and f0 > 50 and f0 < 550:
        return f0
    else:
        return 0;    


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

                autocorr_response_vector = []

                # From miliseconds to samples
                ns_windowlength = int(round((options.windowlength * rate) / 1000))
                ns_framelength = int(round((options.framelength * rate) / 1000))
                for ini in range(0, nsamples - ns_windowlength + 1, ns_framelength):
                    frame = lp_filtered_data[ini:ini+ns_windowlength]
                    f0 = autocorr_method(frame, rate)
                    autocorr_response_vector.append(f0) 

                #Median filter    
                median_filtered_data = medfilt(autocorr_response_vector,3)
                
                #Finally we print the results into the f0 file
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
