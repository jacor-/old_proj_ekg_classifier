
'''
INPUT:

A qrsDetector takes a signal: a numpy array-like structure where each arrayed element is a lead. So this array is, in fact, a rectangular matrix.

OUTPUT:

The output of a qrsDetector is an instance of "structure_QrsCandidates". The information it contains is:
   - outputData: data after the detection (for example, it can be transformed, lead combined...)
   - beatPosition: beats detected in EACH ROW of outputData. So, it's an array with the same rows as signal input and it may not be rectangular (different number of beats can be deteted in different leads).  
   - beatInformation: for each beatDetected and informed in beatPosition, information that can be important (for example, a confidence parameter).
'''
from distutils.command.clean import clean

def transformation(sig_original_input):
    import scipy.signal as signal
    from numpy import diff,ones,convolve,array, shape, argmax, sqrt, dot


    sig = array(sig_original_input).astype('float')

    haar = array([1]*16+[-1]*16).astype('float')
    haar = haar / sqrt(dot(haar,haar))
    sig = convolve(sig,haar)
    pulse = ones(8)/sqrt(8)
    sig = convolve(abs(sig), pulse)
    
    import scipy.signal as signal
    b,a = signal.iirfilter(2, 5./100, ftype = 'butter',btype='lowpass')
    sig = signal.filtfilt(b,a,sig)
    return sig[31:]
