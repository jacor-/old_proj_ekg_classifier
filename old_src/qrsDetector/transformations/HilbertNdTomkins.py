


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
    from numpy import diff,ones,convolve,array, shape, argmax, fft,mod

    '''
    1) do a lot of things
    2) fill res structure
    '''
    sig = array(sig_original_input).astype('float')
    hilbert = lambda x: fft.ifft(-1j*array( list(ones(len(x)/2))+list(ones(len(x)/2)*-1))*fft.fft(x))    
    if mod(len(sig),2) == 0:
        sig = hilbert(sig)
    else:
        sig = hilbert(sig[:-1])
    b,a = signal.iirfilter(2, [10./100,30./100], ftype = 'butter',btype='bandpass')
    sig = signal.lfilter(b,a,array(sig))
    sig = diff(sig)
    sig = sig*sig
    pulse = ones(12)/12
    sig2 = convolve(sig, pulse)
    return sig2[12:]
    