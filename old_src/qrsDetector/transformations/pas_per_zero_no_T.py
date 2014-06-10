
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
    from numpy import array, abs, max, min,mod, sqrt, dot, convolve
    sig_original = array(sig_original_input).astype('float')
    haar = array([1,-1]).astype('float')
    haar = haar/sqrt(dot(haar,haar))
    
    sig_original = convolve(sig_original, haar)
    
    from signalCleaning.sample import sampleCleaner
    sig_otiginal,aux = sampleCleaner(sig_original, 1)
    perfil = abs(array([max(sig_original[max([i-200,0]):min([i+200,len(sig_original)])]) for i,x in enumerate(sig_original)])).astype('float')/4
    for i,x in enumerate(range(len(sig_original))):
        if mod(i,2) == 0:
            sig_original[i] = sig_original[i]-perfil[i]
        else:
            sig_original[i] = sig_original[i]+perfil[i]
    
    transf = __passosPerZero__(sig_original, finestra = 20)
    return transf

def __passosPerZero__(sig, finestra = 20):
    from numpy import diff, array, sum
    z = []
    for i in range(len(sig)-1):
        z.append(sig[i]*sig[i+1])
    z.append(z[-1])
    z = array(z)
    z = (z <= 0)
    z2 = []
    f = finestra/2
    for i in range(len(sig)):
        z2.append(sum(z[i-f:i+f]))
        
    from numpy import diff,ones,convolve,array, shape, argmax
    import scipy.signal as signal
    z2 = array(z2).astype('float')
    b,a = signal.iirfilter(2, 10./100, ftype = 'butter',btype='lowpass')
    z2 = signal.lfilter(b,a,z2)
    z2 = 1./(z2+1)
    return z2-min(z2)