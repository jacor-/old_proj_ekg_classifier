'''

A signal cleaning module has the objetive of clean the input signal. It's important to think on the complete structure of the project.
We want to do a iterative and incremental (more deep, more flexible) of the signal. 
It means that it's interesting to include a parameter (a threshold, for example) for implements this idea.

INPUT:

    signal: a numpy array-like structure where each arrayed element is a lead. So this array is, in fact, a rectangular matrix.

OUTPUT:

    signal: a numpy array-like structure where each arrayed element is a lead. So this array is, in fact, a rectangular matrix.
    parameter: some information related with the parameter (for example, the next one that has to be used)
    
NOTES:
    obviously, input and output signal don't necessarily have the same shape. BUT, be careful! Blocks connected to the output of this signal cleaner will see only the output of this block and it may introduce some kind of delay.

'''

import scikits.statsmodels.tsa.filters as sm

def sampleCleaner(signal, fm):
    '''
    1) clean signal!
    '''
    from numpy import cos, pi
    tall = 1./4/(1-cos(2*pi*fm/100))**2
    return sm.hpfilter(signal, tall)[0], fm + 1

