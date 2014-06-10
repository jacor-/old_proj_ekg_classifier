import scikits.statsmodels.tsa.filters as sm
import scipy.signal as signal

def hp(signal, fm):
    '''
    1) clean signal!
    '''
    from numpy import cos, pi
    tall = 1./4/(1-cos(2*pi*fm/100))**2
    return sm.hpfilter(signal, tall)[0]

def bandpass(sigin, f1,f2,fm=100):
    from numpy import cos, pi,array
    sig = array(sigin).astype('float')
    b,a = signal.iirfilter(2, [float(f1)/fm,float(f2)/fm], ftype = 'butter',btype='bandpass')
    sig = signal.lfilter(b,a,array(sig))
    return sig

def highpass(sigin, f1,fm=100):
    from numpy import cos, pi,array
    sig = array(sigin).astype('float')
    b,a = signal.iirfilter(2, float(f1)/fm, ftype = 'butter',btype='highpass')
    sig = signal.lfilter(b,a,array(sig))
    return sig

def lowpass(sigin, f1,fm=100):
    from numpy import cos, pi,array
    sig = array(sigin).astype('float')
    b,a = signal.iirfilter(2, float(f1)/fm, ftype = 'butter',btype='lowpass')
    sig = signal.lfilter(b,a,array(sig))
    return sig

def a_saquisimo(sig, points, labels):
    from numpy import argmax, argmin, max, min, mean,array
    from scipy.interpolate import splev,splrep
    
    sig = hp(sig, 1.5)
    #sig = lowpass(abs(highpass(sig,50.)),10.)[5:]*sig[:-5]
    
    
    #values_m_0 = [mean(sig[pos-20:pos+40]) for i,pos in enumerate(points)  if labels[i] in 'SN']
    #tck = splrep([pos for i,pos in enumerate(points)  if labels[i] in 'SN'], values_m_0)
    #means_0 = splev(range(len(sig)),tck,der=0)
    #sig = sig-means_0
    
    values_0_max = abs(array([sig[argmax(sig[pos-10:pos+10])+pos-10] for i,pos in enumerate(points)  if labels[i] in 'SN']))
    values_0_min = abs(array([sig[argmin(sig[pos-10:pos+10])+pos-10] for i,pos in enumerate(points)  if labels[i] in 'SN']))
    
    tck = splrep([pos for i,pos in enumerate(points)  if labels[i] in 'SN'], values_0_max)
    envolve_max = splev(range(len(sig)),tck,der=0)
    for i in len(envolve_max):
        envolve_max[i] = 1.
    
    
    tck = splrep([pos for i,pos in enumerate(points)  if labels[i] in 'SN'], values_0_min)
    envolve_min = splev(range(len(sig)),tck,der=0)
    for i in len(envolve_min):
        envolve_min[i] = 1.
    
    
    aux = array([sig/envolve_min,sig/envolve_max])
    return array([aux[x][i] for i,x in enumerate(sig>0)])
    
def normalizer(sig, points, labels):
    from numpy import argmax, argmin, max, min, mean,array,median,zeros,arange
    from scipy.interpolate import splev,splrep


    values_0_max = [max(abs(sig[pos-5:pos+5])) for i,pos in enumerate(points)  ]
    thrs = array([median(values_0_max[max([0,i-10]):i+10]) for i in range(len(points))])
    
    entorn = zeros(len(sig))
     
     
    for j in range(points[0]):
        entorn[j] = thrs[0]
    for j in range(len(sig)-points[-1]):
        entorn[points[-1]+j] = thrs[-1]
    
    for i in range(len(points)-1):
        entorn[points[i]] = thrs[i]
        entorn[points[i+1]] = thrs[i+1]
        if points[i+1] == points[i]:
            continue
        c = float(thrs[i+1]-thrs[i])/float(points[i+1]-points[i])
        for j in range(points[i+1]-points[i]):
            #print str(j*c+entorn[i])
            entorn[j+1+points[i]] = j*c+entorn[points[i]]
            
    
    return sig/entorn    
    
    
    
    
    
    

    
