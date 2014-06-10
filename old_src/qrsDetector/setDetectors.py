from qrsDetector.transformations import panTomkins
from qrsDetector.transformations import haar16
from qrsDetector.transformations import HilbertNdTomkins
from qrsDetector.transformations import pas_per_zero
from qrsDetector.transformations import pas_per_zero_no_T

from qrsDetector.thresholds import adaptive
from qrsDetector.thresholds import adaptive_exp
from qrsDetector.thresholds import fix



from qrsDetector.utils import getMaxs
from qrsDetector.utils import recoloca


from signalCleaning.sample import sampleCleaner

reload(adaptive)
reload(adaptive_exp)
reload(fix)

reload(recoloca)
reload(getMaxs)


def detectTomkinsAdaptive(data):
    return __detectGeneric__(data, panTomkins, getMaxs, adaptive)

def detectTomkinsAdaptiveExp(data):
    return __detectGeneric__(data, panTomkins, getMaxs, adaptive_exp)

def detectTomkinsFix(data):
    return __detectGeneric__(data, panTomkins, getMaxs, fix)


def detectHaar16Adaptive(data):
    return __detectGeneric__(data, haar16, getMaxs, adaptive)

def detectHaar16AdaptiveExp(data):
    return __detectGeneric__(data, haar16, getMaxs, adaptive_exp)

def detectHaar16Fix(data):
    return __detectGeneric__(data, haar16, getMaxs, fix)


def detectHilbertAdaptive(data):
    return __detectGeneric__(data, HilbertNdTomkins, getMaxs, adaptive)

def detectHilbertAdaptiveExp(data):
    return __detectGeneric__(data, HilbertNdTomkins, getMaxs, adaptive_exp)

def detectHilbertFix(data):
    return __detectGeneric__(data, HilbertNdTomkins, getMaxs, fix)


def detectZeroCrossAdaptive(data):
    return __detectGeneric__(data, pas_per_zero_no_T, getMaxs, adaptive)

def detectZeroCrossAdaptiveExp(data):
    return __detectGeneric__(data, pas_per_zero_no_T, getMaxs, adaptive_exp)

def detectZeroCrossFix(data):
    return __detectGeneric__(data, pas_per_zero_no_T, getMaxs, fix)



def __detectGeneric__(data, transformator, getMaxs, thrss):
    
    
    reload(adaptive)
    reload(adaptive_exp)
    reload(fix)
    
    reload(recoloca)
    reload(getMaxs)

    
    
    
    
    
    #import ipdb
    #ipdb.set_trace()
    
    
    
    data,aux = sampleCleaner(data, 1)

    sig = transformator.transformation(data)
    maxs = getMaxs.maxs_rel_positions(sig)
    
    
    thrs = thrss.thrs(sig) 
    
    
    detected_beats = []
    #xxx = []
    #last_xxx = 0
    #l_thrs = []
    from numpy import isnan
    for x in maxs:
        #for i in range(x-last_xxx):
            #xxx.append(thrs.getThrs(i+last_xxx))
        #last_xxx = x
        if isnan(thrs.getThrs(x)):
            import ipdb
            ipdb.set_trace()
        
        if sig[x] > thrs.getThrs(x):
            detected_beats.append(x)
            thrs.setNewPos(x)
        #l_thrs.append([x,thrs.getThrs(x)])
    
    det2 = recoloca.recoloca(data, detected_beats)
    #xxx = [min([500,x]) for x in xxx]
    
    #plot(data)
    #plot(det2, [data[x] for x in det2],'o')
    return det2#,l_thrs

