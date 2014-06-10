
                
def _buscaPics(alRescate, quantitat, distancia, left = 30, right = 30):
    pics = []
    from numpy import argmax, max, array,zeros, isnan, ndarray
    if type(alRescate) == list or type(alRescate) == array or type(alRescate) == ndarray:
        alRescate = array(alRescate)
        if len(alRescate) > 0:
            alRescate[:left] = 0
            alRescate[-right:] = 0    
            for i in range(quantitat):
                pos = argmax(alRescate)
                if pos > 11 and pos < len(alRescate) -11:
                    pics.append(pos)
                alRescate[max(pos-distancia,0):min(pos+distancia,len(alRescate))] = 0
    return pics

def _esBatec(sample, pics, entorn, thrs):
    from numpy import convolve, correlate,argmax, max, sqrt, dot, max

    new_beats = []
    
    for pic in pics:
        batecs_waveform = sample[pic-10:pic+20]
        batecs_waveform = batecs_waveform/sqrt(dot(batecs_waveform,batecs_waveform))
        sim = [dot(batecs_waveform,wave) for wave in entorn]
        if max(sim) > thrs:
            new_beats.append(pic)
    return new_beats

def getPercentil(entorn, quant_entorn, percentil):
    '''
    Selecciono grups representatius del 75% dels batecs
    '''
    from numpy import arange,shape
    grups_quality = sorted(zip(quant_entorn, arange(len(quant_entorn))), key = lambda x: x[0])
    entorn = [x for i,x in enumerate(entorn) if sum([quant_entorn[y[1]] for y in grups_quality[:i+1]])/sum(quant_entorn) < percentil]
    return entorn

def sampleBeatHunterLikelihood(signal, begin,  end, referencies, RR_estimat, index, **kwargs):
    '''
    referencies[0] media del grupo
    referencies[1] quantitat elements representats
    '''
    from numpy import mean, correlate, sum, median, diff,array
    percentil = 0.80
    referencies[0] = getPercentil(referencies[0],referencies[1], percentil)
    #alRescate = [correlate(signal[begin:end], model_beat, mode = 'same')[5:]*float(referencies[1][m])/sum(referencies[1]) for m, model_beat in enumerate(referencies[0])]
    alRescate = [correlate(signal[begin:end], model_beat, mode = 'valid') for m, model_beat in enumerate(referencies[0])]
    #return alRescate
    alRescate = mean(alRescate, 0)
    pics = array(_buscaPics(alRescate, int(round(float(end-begin+1)/RR_estimat + 2)), 15, left = 30, right = 30))+10
    candidats =  _esBatec(signal[begin:end], pics, referencies[0], index)
    return candidats

def sampleBeatHunterLikelihoodPFC(signal, begin,  end, referencies, RR_estimat, index, **kwargs):
    '''
    referencies[0] media del grupo
    referencies[1] quantitat elements representats
    '''
    #a,b = kmeans2(array(beats),20)
    from numpy import mean, correlate, sum, median, diff,array,max
    percentil = 0.8
    ref_utils = getPercentil(referencies[0],referencies[1], percentil)
    #alRescate = [correlate(signal[begin:end], model_beat, mode = 'same')[5:]*float(referencies[1][m])/sum(referencies[1]) for m, model_beat in enumerate(referencies[0])]
    alRescate = [correlate(signal[begin:end], model_beat, mode = 'valid') for m, model_beat in enumerate(ref_utils)]
    #return alRescate
    #alRescate = mean(alRescate, 0)
    alRescate = max(alRescate, 0)
    pics = array(_buscaPics(alRescate, int(round(float(end-begin+1)/RR_estimat + 2)), 20, left = RR_estimat*0.65, right = RR_estimat*0.65))+10
    candidats =  _esBatec(signal[begin:end], pics, referencies[0], index)
    
    from qrsDetector.utils import recoloca
    reload(recoloca)
    candidats2 = recoloca.recoloca2(signal, candidats)
    
    return candidats,candidats2

