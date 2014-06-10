
from numpy import array,sqrt,dot,argmax,max,argmin,min,mean,median
from system.settings import *
import signalCleaning.sample as cl
from mdp.nodes import JADENode 
from system.settings import *

cases = h_io.get_usable_cases(REDIAGNOSE_DIAGNOSER)
from time import time
import beatWork.characterization.waveform_characterizers as chr

###Caracteritzadors Normalitzats!!! PFC

def __chr_amplitude_one_lead__(sig,valid_labs):
    import pywt
    wave2 = pywt.cwt.CWavelet('gauss2')
    a2 = wave2.wavefun(2)[0]
    from numpy import convolve,std,zeros
    import signalCleaning.cleaners as cl2
    sig0_01 = convolve(sig,a2,mode='same')
    sig0_1 = convolve(cl2.bandpass(sig0_01,10,25),[1,-1])[1:]
    sig0_1 = sig0_1-cl2.hp(sig0_1,3)
    
    cars = chr.area(sig0_01*sig0_1,valid_labs[0])
    del sig0_1
    del sig0_01
    cars_thrs_mean = array([median(cars[max([i-100,0]):min([i+100,len(cars)])]) for i in range(len(cars))])
    cars = abs(cars / cars_thrs_mean - 1)
    return cars

def __chr_morph_one_lead__(sig,valid_labs):
    waverefs = [x-10+argmax(sig[x-10:x+10]) for x in valid_labs[0]]
    waverefs = [sig[x-20:x+30] for x in waverefs]
    waverefs = [x-mean(x) for x in waverefs]
    waverefs = [x/sqrt(dot(x,x)) for x in waverefs]
    
    likelihood_pos = [mean(dot(waverefs[i],array(waverefs[i-25:i]+waverefs[i+1:i+25]).transpose())) for i in range(len(valid_labs[0]))]
    
    likelihood_vals = array(likelihood_pos)
    cars_thrs_mean = (array([median(likelihood_vals[max([i-100,0]):min([i+100,len(likelihood_vals)])]) for i in range(len(likelihood_vals))]))
    #cars_thrs_mean = max(array([cars_thrs_mean+std(cars_thrs_mean)*2,ones(len(cars_thrs_mean))-0.75]),0)
    
    
    #cars_thrs_mean_norm = abs(cl2.hp(array(likelihood_vals),3)-1)
    cars_thrs_mean_norm = abs(likelihood_vals/cars_thrs_mean-1)
    return cars_thrs_mean_norm


def __chr_amplada_per_lead__(sig, valid_labs, parabolas):
    from numpy import correlate,sign,median,array
    amplada1 = array([argmax(array([sign(max(sig[pos_beat-5:pos_beat+5]))*max(correlate(sig[pos_beat-10:pos_beat+10],par,mode='full'))   for par in parabolas]))  for pos_beat in valid_labs[0] ])
    cars_amplada_mean = array([median(amplada1[i-100:i+100]) for i in range(len(valid_labs[0]))])
    cars_thrs_mean_norm = abs(amplada1/cars_amplada_mean-1)
    return cars_thrs_mean_norm

def __chr__getRRCriteria(data,valid_labs):
    rr = estimator.estimate(valid_labs[0],0,len(data[0]),ENTORN=10)
    rr2 = [0]+list(array([float(valid_labs[0][i+2]-valid_labs[0][i+1])/(valid_labs[0][i+1]-valid_labs[0][i]) for i in range(len(valid_labs[0])-2)])-1)+[0]
    
    rr_low = rr > 0.15
    rr_hight = rr < -0.15
    jmps = rr_low.astype('int')-rr_hight.astype('int')
    
    return jmps


















def fes_grups(sig,posicions,thrs):
    from numpy import array,sqrt,dot,argmax,max,argmin,min,mean,median,correlate
    posicions2 = [argmax(sig[posicions[i]-5:posicions[i]+5])+posicions[i]-5 for i in range(len(posicions))]
    beats = [sig[posicions2[i]-10:posicions2[i]+30] for i in range(len(posicions2))]
    beats = [i-mean(i) for i in beats]
    beats = [i/sqrt(dot(i,i)) for i in beats]    
    
    pos = []
    medias = []
    pos.append([posicions[0]])
    medias.append(beats[0])
    for x in range(len(beats)-1):
        thr_empirical = [max(dot(beats[x+1],zz)) for zz in medias ]
        if max(thr_empirical) > thrs:
            medias[argmax(thr_empirical)] = medias[argmax(thr_empirical)]*0.7+0.3*beats[x+1]
            medias[argmax(thr_empirical)] = medias[argmax(thr_empirical)] /sqrt(dot(medias[argmax(thr_empirical)],medias[argmax(thr_empirical)]))
            pos[argmax(thr_empirical)].append(posicions[x+1])
        else:
            medias.append(beats[x+1])
            pos.append([posicions[x+1]])
    return medias,pos


def getResults(vs, ks):
    vs_detected_aplitud2 = sum([1 for x in ks  if x[0] in vs])
    total_detected_amplitud2 = len(ks)
    return vs_detected_aplitud2, total_detected_amplitud2

def getResultAsVector(ks,valid_labs):
    from numpy import zeros
    z = zeros(len(valid_labs[0]))
    kss = [x[0] for x in ks]
    vs_detected = [i for i,x in enumerate(valid_labs[0]) if x in kss]
    for i in vs_detected:
        z[i] = 1.
    return z


def getAmplitudeCriteria(data,valid_labs):
    import pywt
    wave2 = pywt.cwt.CWavelet('gauss2')
    a2 = wave2.wavefun(2)[0]


    def __amplitude_one_lead__(sig,valid_labs):
    #lead C1
        from numpy import convolve
        sig0_01 = convolve(sig,a2,mode='same')
        sig0_1 = convolve(cl2.bandpass(sig0_01,10,25),[1,-1])[1:]
        sig0_1 = sig0_1-cl2.hp(sig0_1,3)
        
        cars = chr.area(sig0_01*sig0_1,valid_labs[0])
        del sig0_1
        del sig0_01
        cars_thrs_mean = array([mean(cars[max([i-200,0]):min([i+200,len(cars)])]) for i in range(len(cars))])*3
        ks_amplitud0 = [[valid_labs[0][i],valid_labs[1][i]]  for i,x in enumerate(cars) if x < cars_thrs_mean[i]]
        ks_amplitud0 = sorted(ks_amplitud0,key=lambda x:x[0])
        del cars_thrs_mean
        return ks_amplitud0
    
    ks_amplitud0 = __amplitude_one_lead__(data[0], valid_labs)
    ks_amplitud1 = __amplitude_one_lead__(data[1], valid_labs)
    ks_amplitud2 = __amplitude_one_lead__(data[2], valid_labs)
    
    vs_detected_amplitud0,total_detected_amplitud0 =  getResults(vs, ks_amplitud0)
    vs_detected_amplitud1,total_detected_amplitud1 =  getResults(vs, ks_amplitud1)
    vs_detected_amplitud2,total_detected_amplitud2 =  getResults(vs, ks_amplitud2)    
    print case + "  Amplitud C1        " + str(vs_detected_amplitud0) + " of " + str(len(vs)) + "       saw   " + str(total_detected_amplitud0)
    print case + "  Amplitud C2        " + str(vs_detected_amplitud1) + " of " + str(len(vs)) + "       saw   " + str(total_detected_amplitud1)
    print case + "  Amplitud C3        " + str(vs_detected_amplitud2) + " of " + str(len(vs)) + "       saw   " + str(total_detected_amplitud2)

    
    return [ks_amplitud0, ks_amplitud1, ks_amplitud2]

def getAmplitudeCriteria3(data,valid_labs):
    


    def __amplitude_one_lead__(sig,valid_labs):
    #lead C1
        import pywt
        wave2 = pywt.cwt.CWavelet('gauss2')
        a2 = wave2.wavefun(2)[0]
        from numpy import convolve,std,zeros
        import signalCleaning.cleaners as cl2
        sig0_01 = convolve(sig,a2,mode='same')
        sig0_1 = convolve(cl2.bandpass(sig0_01,10,25),[1,-1])[1:]
        sig0_1 = sig0_1-cl2.hp(sig0_1,3)
        
        cars = chr.area(sig0_01*sig0_1,valid_labs[0])
        del sig0_1
        del sig0_01
        cars_thrs_mean = array([median(cars[max([i-100,0]):min([i+100,len(cars)])]) for i in range(len(cars))])
        cars = abs(cars / cars_thrs_mean - 1)
        cars_thrs_mean = zeros(len(cars_thrs_mean))+std(cars_thrs_mean)*5

        ks_amplitud0 = [[valid_labs[0][i],valid_labs[1][i]]  for i,x in enumerate(cars) if x > 1.25]
        ks_amplitud0 = sorted(ks_amplitud0,key=lambda x:x[0])
        del cars_thrs_mean
        return ks_amplitud0
    
    ks_amplitud0 = __amplitude_one_lead__(data[0], valid_labs)
    ks_amplitud1 = __amplitude_one_lead__(data[1], valid_labs)
    ks_amplitud2 = __amplitude_one_lead__(data[2], valid_labs)
    
    vs_detected_amplitud0,total_detected_amplitud0 =  getResults(vs, ks_amplitud0)
    vs_detected_amplitud1,total_detected_amplitud1 =  getResults(vs, ks_amplitud1)
    vs_detected_amplitud2,total_detected_amplitud2 =  getResults(vs, ks_amplitud2)    
    print case + "  Amplitud C1   3    " + str(vs_detected_amplitud0) + " of " + str(len(vs)) + "       saw   " + str(total_detected_amplitud0)
    print case + "  Amplitud C2   3    " + str(vs_detected_amplitud1) + " of " + str(len(vs)) + "       saw   " + str(total_detected_amplitud1)
    print case + "  Amplitud C3   3    " + str(vs_detected_amplitud2) + " of " + str(len(vs)) + "       saw   " + str(total_detected_amplitud2)

    
    return [ks_amplitud0, ks_amplitud1, ks_amplitud2]

def getAmplitudeCriteria2(data,valid_labs):
    import pywt
    wave2 = pywt.cwt.CWavelet('gauss2')
    a2 = wave2.wavefun(2)[0]
    
    
    def __amplitude_one_lead__(sig,valid_labs):
        #lead C1
        from numpy import convolve,abs,array
        
        cars = [max(abs(sig[x-5:x+5])) for x in valid_labs[0]]
        cars_thrs_mean = array([mean(cars[max([i-20,0]):min([i+20,len(cars)])]) for i in range(len(cars))])
        cars = abs(cars/cars_thrs_mean -1)
        ks_amplitud0 = [[valid_labs[0][i],valid_labs[1][i]]  for i,x in enumerate(cars) if x > 0.15]
        ks_amplitud0 = sorted(ks_amplitud0,key=lambda x:x[0])
        del cars_thrs_mean
        return ks_amplitud0
    
    ks_amplitud0 = __amplitude_one_lead__(data[0], valid_labs)
    ks_amplitud1 = __amplitude_one_lead__(data[1], valid_labs)
    ks_amplitud2 = __amplitude_one_lead__(data[2], valid_labs)
    
    vs_detected_amplitud0,total_detected_amplitud0 =  getResults(vs, ks_amplitud0)
    vs_detected_amplitud1,total_detected_amplitud1 =  getResults(vs, ks_amplitud1)
    vs_detected_amplitud2,total_detected_amplitud2 =  getResults(vs, ks_amplitud2)    
    print case + "  Amplitud C1  Jose  " + str(vs_detected_amplitud0) + " of " + str(len(vs)) + "       saw   " + str(total_detected_amplitud0)
    print case + "  Amplitud C2  Jose  " + str(vs_detected_amplitud1) + " of " + str(len(vs)) + "       saw   " + str(total_detected_amplitud1)
    print case + "  Amplitud C3  Jose  " + str(vs_detected_amplitud2) + " of " + str(len(vs)) + "       saw   " + str(total_detected_amplitud2)


    return [ks_amplitud0, ks_amplitud1, ks_amplitud2]


def getRRCriteria(data,valid_labs):
    rr = estimator.estimate(valid_labs[0],0,len(data[0]),ENTORN=10)
    rr2 = [0]+list(array([float(valid_labs[0][i+2]-valid_labs[0][i+1])/(valid_labs[0][i+1]-valid_labs[0][i]) for i in range(len(valid_labs[0])-2)])-1)+[0]
    
    rr_low = rr > 0.15
    rr_hight = rr < -0.15
    jmps = rr_low.astype('int')-rr_hight.astype('int')
    

    ks2 = [[valid_labs[0][i],valid_labs[1][i]] for i in range(len(jmps)-1) if jmps[i] == -1 and jmps[i+1] == 1]
    #ks = [[valid_labs[0][i],valid_labs[1][i]] for i in range(len(rr2)) if rr2[i] > 0.3 and not (jmps[i] == -1 and jmps[i-1] == -1)]
    ks = [[valid_labs[0][i],valid_labs[1][i]] for i in range(len(rr2)) if rr2[i] > 0.3 ]
    ks_fora_temps = [[valid_labs[0][i-1],valid_labs[1][i-1]] for i in range(len(jmps)) if jmps[i] == -1 and jmps[i-1] == -1]
    
    ks = sorted(ks + ks_fora_temps,key=lambda x:x[0])

    vs_detected_rr,total_detected_rr = getResults(vs, ks)    
    print case + "  RR                 " + str(vs_detected_rr) + " of " + str(len(vs)) + "       saw   " + str(total_detected_rr)

    return [ks,ks,ks]
    
def getMorphologyCriteria(data,valid_labs):
    from numpy import correlate,std,ones,min,max
    
    
    def __morph_one_lead__(sig,valid_labs):
        waverefs = [x-10+argmax(sig[x-10:x+10]) for x in valid_labs[0]]
        waverefs = [sig[x-20:x+30] for x in waverefs]
        waverefs = [x-mean(x) for x in waverefs]
        waverefs = [x/sqrt(dot(x,x)) for x in waverefs]
        
        likelihood_pos = [mean(dot(waverefs[i],array(waverefs[i-25:i]+waverefs[i+1:i+25]).transpose())) for i in range(len(valid_labs[0]))]
        
        likelihood_vals = array(likelihood_pos)
        cars_thrs_mean = (array([median(likelihood_vals[max([i-100,0]):min([i+100,len(likelihood_vals)])]) for i in range(len(likelihood_vals))]))
        #cars_thrs_mean = max(array([cars_thrs_mean+std(cars_thrs_mean)*2,ones(len(cars_thrs_mean))-0.75]),0)
        
        
        #cars_thrs_mean_norm = abs(cl2.hp(array(likelihood_vals),3)-1)
        cars_thrs_mean_norm = abs(likelihood_vals/cars_thrs_mean-1)
        
        ks_morphology = [valid_labs[0][i] for i,x in enumerate(valid_labs[0]) if cars_thrs_mean_norm[i] > 0.3]        
        ks_morphology = [[x, 'V'] for x in ks_morphology]
        return ks_morphology
    
    
    ks_morphology = __morph_one_lead__(data[0], valid_labs)
    ks_morphology1 = __morph_one_lead__(data[1], valid_labs)
    ks_morphology2 = __morph_one_lead__(data[2], valid_labs)
    
    
    vs_detected_morphology0,total_detected_morphology0 =  getResults(vs, ks_morphology)
    vs_detected_morphology1,total_detected_morphology1 =  getResults(vs, ks_morphology1)
    vs_detected_morphology2,total_detected_morphology2 =  getResults(vs, ks_morphology2)    
    print case + "  morphology C1      " + str(vs_detected_morphology0) + " of " + str(len(vs)) + "       saw   " + str(total_detected_morphology0)
    print case + "  morphology C2      " + str(vs_detected_morphology1) + " of " + str(len(vs)) + "       saw   " + str(total_detected_morphology1)
    print case + "  morphology C3      " + str(vs_detected_morphology2) + " of " + str(len(vs)) + "       saw   " + str(total_detected_morphology2)

    
    return [ks_morphology,ks_morphology1,ks_morphology2]

def getMorphologyCriteriaAndAmplitude(data,valid_labs):
    from numpy import correlate,std,ones,min,max
    ks_1 = getMorphologyCriteria(data, valid_labs)
    ks_2 = getAmplitudeCriteria2(data, valid_labs)
    candidates_1 = [x[0] for x in ks_1[0]]
    candidates_2 = [x[0] for x in ks_1[1]]
    candidates_3 = [x[0] for x in ks_1[2]]

    ks_cosa1 = [[x[0],'V'] for x in ks_2[0] if x[0] in candidates_1]
    ks_cosa2 = [[x[0],'V'] for x in ks_2[1] if x[0] in candidates_2]
    ks_cosa3 = [[x[0],'V'] for x in ks_2[2] if x[0] in candidates_3]
    
    
    vs_detected_morphology0,total_detected_morphology0 =  getResults(vs, ks_cosa1)
    vs_detected_morphology1,total_detected_morphology1 =  getResults(vs, ks_cosa2)
    vs_detected_morphology2,total_detected_morphology2 =  getResults(vs, ks_cosa3)    
    print case + " morphology mezcla C1" + str(vs_detected_morphology0) + " of " + str(len(vs)) + "       saw   " + str(total_detected_morphology0)
    print case + " morphology mezcla C2" + str(vs_detected_morphology1) + " of " + str(len(vs)) + "       saw   " + str(total_detected_morphology1)
    print case + " morphology mezcla C3" + str(vs_detected_morphology2) + " of " + str(len(vs)) + "       saw   " + str(total_detected_morphology2)

    
    return [ks_cosa1,ks_cosa2,ks_cosa3]


def getAmpladaCriteria(data,valid_labs):
    from numpy import arange,sign, correlate
    def parabola(c):
        paux = array([-1.*(float(x*x)/c/c-1) for x in arange(-c,c+1,1)]).astype('float')
        paux = paux/sqrt(dot(paux,paux))
        return paux
    
    parabolas = [parabola(i+2) for i in range(15)]    
    
    def __amplada_per_lead__(sig, valid_labs, parabolas):        
        amplada1 = [argmax(array([sign(max(sig[pos_beat-5:pos_beat+5]))*max(correlate(sig[pos_beat-10:pos_beat+10],par,mode='full'))   for par in parabolas]))  for pos_beat in valid_labs[0] ]
        cars_amplada_mean = array([mean(amplada1[i-100:i+100]) for i in range(len(valid_labs[0]))])+0.98
        ks_amplada1 = [valid_labs[0][i] for i,x in enumerate(valid_labs[0]) if amplada1[i] > cars_amplada_mean[i]]
        ks_amplada1 = [[x, 'V'] for x in ks_amplada1]
        return ks_amplada1
    
    ks_amplada1 = __amplada_per_lead__(data[0], valid_labs, parabolas)
    ks_amplada2 = __amplada_per_lead__(data[1], valid_labs, parabolas)
    ks_amplada3 = __amplada_per_lead__(data[2], valid_labs, parabolas)
    
    

    vs_detected_amplada0,total_detected_amplada0 =  getResults(vs, ks_amplada1)
    vs_detected_amplada1,total_detected_amplada1 =  getResults(vs, ks_amplada2)
    vs_detected_amplada2,total_detected_amplada2 =  getResults(vs, ks_amplada3)    
    print case + "  Anchura  C1        " + str(vs_detected_amplada0) + " of " + str(len(vs)) + "       saw   " + str(total_detected_amplada0)
    print case + "  Anchura  C2        " + str(vs_detected_amplada1) + " of " + str(len(vs)) + "       saw   " + str(total_detected_amplada1)
    print case + "  Anchura  C3        " + str(vs_detected_amplada2) + " of " + str(len(vs)) + "       saw   " + str(total_detected_amplada2)
    
    return [ks_amplada1,ks_amplada2, ks_amplada3]

def getAmpladaCriteria2(data,valid_labs):
    from numpy import arange,sign, correlate
    def parabola(c):
        paux = array([-1.*(float(x*x)/c/c-1) for x in arange(-c,c+1,1)]).astype('float')
        paux = paux/sqrt(dot(paux,paux))
        return paux
    
    parabolas = [parabola(i+2) for i in range(15)]    
    
    def __amplada_per_lead__(sig, valid_labs, parabolas):        
        amplada1 = [max([argmax(array([sign(max(sig[pos_beat-5:pos_beat+5]))*max(correlate(sig[pos_beat-10:pos_beat+10],par,mode='full'))   for par in parabolas])),1])  for pos_beat in valid_labs[0] ]
        cars_amplada_mean = array([mean(amplada1[max([i-100,0]):i+100]) for i in range(len(valid_labs[0]))])+0.5
        #amplada1 = abs(amplada1 / cars_amplada_mean -1) 
        ks_amplada1 = [valid_labs[0][i] for i,x in enumerate(valid_labs[0]) if amplada1[i] > cars_amplada_mean[i]]
        ks_amplada1 = [[x, 'V'] for x in ks_amplada1]
        return ks_amplada1
    
    ks_amplada1 = __amplada_per_lead__(data[0], valid_labs, parabolas)
    ks_amplada2 = __amplada_per_lead__(data[1], valid_labs, parabolas)
    ks_amplada3 = __amplada_per_lead__(data[2], valid_labs, parabolas)
    
    

    vs_detected_amplada0,total_detected_amplada0 =  getResults(vs, ks_amplada1)
    vs_detected_amplada1,total_detected_amplada1 =  getResults(vs, ks_amplada2)
    vs_detected_amplada2,total_detected_amplada2 =  getResults(vs, ks_amplada3)    
    print case + "  Anchura  C1  2     " + str(vs_detected_amplada0) + " of " + str(len(vs)) + "       saw   " + str(total_detected_amplada0)
    print case + "  Anchura  C2  2     " + str(vs_detected_amplada1) + " of " + str(len(vs)) + "       saw   " + str(total_detected_amplada1)
    print case + "  Anchura  C3  2     " + str(vs_detected_amplada2) + " of " + str(len(vs)) + "       saw   " + str(total_detected_amplada2)
    
    return [ks_amplada1,ks_amplada2, ks_amplada3]


def getAmpladaCriteriaProfe(data,valid_labs):
    from numpy import arange,sign, correlate
        
    def __amplada_per_beat__(beat, thrs = 0.3):
        from numpy import argmax, abs,max,sign
        pos = argmax(abs(beat))
        beat = sign(beat[pos])*beat
        val = max(beat) * thrs
        left = pos
        right = pos
        while left > 0:
            if beat[left] > val and beat[left-1] <= val: 
                left = left - abs(float(val-beat[left-1])/float(beat[left]-beat[left-1]))
                break
            left = left-1
        while right < len(beat)-1:
            if beat[right] > val and beat[right+1] <= val: 
                right = right + abs(float(val-beat[right+1])/float(beat[right+1]-beat[right]))
                break
            right = right+1
        
        return right-left    
        
    def __amplada_per_lead__profe__(sig, valid_labs):        
        amplada1 = [__amplada_per_beat__(sig[x-5:x+10],0.3) for x in valid_labs[0]]
        cars_amplada_mean = array([mean(amplada1[i-100:i+100]) for i in range(len(valid_labs[0]))])+0.98
        ks_amplada1 = [valid_labs[0][i] for i,x in enumerate(valid_labs[0]) if amplada1[i] > cars_amplada_mean[i]]
        ks_amplada1 = [[x, 'V'] for x in ks_amplada1]
        return ks_amplada1
    
    ks_amplada1 = __amplada_per_lead__profe__(data[0], valid_labs)
    ks_amplada2 = __amplada_per_lead__profe__(data[1], valid_labs)
    ks_amplada3 = __amplada_per_lead__profe__(data[2], valid_labs)
    
    

    vs_detected_amplada0,total_detected_amplada0 =  getResults(vs, ks_amplada1)
    vs_detected_amplada1,total_detected_amplada1 =  getResults(vs, ks_amplada2)
    vs_detected_amplada2,total_detected_amplada2 =  getResults(vs, ks_amplada3)    
    print case + "  Anchura  C1 Profe  " + str(vs_detected_amplada0) + " of " + str(len(vs)) + "       saw   " + str(total_detected_amplada0)
    print case + "  Anchura  C2 Profe  " + str(vs_detected_amplada1) + " of " + str(len(vs)) + "       saw   " + str(total_detected_amplada1)
    print case + "  Anchura  C3 Profe  " + str(vs_detected_amplada2) + " of " + str(len(vs)) + "       saw   " + str(total_detected_amplada2)
    
    return [ks_amplada1,ks_amplada2, ks_amplada3]



def getMITCriteria(data,valid_labs):
    from beatWork.characterization import estocastic_characterizers
    from numpy import std
    def __prepare_MIT__(sig,valid_labs):
        ks_MIT = estocastic_characterizers.caracterize_beats_MIT(sig, valid_labs[0])
        
        cars_amplada_mean = array([mean(ks_MIT[max([i-100,0]):i+100]) for i in range(len(valid_labs[0]))])
        cars_amplada_mean = cars_amplada_mean *1.3+ std(cars_amplada_mean)*2
        ks_MIT = [valid_labs[0][i] for i,x in enumerate(valid_labs[0]) if ks_MIT[i] > cars_amplada_mean[i]]    
        ks_MIT = [[x,'V'] for x in ks_MIT]
        return ks_MIT
    
    ks_MIT1 = __prepare_MIT__(data[0],valid_labs)
    ks_MIT2 = __prepare_MIT__(data[1],valid_labs)
    ks_MIT3 = __prepare_MIT__(data[2],valid_labs)
    
    vs_detected_MIT0,total_detected_MIT0 =  getResults(vs, ks_MIT1)
    vs_detected_MIT1,total_detected_MIT1 =  getResults(vs, ks_MIT2)
    vs_detected_MIT2,total_detected_MIT2 =  getResults(vs, ks_MIT3)    
    print case + "  MIT C1             " + str(vs_detected_MIT0) + " of " + str(len(vs)) + "       saw   " + str(total_detected_MIT0)
    print case + "  MIT C2             " + str(vs_detected_MIT1) + " of " + str(len(vs)) + "       saw   " + str(total_detected_MIT1)
    print case + "  MIT C3             " + str(vs_detected_MIT2) + " of " + str(len(vs)) + "       saw   " + str(total_detected_MIT2)

    return [ks_MIT1,ks_MIT2, ks_MIT3]




def logicalCombination1(ks,ks_amplada1,ks_amplitud0,ks_morphology):
    ks_conjunt = []
    aux3 = [x[0] for x in ks]
    
    
    aux4 = [x[0] for x in ks_amplada1]
    aux1 = [x[0] for x in ks_amplitud0]
    aux2 = [x[0] for x in ks_morphology]
    
    
    ks_conjunts1 = [x for x in valid_labs[0] if (x in aux2) and ((x in aux4 and x in aux3) or (x in aux4 and x in aux1) or (x in aux3 and x in aux1))]
    ks_conjunt = ks_conjunt + ks_conjunts1
    ks_conjunts1 = [[x, 'V'] for x in ks_conjunts1]
    return ks_conjunts1



import qrsDetector.supervision.detection.qualityEstimation.normalizedRRestimator as estimator

#107-00584
cases = h_io.get_usable_cases(REDIAGNOSE_DIAGNOSER)
a = []
b = []

Xs_C1 = []
Xs_C2 = []
Xs_C3 = []
Xs_Combined = []

Ys = []
indexs = [0]

for num_case, case in enumerate(cases[:1]):
    #try:
    #case = '107-00679'
    #case = '107-00599'
    #case  = '107-00581'
    #  en perdo 1000 i pico !!!!  107-00729
    #107-00729
    case = "107-00729"

    print str(num_case) + "  of  " + str(len(cases))
    data,labs = h_io.get_complete_exam(case, REDIAGNOSE_DIAGNOSER)
    
    import signalCleaning.cleaners as cl2
    data[0] = cl2.hp(data[0],1.)
    data[0] = cl2.bandpass(data[0],1.,40.)
    data[1] = cl2.hp(data[1],1.)
    data[1] = cl2.bandpass(data[1],1.,40.)
    data[2] = cl2.hp(data[2],1.)
    data[2] = cl2.bandpass(data[2],1.,40.)

    valid_labs = sorted(zip([i for x,i in enumerate(labs[0]) if labs[1][x] in 'SVN'],[labs[1][x] for x,i in enumerate(labs[0]) if labs[1][x] in 'SVN']),key=lambda x:x[0])
    valid_labs = [[ss[0] for ss in valid_labs],[ss[1] for ss in valid_labs]]
    vs = [x for i,x in enumerate(valid_labs[0]) if valid_labs[1][i] in 'V']
    tr = {'V':1,'N':0,'S':0}    
    Z = array(map(lambda x:tr[x],valid_labs[1]))
    
    



    
    
    ks_rr = getRRCriteria(data, valid_labs)
    ks_MIT = getMITCriteria(data,valid_labs)

    #ks_amplada_profe = getAmpladaCriteriaProfe(data,valid_labs)
    ks_amplada = getAmpladaCriteria(data, valid_labs)  
    #ks_amplitud2 = getAmplitudeCriteria2(data, valid_labs)
    ks_amplada2 = getAmpladaCriteria2(data,valid_labs)
    ks_amplitud3 = getAmplitudeCriteria3(data,valid_labs)
    ks_amplitud = getAmplitudeCriteria(data, valid_labs)
    ks_morphology = getMorphologyCriteria(data, valid_labs)
    #ks_morhampl = getMorphologyCriteriaAndAmplitude(data,valid_labs)
    
    
    '''
    v_rr = getResultAsVector(ks_rr[0], valid_labs)
    v_mit1 = getResultAsVector(ks_MIT[0], valid_labs)
    v_mit2 = getResultAsVector(ks_MIT[1], valid_labs)
    v_mit3 = getResultAsVector(ks_MIT[2], valid_labs)
    v_mit = v_mit1 + v_mit2 + v_mit3

    v_amplada_morph1 = getResultAsVector(ks_morhampl[0],valid_labs)
    v_amplada_morph2 = getResultAsVector(ks_morhampl[1],valid_labs)
    v_amplada_morph3 = getResultAsVector(ks_morhampl[2],valid_labs)
    v_amplada_morph = v_amplada_morph1 + v_amplada_morph2 + v_amplada_morph3
    
    v_amplada1 = getResultAsVector(ks_amplada[0], valid_labs)
    v_amplada2 = getResultAsVector(ks_amplada[1], valid_labs)
    v_amplada3 = getResultAsVector(ks_amplada[2], valid_labs)
    v_amplada = v_amplada1 + v_amplada2 + v_amplada3
    
    v_amplitud1 = getResultAsVector(ks_amplitud[0], valid_labs)
    v_amplitud2 = getResultAsVector(ks_amplitud[1], valid_labs)
    v_amplitud3 = getResultAsVector(ks_amplitud[2], valid_labs)
    v_amplitud = v_amplitud1 + v_amplitud2 + v_amplitud3
    

    
    v_morphology1 = getResultAsVector(ks_morphology[0], valid_labs)
    v_morphology2 = getResultAsVector(ks_morphology[1], valid_labs)
    v_morphology3 = getResultAsVector(ks_morphology[2], valid_labs)
    v_morphology = v_morphology1 + v_morphology2 + v_morphology3
    
    Xs_C1 = Xs_C1 + list(array([v_rr, v_mit1,v_amplada1,v_amplada_morph1,v_amplitud1,v_morphology1]).transpose())
    Xs_C2 = Xs_C2 + list(array([v_rr, v_mit2,v_amplada2,v_amplada_morph2,v_amplitud2,v_morphology2]).transpose())
    Xs_C3 = Xs_C3 + list(array([v_rr, v_mit3,v_amplada3,v_amplada_morph3,v_amplitud3,v_morphology3]).transpose())
    Xs_Combined = Xs_Combined + list(array([v_rr, v_mit,v_amplada,v_amplada_morph,v_amplitud,v_morphology]).transpose())    
    #Ys = Ys + map(lambda x: {'V':1,'S':0,'N':0}[x],valid_labs[1])
    #indexs.append(len(Ys))
    #Z = clf.predict(X.transpose())
    #fallos1 = sum((Z - array(Y))== -1)
    #fallos2 = sum((Z - array(Y))== 1)
    #Xs.append(fallos1)
    #Ys.append(fallos2)
    #print str(fallos1) + "        " + str(fallos2)
    
    
    
    
       
    ks_conjunts1 = logicalCombination1(ks_rr[0],ks_amplada[0],ks_amplitud[0],ks_morphology[0])
    ks_conjunts2 = logicalCombination1(ks_rr[1],ks_amplada[1],ks_amplitud[1],ks_morphology[1])
    ks_conjunts3 = logicalCombination1(ks_rr[2],ks_amplada[2],ks_amplitud[2],ks_morphology[2])
    ks_conjunt = [x[0] for x in ks_conjunts1]+[x[0] for x in ks_conjunts2]+[x[0] for x in ks_conjunts3]
    ks_conjunt = sorted(list(set(ks_conjunt)))
    ks_conjunt = [[x,'V'] for x in ks_conjunt]
    
        
    vs_detected_conjunt1, total_vs_detected_conjunt1 = getResults(vs,ks_conjunts1)
    vs_detected_conjunt2, total_vs_detected_conjunt2 = getResults(vs,ks_conjunts2)
    vs_detected_conjunt3, total_vs_detected_conjunt3 = getResults(vs,ks_conjunts3)
    vs_detected_conjunt, total_vs_detected_conjunt = getResults(vs,ks_conjunt)
    
    print case + "  Conjunt   C1       " + str(vs_detected_conjunt1) + " of " + str(len(vs)) + "       saw   " + str(total_vs_detected_conjunt1)
    print case + "  Conjunt   C2       " + str(vs_detected_conjunt2) + " of " + str(len(vs)) + "       saw   " + str(total_vs_detected_conjunt2)
    print case + "  Conjunt   C3       " + str(vs_detected_conjunt3) + " of " + str(len(vs)) + "       saw   " + str(total_vs_detected_conjunt3)
    print case + "  Conjunt global     " + str(vs_detected_conjunt) + " of " + str(len(vs)) + "       saw   " + str(total_vs_detected_conjunt)    


    res = [vs_detected_conjunt,len(vs),total_vs_detected_conjunt,len(valid_labs[0])]
    a.append(res)
    
    if len(vs) > 0:
        tp = res[0]
        fn = res[1]-res[0]
        fp = res[2]-res[0]
        tn = res[-1]-res[1]-fp
        
        sens = float(tp)/(tp+fn)*100
        spec = float(tn)/(tn+fp)*100
        ppv  = float(tp)/(tp+fp)*100
        npv  = float(tn)/(fn+tn)*100
        b.append([sens,spec,ppv,npv])
    else:
        b.append([-1,-1,-1,-1])
    print str(b[-1])
    print "-------------------------------------------"
    '''
    #except:
    #    print "fallo  " + case
    #    pass
    




















'''    
def resRaro(j,indexs,XX,YY):
    from beatWork.classification import basic_classifiers as cl
    clf = cl.clf_SVM(array(list(XX[:indexs[j]])+list(XX[indexs[j+1]:])),array(list(YY[:indexs[j]])+list(YY[indexs[j+1]:])))
    Z = clf.predict(XX[indexs[j]:indexs[j+1]])
    return [sum((Z - array(YY[indexs[j]:indexs[j+1]]))== 1),sum((Z - array(YY[indexs[j]:indexs[j+1]]))== -1),sum(array(YY[indexs[j]:indexs[j+1]]))]

import cPickle
f = open('testingSVMRaro','rb')
Xs_C1 = cPickle.load(f)
Xs_C2 = cPickle.load(f)
Xs_C3 = cPickle.load(f)
Xs_Combined = cPickle.load(f)
Ys = cPickle.load(f)
f.close()







candidats = sum(array(Xs_C2),1)
X_train = [Xs_C2[i] for i,x in enumerate(candidats) if sum(x) > 0]
Y_train = [Ys[i] for i,x in enumerate(candidats) if sum(x) > 0]
clf = cl.clf_SVM(array(X_train),array(Y_train))
'''