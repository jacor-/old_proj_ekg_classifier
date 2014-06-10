
from numpy import array,sqrt,dot,argmax,max,argmin,min,mean,median
from system.settings import *
import signalCleaning.sample as cl
from mdp.nodes import JADENode 
from system.settings import *

cases = h_io.get_usable_cases(REDIAGNOSE_DIAGNOSER)
from time import time
import beatWork.characterization.waveform_characterizers as chr

import qrsDetector.supervision.detection.qualityEstimation.normalizedRRestimator as estimator






###Caracteritzadors Normalitzats!!! PFC

def getCharacterization2(sig,valid_labs):
    import signalCleaning.cleaners as cl2
    from numpy import array
    
    Z1_morph = array(__chr_morph_criteria_2__(sig,valid_labs))
    Z1_ampl = array(__chr_amplitude_one_lead__(sig,valid_labs))
    Z1_weig = array(__chr_amplada_per_lead_2__(sig,valid_labs))
    Z1_mit = __chr_mit_per_lead_3__(sig,valid_labs)
    rr1 = __chr__getRRCriteria_2__(sig,valid_labs)
    #morph3 = __chr_morph_mahalanobis_2__(sig,valid_labs)
    
    from numpy.linalg import inv
    from scipy.spatial.distance import mahalanobis
    from numpy import cov,mean
    
    from numpy.linalg import inv
    from scipy.spatial.distance import mahalanobis, euclidean,cosine
    from numpy import cov,mean,array,shape

    #print "muntant array"
    aux = array([Z1_morph,Z1_ampl,Z1_weig,Z1_mit,rr1])
    

    #print "calculant cov"
    covar = inv(cov(aux))
    
    aux = aux.transpose()   
    me = mean(aux,0)
    
    #covar = inv(cov(array(subsels_norm).transpose()))
    #print "calculant distancies"
    #mahalanobis_tots = [mahalanobis(x,me,covar) for x in aux]
    
    return array([Z1_morph,Z1_ampl,Z1_weig,Z1_mit,rr1]).transpose()


def getCharacterization(sig,valid_labs):
    import signalCleaning.cleaners as cl2
    from numpy import array
    
    Z1_morph = array(__chr_morph_criteria_2__(sig,valid_labs))
    Z1_ampl = array(__chr_amplitude_one_lead_2__(sig,valid_labs))
    Z1_weig = array(__chr_amplada_per_lead_2__(sig,valid_labs))
    Z1_mit = __chr_mit_per_lead_3__(sig,valid_labs)
    rr1 = __chr__getRRCriteria_2__(sig,valid_labs)
    #morph3 = __chr_morph_mahalanobis_2__(sig,valid_labs)
    
    from numpy.linalg import inv
    from scipy.spatial.distance import mahalanobis
    from numpy import cov,mean
    
    from numpy.linalg import inv
    from scipy.spatial.distance import mahalanobis, euclidean,cosine
    from numpy import cov,mean,array,shape

    #print "muntant array"
    aux = array([Z1_morph,Z1_ampl,Z1_weig,Z1_mit,rr1])
    

    #print "calculant cov"
    covar = inv(cov(aux))
    
    aux = aux.transpose()   
    me = mean(aux,0)
    
    #covar = inv(cov(array(subsels_norm).transpose()))
    #print "calculant distancies"
    #mahalanobis_tots = [mahalanobis(x,me,covar) for x in aux]
    
    return array([Z1_morph,Z1_ampl,Z1_weig,Z1_mit,rr1]).transpose()


def __chr_morph_mahalanobis_2__(sig,valid_labs):
    waverefs = [x-5+argmax(abs(sig[x-5:x+5])) for x in valid_labs[0]]
    waverefs = [sig[x-20:x+30] for x in waverefs]
    #waverefs = [x-mean(x) for x in waverefs]
    #waverefs = [x/sqrt(dot(x,x)) for x in waverefs]
    
    from numpy.linalg import inv
    from scipy.spatial.distance import mahalanobis
    from numpy import cov,mean
    
    from numpy.linalg import inv
    from scipy.spatial.distance import mahalanobis, euclidean,cosine
    from numpy import cov,mean,array
    
    me = mean(waverefs,0)

    covar = inv(cov(array(waverefs).transpose()))
    #covar = inv(cov(array(subsels_norm).transpose()))
    likelihood_pos = [mahalanobis(x,me,covar) for x in waverefs]
    

    likelihood_vals = array(likelihood_pos)
    cars_thrs_mean = (array([median(likelihood_vals[max([i-25,0]):min([i+25,len(likelihood_vals)])]) for i in range(len(likelihood_vals))]))
        #cars_thrs_mean = max(array([cars_thrs_mean+std(cars_thrs_mean)*2,ones(len(cars_thrs_mean))-0.75]),0)
    

    
    
    #cars_thrs_mean_norm = abs(cl2.hp(array(likelihood_vals),3)-1)
    cars_thrs_mean_norm = (likelihood_vals-cars_thrs_mean)/cars_thrs_mean
    #cars_thrs_mean_norm = (likelihood_vals-cars_thrs_mean)
    return cars_thrs_mean_norm


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
    cars = (cars - cars_thrs_mean)/cars_thrs_mean
    #cars = (cars - cars_thrs_mean)
    return cars

def __chr_amplitude_one_lead_2__(sig,valid_labs):
    import pywt
    from numpy import max,min,median
    maxs = array([max(sig[max([pos-10,0]):pos+10]) for pos in valid_labs[0]])
    mins = array([min(sig[max([pos-10,0]):pos+10]) for pos in valid_labs[0]])
    cars = maxs-mins
    cars_thrs_mean = array([median(cars[max([i-100,0]):min([i+100,len(cars)])]) for i in range(len(cars))])
    return (cars - cars_thrs_mean)/cars_thrs_mean

def __chr_morph_criteria_3__(sig,valid_labs):
    
    #waverefs = [x-mean(x) for x in waverefs]
    #waverefs = [x/sqrt(dot(x,x)) for x in waverefs]
    from numpy import correlate
    likelihood_pos = [mean(   [max(correlate(sig[valid_labs[0][i]-20:valid_labs[0][i]+30],sig[valid_labs[0][i-5+j]-30:valid_labs[0][i-5+j]+40],mode='valid'   )) for j in range(20) if i-5+j>= 0 and j!=5j and i-5+j < len(valid_labs[0])]  ) for i in range(len(valid_labs[0]))]
    likelihood_pos = [likelihood_pos[i]/median(likelihood_pos[max([0,i-10]):i+10]) for i in range(len(valid_labs[0]))]
    return array(likelihood_pos)-1
    #plot dot(carro),, sale una cosa chula de cojones!!!
    likelihood_vals = array(likelihood_pos)
    cars_thrs_mean = (array([median(likelihood_vals[max([i-25,0]):min([i+25,len(likelihood_vals)])]) for i in range(len(likelihood_vals))]))
        #cars_thrs_mean = max(array([cars_thrs_mean+std(cars_thrs_mean)*2,ones(len(cars_thrs_mean))-0.75]),0)
    
    
    #cars_thrs_mean_norm = abs(cl2.hp(array(likelihood_vals),3)-1)
    #cars_thrs_mean_norm = (likelihood_vals-cars_thrs_mean)/cars_thrs_mean
    cars_thrs_mean_norm = (likelihood_vals-cars_thrs_mean)
    return cars_thrs_mean_norm


def __chr_morph_criteria_2__(sig,valid_labs):
    waverefs = [x-5+argmax(abs(sig[x-5:x+5])) for x in valid_labs[0]]
    waverefs = [sig[x-20:x+30] for x in waverefs]
    #waverefs = [x-mean(x) for x in waverefs]
    #waverefs = [x/sqrt(dot(x,x)) for x in waverefs]
    
    likelihood_pos = [dot(waverefs[i],waverefs[i])/median(dot(waverefs[i],array(waverefs[max([i-40,0]):i+40]).transpose())) for i in range(len(valid_labs[0]))]
    return array(likelihood_pos)-1
    #plot dot(carro),, sale una cosa chula de cojones!!!
    likelihood_vals = array(likelihood_pos)
    cars_thrs_mean = (array([median(likelihood_vals[max([i-25,0]):min([i+25,len(likelihood_vals)])]) for i in range(len(likelihood_vals))]))
        #cars_thrs_mean = max(array([cars_thrs_mean+std(cars_thrs_mean)*2,ones(len(cars_thrs_mean))-0.75]),0)
    
    
    #cars_thrs_mean_norm = abs(cl2.hp(array(likelihood_vals),3)-1)
    #cars_thrs_mean_norm = (likelihood_vals-cars_thrs_mean)/cars_thrs_mean
    cars_thrs_mean_norm = (likelihood_vals-cars_thrs_mean)
    return cars_thrs_mean_norm


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
    #cars_thrs_mean_norm = (likelihood_vals-cars_thrs_mean)/cars_thrs_mean
    cars_thrs_mean_norm = (likelihood_vals-cars_thrs_mean)
    return cars_thrs_mean_norm

'''
def __chr_amplada_per_lead__(sig, valid_labs):
    from numpy import correlate,sign,median,array,arange

    def parabola(c):
        paux = array([-1.*(float(x*x)/c/c-1) for x in arange(-c,c+1,1)]).astype('float')
        paux = paux/sqrt(dot(paux,paux))
        return paux

    
    parabolas = [parabola(i+1) for i in range(15)]    
    amplada1 = array([argmax(array([sign(max(sig[pos_beat-5:pos_beat+5]))*max(correlate(sig[pos_beat-10:pos_beat+10],par,mode='full'))   for par in parabolas]))  for pos_beat in valid_labs[0] ])
    cars_amplada_mean = array([median(amplada1[max([i-100,0]):i+100]) for i in range(len(valid_labs[0]))])+1
    cars_thrs_mean_norm =  amplada1/cars_amplada_mean
    return cars_thrs_mean_norm
'''

def __chr_amplada_per_lead_2__(sig, valid_labs):
    from numpy import correlate,sign,median,array,arange
    
    def parabola(c):
        paux = array([-1.*(float(x*x)/c/c-1) for x in arange(-c,c+1,1)]).astype('float')
        paux = paux/sqrt(dot(paux,paux))
        return paux
    
    
    from numpy import arange
    from scipy.interpolate import interp1d
    parabolas = [parabola(i+1) for i in arange(4,60)]
    
    amplada1 = []
    for pos_beat in valid_labs[0]:
        beat = array(sig[pos_beat-10:pos_beat+25])
        #beat = beat * (beat> 0)
        interpoler = interp1d(arange(len(beat)),beat,kind='quadratic')
        beat = interpoler(arange(0,len(beat)-1,0.1))
        cosa = beat*sign(sig[argmax(abs(sig[pos_beat-5:pos_beat+5]))+pos_beat-5])
        amplada1.append(argmax(array([max(correlate(cosa,par,mode='full'))   for par in parabolas])))
    
    amplada1 = array(amplada1)
    #return amplada1
    cars_amplada_mean = array([median(amplada1[max([i-30,0]):i+30]) for i in range(len(valid_labs[0]))])+1
    #cars_thrs_mean_norm =  (amplada1-cars_amplada_mean) / cars_amplada_mean
    cars_thrs_mean_norm =  amplada1
    return cars_thrs_mean_norm


def __chr_amplada_per_lead_3__(sig, valid_labs):
    from numpy import correlate,sign,median,array,arange
    
    def parabola(c):
        paux = array([-1.*(float(x*x)/c/c-1) for x in arange(-c,c+1,1)]).astype('float')
        paux = paux/sqrt(dot(paux,paux))
        return paux
    
    
    from numpy import arange,mod
    from scipy.interpolate import interp1d
    parabolas = [parabola(i+1) for i in arange(4,60)]
    
    amplada1 = []
    for xxx,pos_beat in enumerate(valid_labs[0]):
        beat = array(sig[pos_beat-10:pos_beat+25])
        #beat = beat * (beat> 0)
        interpoler = interp1d(arange(len(beat)),beat,kind='quadratic')
        beat = interpoler(arange(0,len(beat)-1,0.1))
        #cosa = beat*sign(sig[argmax(abs(sig[pos_beat-5:pos_beat+5]))+pos_beat-5])
        cosa = beat
        valmax = max(cosa)
        pos_max = argmax(cosa)
        cosa = cosa - valmax * 0.2
        i_left = 0
        i_right = 0
        while i_left < 50 and pos_max+i_left > 0:
            if cosa[pos_max-i_left]*cosa[pos_max-i_left-1] <= 0:
                break
            i_left = i_left + 1
        while i_right < 50 and pos_max+i_right < len(cosa)-1:
            if cosa[pos_max+i_right]*cosa[pos_max+i_right+1] <= 0:
                break
            i_right = i_right + 1
            
            
        
        #cosa = beat*-1
        valmax = min(cosa)
        pos_max = argmin(cosa)
        cosa = cosa + valmax * 0.2
        i_left2 = 0
        i_right2 = 0
        while i_left2 < 50 and pos_max+i_left2 > 0:
            if cosa[pos_max-i_left2]*cosa[pos_max-i_left2-1] <= 0:
                break
            i_left2 = i_left2 + 1
        while i_right2 < 50 and pos_max+i_right2 < len(cosa)-1:
            if cosa[pos_max+i_right2]*cosa[pos_max+i_right2+1] <= 0:
                break
            i_right2 = i_right2 + 1
        
            
        amplada1.append(max([i_right+i_left,i_right2+i_left2]))
        
        
    amplada1 = array(amplada1)
    #return amplada1
    cars_amplada_mean = array([median(amplada1[max([i-30,0]):i+30]) for i in range(len(valid_labs[0]))])+1
    #cars_thrs_mean_norm =  (amplada1-cars_amplada_mean)/cars_amplada_mean
    cars_thrs_mean_norm =  amplada1
    return cars_thrs_mean_norm



def __chr__getRRCriteria(sig,valid_labs):
    rr = estimator.estimate(valid_labs[0],0,len(sig),ENTORN=10)
    #rr2 = [0]+list(array([float(valid_labs[0][i+2]-valid_labs[0][i+1])/(valid_labs[0][i+1]-valid_labs[0][i]) for i in range(len(valid_labs[0])-2)])-1)+[0]
    
    rr_low = rr > 0.15
    rr_hight = rr < -0.15
    jmps = rr_low.astype('int')-rr_hight.astype('int')
    
    return jmps

def __chr__getRRCriteria_2__(sig,valid_labs):
    rr = estimator.estimate(valid_labs[0],0,len(sig),ENTORN=10)
    #rr2 = [0]+list(array([float(valid_labs[0][i+2]-valid_labs[0][i+1])/(valid_labs[0][i+1]-valid_labs[0][i]) for i in range(len(valid_labs[0])-2)])-1)+[0]

    return rr
















def __chr_mit_per_lead_3__(sig, valid_labs):
    from numpy import correlate,sign,median,array,arange
    from numpy import arange
    from scipy.interpolate import interp1d
    
    import beatWork.characterization.estocastic_characterizers as chrr
    reload(chrr)
    
    mit = []
    for pos_beat in valid_labs[0]:
        beat = array(sig[pos_beat-20:pos_beat+30])
        interpoler = interp1d(arange(len(beat)),beat,kind='cubic')
        beat = interpoler(arange(0,len(beat-1)-1,0.5))
        mit.append(chrr.caracterize_beats_MIT(beat,[40],fs=200.)[0])    
                
            
        #    mit.append(i_right+i_left)
        
    mit = array(mit)
    #return amplada1
    cars_amplada_mean = array([median(mit[max([i-30,0]):i+30]) for i in range(len(valid_labs[0]))])
    #cars_thrs_mean_norm =  (mit-cars_amplada_mean)/cars_amplada_mean
    cars_thrs_mean_norm =  (mit-cars_amplada_mean)/cars_amplada_mean
    return cars_thrs_mean_norm

