
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
    #cars = (cars - cars_thrs_mean)/cars_thrs_mean
    cars = (cars - cars_thrs_mean)
    return cars

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
        valmax = max(cosa)
        pos_max = argmax(cosa)
        cosa = cosa - valmax * 0.2
        i_left = 0
        i_right = 0
        while i_left < 50:
            if cosa[pos_max-i_left]*cosa[pos_max-i_left-1] <= 0:
                break
            i_left = i_left + 1
        while i_right < 50:
            if cosa[pos_max+i_right]*cosa[pos_max+i_right+1] <= 0:
                break
            i_right = i_right + 1
        amplada1.append(i_right+i_left)
    
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
    cars_thrs_mean_norm =  (mit-cars_amplada_mean)
    return cars_thrs_mean_norm


































'''
def getCandidatsMetodeAmpladaIMorph(sig,valid_labs):
    from numpy import convolve, min, max, correlate, zeros
    morph = __chr_morph_one_lead__(sig, valid_labs)
    amplitud = __chr_amplitude_one_lead__(sig, valid_labs)
    cosa = amplitud*morph > 0.2
    candidats_previs = [valid_labs[0][i] for i,x in enumerate(cosa) if x == True]
    a,b = fes_grups(data[0],candidats_previs,0.7)
    Zs_candidats = []
    for poss in b:
        #beats = mean([sig[x-10:x+20] for i,x in enumerate(valid_labs[0]) if cosa[i] == True],0)
        beats = mean([sig[x-10:x+20] for i,x in enumerate(poss)],0)
        aux = correlate(sig,beats,mode='same')
        
        ref_value = mean([max(aux[x-10:x+20]) for i,x in enumerate(valid_labs[0]) if x in poss],0)
        no_ref_value = mean([max(aux[x-10:x+20]) for i,x in enumerate(valid_labs[0]) if x not in poss],0)
        thrs = (ref_value+no_ref_value)/2
        
        candidats = [i for i,x in enumerate(valid_labs[0]) if thrs < max(aux[x-10:x+20])]
        
        Z2 = zeros(len(valid_labs[0]))
        for x in candidats:
            Z2[x] = 1.
        Zs_candidats.append(Z2)
    print str(len(a)) + "   " + str([len(x) for x in b])
    return Zs_candidats
'''







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

for num_case, case in enumerate(cases[:]):
    try:
        #case = '107-00679'
        #case = '107-00599'
        #case  = '107-00581'
        #  en perdo 1000 i pico !!!!  107-00729
        #107-00729
        #case = "107-00729"
        #case = '107-00728'
        #case = '107-00599'
        #if num_case != 0:
        #    continue
        
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
        
        
        Z1_morph = array(__chr_morph_criteria_2__(data[0],valid_labs))
        Z2_morph = array(__chr_morph_criteria_2__(data[1],valid_labs))
        Z3_morph = array(__chr_morph_criteria_2__(data[2],valid_labs))
        Z_morph_tot = mean([Z1_morph, Z2_morph, Z3_morph],0)
        
        Z1_ampl = array(__chr_amplitude_one_lead__(data[0],valid_labs))
        Z2_ampl = array(__chr_amplitude_one_lead__(data[1],valid_labs))
        Z3_ampl = array(__chr_amplitude_one_lead__(data[2],valid_labs))
        Z_ampl_tot = mean([Z1_ampl, Z2_ampl, Z3_ampl],0)
        
        Z1_weig = array(__chr_amplada_per_lead_2__(data[0],valid_labs))
        Z2_weig = array(__chr_amplada_per_lead_2__(data[1],valid_labs))
        Z3_weig = array(__chr_amplada_per_lead_2__(data[2],valid_labs))
        Z_weig_tot = mean([Z1_weig, Z2_weig, Z3_weig],0)
        
        #Z1_weig2 = __chr_amplada_per_lead_3__(data[0],valid_labs)
        #Z2_weig2 = __chr_amplada_per_lead_3__(data[1],valid_labs)
        #Z3_weig2 = __chr_amplada_per_lead_3__(data[2],valid_labs)
        #Z_weig2_tot = mean([Z1_weig2, Z2_weig2, Z3_weig2],0)
    
        Z1_mit = __chr_mit_per_lead_3__(data[0],valid_labs)
        Z2_mit = __chr_mit_per_lead_3__(data[1],valid_labs)
        Z3_mit = __chr_mit_per_lead_3__(data[2],valid_labs)
        Z_mit_tot = mean([Z1_mit, Z2_mit, Z3_mit],0)
    
        rr1 = __chr__getRRCriteria_2__(data[0],valid_labs)
        
    
        import cPickle

        lead1 = array([Z1_morph,Z1_ampl,Z1_weig,Z1_mit,rr1]).transpose()    
        f = open('clasificacionPFC/'+case+'_lead1_4_components_no_norm','wb')
        cPickle.dump(lead1,f)
        del lead1
        f.close()
        
        lead2 = array([Z2_morph,Z2_ampl,Z2_weig,Z2_mit,rr1]).transpose()    
        f = open('clasificacionPFC/'+case+'_lead2_4_components_no_norm','wb')
        cPickle.dump(lead2,f)
        del lead2
        f.close()
    
        lead3 = array([Z3_morph,Z3_ampl,Z3_weig,Z3_mit,rr1]).transpose()    
        f = open('clasificacionPFC/'+case+'_lead3_4_components_no_norm','wb')
        cPickle.dump(lead3,f)
        del lead3
        f.close()
        
        lead_tot = array([Z_morph_tot,Z_ampl_tot,Z_weig_tot,Z_mit_tot,rr1]).transpose()    
        f = open('clasificacionPFC/'+case+'_lead_all_4_components_no_norm','wb')
        cPickle.dump(lead_tot,f)
        del lead_tot
        f.close()
    
        f = open('clasificacionPFC/'+case+'_valid_labs_4_components_norm','wb')
        cPickle.dump(valid_labs,f)
        f.close()
        
        #from pylab import *
        #figure()
        #amplada_C1 = __chr_morph_criteria_2__(data[0], valid_labs)
        #plot(valid_labs[0],amplada_C1)
        #plot([x for i,x in enumerate(labs[0]) if labs[1][i] in 'V'], [-1 for i,x in enumerate(labs[0]) if labs[1][i] in 'V'],'yo')
    
        
        #print case + " C1   " + str(sum(Z - Z1 == -1)) + "  "  + str(sum(Z - Z1 == 1)) + "  "  + str(sum(Z - Z1 == 0))  + "  "  + str(sum(Z))
        #print case + " C2   " + str(sum(Z - Z2 == -1)) + "  "  + str(sum(Z - Z2 == 1)) + "  "  + str(sum(Z - Z2 == 0))  + "  "  + str(sum(Z))
        #print case + " C3   " + str(sum(Z - Z3 == -1)) + "  "  + str(sum(Z - Z3 == 1)) + "  "  + str(sum(Z - Z3 == 0))  + "  "  + str(sum(Z))
        #print case + " Cmaj " + str(sum(Z - Z_tot == -1)) + "  "  + str(sum(Z - Z_tot == 1)) + "  "  + str(sum(Z - Z_tot == 0))  + "  "  + str(sum(Z))
    
        
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
    except:
        print "fallo  " + case
        pass
    












