
from numpy import array,sqrt,dot,argmax,max,argmin,min,mean,median
from system.settings import *
import signalCleaning.sample as cl
from mdp.nodes import JADENode 
from system.settings import *
from numpy import random

cases = h_io.get_usable_cases(REDIAGNOSE_DIAGNOSER)
from time import time
import beatWork.characterization.waveform_characterizers as chr

###Caracteritzadors Normalitzats!!! PFC






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
    #try:
    #case = '107-00679'
    #case = '107-00599'
    #case  = '107-00581'
    #  en perdo 1000 i pico !!!!  107-00729
    #107-00729
    #case = "107-00729"
    #case = '107-00728'
    #case = '107-00599'
    
    
    print str(num_case) + "  of  " + str(len(cases))
    data,labs = h_io.get_complete_exam(case, REDIAGNOSE_DIAGNOSER)
    
    import signalCleaning.cleaners as cl2
    reload(cl2)
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
    
    
    data[0] = cl2.normalizer(data[0], valid_labs[0], valid_labs[1])
    data[1] = cl2.normalizer(data[1], valid_labs[0], valid_labs[1])
    data[2] = cl2.normalizer(data[2], valid_labs[0], valid_labs[1])
    
    import extractBeats.extractOneSignalBeats.extractorsNN as ext
    import beatWork.characterization.waveform_characterizers as chr
    reload(chr)
    reload(ext)
    charact = chr.sampleCharacterizator_JustCenter
    
    import cPickle
    end_wave_1, pos1,labs_1 = ext.__generic__extractor__(data[0],valid_labs[0],charact,valid_labs[1])
    f = open('clasificacionPFC/'+case+'_lead1_4_waveform_norm','wb')
    cPickle.dump(array(end_wave_1),f)
    cPickle.dump(labs_1,f)
    cPickle.dump(pos1,f)
    del end_wave_1
    f.close()

    end_wave_2, pos2,labs_2 = ext.__generic__extractor__(data[1],valid_labs[0],charact,valid_labs[1])
    f = open('clasificacionPFC/'+case+'_lead2_4_waveform_norm','wb')
    cPickle.dump(array(end_wave_2),f)
    cPickle.dump(labs_2,f)
    cPickle.dump(pos2,f)        
    del end_wave_2
    f.close()

    end_wave_3, pos3,labs_3 = ext.__generic__extractor__(data[2],valid_labs[0],charact,valid_labs[1])        
    f = open('clasificacionPFC/'+case+'_lead3_4_waveform_norm','wb')
    cPickle.dump(array(end_wave_3),f)
    cPickle.dump(labs_3,f)
    cPickle.dump(pos3,f)        
    del end_wave_3
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
    #except:
    #    print "fallo  " + case
    #    pass
    












