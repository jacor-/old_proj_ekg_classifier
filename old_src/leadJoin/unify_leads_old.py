from system.settings import *
import signalCleaning.cleaners as cl2
import zoneQualityStimation.penalizedZoneQualityStimation as qe2
import error.stats as st

cases = h_io.get_usable_cases(REDIAGNOSE_DIAGNOSER)
res = {}
for cosa in cases:
    #try:
    case,labs = h_io.get_complete_exam(cosa,REDIAGNOSE_DIAGNOSER)
    
    
    sig = cl2.hp(case[0],1.)
    d = dets.detectTomkinsAdaptive(sig)
    kwargs={'beatsPosition':d}
    ret0 = qe2.sampleZoneQualityEstimation(sig,0,len(sig),**kwargs)
    del sig
    
    sig = cl2.hp(case[1],1.)
    d = dets.detectTomkinsAdaptive(sig)
    kwargs={'beatsPosition':d}
    ret1 = qe2.sampleZoneQualityEstimation(sig,0,len(sig),**kwargs)
    del sig
        
    sig = cl2.hp(case[2],1.)
    d = dets.detectTomkinsAdaptive(sig)
    kwargs={'beatsPosition':d}
    ret2 = qe2.sampleZoneQualityEstimation(sig,0,len(sig),**kwargs)
    del sig
    
    i = 0
    j = 0
    k = 0
    
    tots = [-1]
    from numpy import argmin
    
    ret0 = [ret0[0],[0]+ret0[1]]
    ret1 = [ret1[0],[0]+ret1[1]]
    ret2 = [ret2[0],[0]+ret2[1]]
    while i < len(ret0[0]) and j < len(ret1[0]) and k < len(ret2[0]):
        #print str(len(ret0[0])-i) + "    " + str(len(ret1[0])-j) + "    " + str(len(ret2[0])-k) 
        
        aux = [ret0[1][i],ret1[1][j],ret2[1][k]]
        
        posmin = argmin(aux)
        
        tots.append([ret0[0][i],ret1[0][j],ret2[0][k]][posmin])
        if posmin == 0: i = i + 1
        if posmin == 1: j = j + 1
        if posmin == 2: k = k + 1
        
        if i < len(ret0[0]):
            while ret0[0][i]-tots[-1] < 30 and i < len(ret0[0])-1:
                i = i + 1
        if j < len(ret1[0]):
            while ret1[0][j]-tots[-1] < 30 and j < len(ret1[0])-1:
                j = j + 1
        if k < len(ret2[0]):
            while ret2[0][k]-tots[-1] < 30 and k < len(ret2[0])-1:
                k = k + 1
    
    
    tots = tots[1:]
    
    valid_labs = [labs[0][i] for i,x in enumerate(labs[0]) if labs[1][i] in "SVN"]
    
    
    res[cosa] = st.compareLists(valid_labs,tots)
    print str(res[cosa])
    
    del ret0
    del ret1
    del ret2
    del case
    del labs

