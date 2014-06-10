

def analyzeZonaChunga(clean_sample, begin, end, pos_beats_candidates, stableBeatsGroupsChar, RR_reference, deep_iteration, f_tall, **kwargs):
    print "calling recursivitat"
    import analyzer.sample_analyzer as an
    reload(an)    
    zonitas = an.analyze(clean_sample[begin:end], f_tall, deep_iteration+1, [x-begin for x in pos_beats_candidates if x >=begin and x <= end], stableBeatsGroupsChar, RR_reference, **kwargs)
    try:
        return [(x[0]+begin,x[1]+begin,[y+begin for y in x[2]], [x[3][0],x[3][1]], x[4],x[5]) for x in zonitas]
    except:
        pass

def ultimaPasada(signal, deep_iteration, previousCandidates, refference_beats, refference_RR):    
    import beatHunter.sample_beat_hunter as sampleBeatHunter
    reload(sampleBeatHunter)
    candidats =  sampleBeatHunter.sampleBeatHunter(signal, 0, len(signal), refference_beats, refference_RR, 0.8)
    return candidats 

def ultimaPasada2(signal, deep_iteration, previousCandidates, refference_beats, refference_RR):    
    #import beatHunter.sample_beat_hunter as sampleBeatHunter
    #reload(sampleBeatHunter)
    #candidats =  sampleBeatHunter.sampleBeatHunter(signal, 0, len(signal), refference_beats, refference_RR, 0.8)
    
    import analyzer.zoneAnalyzer.goodZoneAnalyzer.sampleGoodZoneAnalyzer as gzA
    reload(gzA)
    import ipdb
    ipdb.set_trace()
    
    positions, types, quality, quant_group, group_char = gzA.analyzeCorrectZone(signal, 0,len(signal), previousCandidates, [], list(refference_beats),0.8+0.5*deep_iteration, use_own_beats = True)
    return positions
