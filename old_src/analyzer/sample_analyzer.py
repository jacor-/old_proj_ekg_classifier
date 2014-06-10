
def analyze(signal, f_tall, deep_iteration, previousCandidates, refference_beats, refference_RR, **kwargs):
    '''
    Is defined by a kwargs argument:
    - signal
    - f_tall
    - deep_iteration
    - previousCandidates
    - refference_beats
    - refference_RR
    - **kwargs:
         {'cleaner':          {'fun_name':NOM, 'params': },
          'detector':         {'fun_name':NOM, 'params': },
          'characterizator':  {'fun_name':NOM, 'params': },
          'identification':   {'fun_name':NOM, 'params': },
          'qualityStimation': {'fun_name':NOM, 'params': },
          'zoneDecisor':      {'fun_name':NOM, 'params': },
          'goodZoneAnalyzer': {'fun_name':NOM, 'params': },
          'badZoneAnalyzer' : {'fun_name':NOM, 'params': }}
    '''
    
    '''
    Signal: is the signal that has to be analyzed
    Zones:  pairs of positions ("begin","end") which defines intervals in the signal that is going to be analyzed
    f_tall: frequency for the filter
    deep_iteration: deep of the signal analyzed 
    Refference_beats: beats in the same signal that has already been detected
    Refference_RR: information about the RR in zones which has been already analyzed
    '''
    
    
    
    print "profundidad " + str(deep_iteration)
    if deep_iteration > 2:
        import zoneAnalyzer.goodZoneAnalyzer.sampleGoodZoneAnalyzer as gzA
        import zoneAnalyzer.badZoneAnalyzer.sampleBadZoneAnalyzer as bzA
        reload(bzA)
        reload(gzA)
        return [(0, len(signal), bzA.ultimaPasada2(signal, deep_iteration, previousCandidates, refference_beats, refference_RR),[[],[]], 0.0 , deep_iteration)]
    
    '''
    STEP 1:
    First of all we have to clean the signal.
    '''    
    import signalCleaning.sample as cleaner
    reload(cleaner)
    clean_sample, f_tall = cleaner.sampleCleaner(signal, f_tall)

    '''
    STEP 2:
    Candidates detection
    '''
    import qrsDetector.sample as sampleDetector
    reload(sampleDetector)
    outputData, beatsPosition, beatInformation = sampleDetector.sampleDetector(clean_sample)


    '''
    STEP 3:
    Candidates characterization
    '''
    import beatWork.characterization.sample as charact
    reload(charact)
    beatsChar, beatsPosition = charact.sampleCharacterizator(clean_sample, beatsPosition, beatInformation, normalize = False)
    
    '''
    STEP 4:
    Candidates identification
    '''
    #import cPickle
    #if deep_iteration != -1:
    import beatWork.identification.sample as identification
    reload(identification)
    
    real_pos_beats, ident_quality, ventriculars = identification.identification(beatsChar, beatsPosition, lead_name = kwargs['lead_name'])

    '''
    STEP 5:
    QualityPerZoneStimation
    '''
    import zoneQualityStimation.penalizedZoneQualityStimation as qualityStimator
    reload(qualityStimator)
    delimiters, zone_quality = qualityStimator.sampleZoneQualityEstimation(clean_sample, 0, len(clean_sample), beatsPosition = real_pos_beats, deep_rec = deep_iteration)
    #return real_pos_beats, zone_quality
            
    '''
    STEP 6:
    Choose zones that has been correctly analyzed and gives bad analyzed zones delimiters, in these cases includes its stimated RR
    '''
    import finishDecision.sample as decisor
    reload(decisor)
    zones_bones, zones_dolentes = decisor.sampleFinishDecision(clean_sample, delimiters, zone_quality, pos_beats = real_pos_beats) #zones dolentes include
    #import finishDecision.finishDecisionMinimumLength as decisor
    #reload(decisor)
    #zones_bones, zones_dolentes = decisor.sampleNotDecision(clean_sample, delimiters, zone_quality, pos_beats = real_pos_beats) #zones dolentes include
    
    '''
    STEP 7:
    Work with correct zones
    '''
    import zoneAnalyzer.goodZoneAnalyzer.sampleGoodZoneAnalyzer as gzA
    reload(gzA)

            
    zones_complete_previous = [] 
    aux_refference_beats = [[],[]]
    zones_complete = []

    from numpy import mean, max, min
    for nn,zone in enumerate(zones_bones):
        #print "profundidad " + str(deep_iteration) + " fent la 1 passada" + str(nn) +" zona neta de " + str(len(zones_bones)) + " usant refs: " +str(len(refference_beats[0])) 
        #if len(refference_beats[0])>0:
        #    print  "  max: " + str(max(refference_beats[1]))  + "  min: " + str(min(refference_beats[1]))+ "  mean: " + str(mean(refference_beats[1])) 
        
        positions, types, quality, quant_group, group_char = gzA.analyzeCorrectZone(clean_sample, zone[0],zone[1], real_pos_beats, ventriculars, list(refference_beats),0.8+0.5*deep_iteration, use_own_beats = True)
        zones_complete_previous.append((zone[0],zone[1], positions, types, 0.0, deep_iteration))
        aux_refference_beats[0] = aux_refference_beats[0] +  group_char
        aux_refference_beats[1] = aux_refference_beats[1] +  quant_group
        
        zones_complete.append((zone[0],zone[1], positions, types, 0.0, deep_iteration))

    refference_beats[0] = refference_beats[0] +  aux_refference_beats[0]
    refference_beats[1] = refference_beats[1] +  aux_refference_beats[1]
    
    '''
    STEP 8:
    Work with incorrect zones
    '''        
    import zoneAnalyzer.badZoneAnalyzer.sampleBadZoneAnalyzer as bzA
    reload(bzA)    
    
    for zone in zones_dolentes:
        aux = zone[2]
        from numpy import isnan
        if isnan(zone[2]) or zone[2] == -1:
            aux = 55
        zones_complete = zones_complete + bzA.analyzeZonaChunga(clean_sample, zone[0], zone[1], real_pos_beats, refference_beats, aux, deep_iteration, f_tall, **kwargs)
    

        
    return zones_complete
