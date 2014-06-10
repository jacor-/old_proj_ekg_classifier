
def analyzeCorrectZone(clean_sample, begin, end, pos_beats_candidates, ventriculars, refference_beats, thrs, use_own_beats = False):
    batecs_pos = [x for x in pos_beats_candidates if x >= begin and x <= end]
    in_ventriculars = [x for x in ventriculars if x >= begin and x <= end]
    batecs_bons = batecs_pos
    
    from numpy import sqrt, dot,diff,median,var,mean,array
    RR_norm = diff(batecs_pos)/median(diff(batecs_pos))
    conflictive_perduts = [i for i,x in enumerate(RR_norm > 1.2) if x == True]
    conflictive_general = array(RR_norm > 1.2) + array(RR_norm < 0.8)
    
    for i,v in enumerate(batecs_bons):
        if i == 0:
            continue
        if v in in_ventriculars:
            conflictive_general[i-1] = False
    
    begins = [i for i,conf in enumerate(conflictive_general) if i > 0 and not conflictive_general[i-1] and conf]
    ends = [i for i,conf in enumerate(conflictive_general) if i < len(conflictive_general)-2 and not conflictive_general[i+1] and conf]

    margin_left = 1
    margin_right = 1
    for i,v in enumerate(batecs_bons):
        if i < 2: 
            continue
        if v in in_ventriculars:
            begins.append(i-2)
            ends.append(i-2)    
    
    aux = [[],[]]
    afegits = []
    if len(begins) and len(ends) > 0:
        zones_chungas = []
        if begins[0] > ends[0]:
            begins.append(0)
            begins = sorted(begins)
        elif begins[-1] > ends[-1]:
            ends.append(len(conflictive_general)-1)
        
        zones_chungas = zip(map(lambda x: max(x-margin_left, 0),begins),map(lambda x: min(x+margin_right, len(RR_norm)), ends))
        batecs_bons = [x for x in batecs_pos if sum([x > batecs_pos[y[0]] and x < batecs_pos[y[1]] for y in zones_chungas]) == 0]
        
    
    


        from beatWork.characterization.sample import sampleCharacterizator
        batecs_waveform, batecs_bons = sampleCharacterizator(clean_sample, batecs_bons, [])
        
        
        if use_own_beats:
            from beatWork.group.sample import sample_make_groups
            gr, entorn, quant_entorn = sample_make_groups(batecs_waveform, 0.9)
            
            refference_beats[0] = refference_beats[0] +  entorn
            refference_beats[1] = refference_beats[1] +  quant_entorn
        else:
            gr = []
            entorn = []
            quant_entorn = []
            
        import beatHunter.sample_beat_hunter as sampleBeatHunter
        reload(sampleBeatHunter)
        

        
        
        nous_batecs = [x for x in batecs_bons if x >= begin and x <= end]+in_ventriculars
        hunted = []
        for pos in zones_chungas:
            beg = batecs_pos[pos[0]]
            nd = batecs_pos[pos[1]]
            candidats =  sampleBeatHunter.sampleBeatHunter(clean_sample, beg, nd, refference_beats, median(diff(batecs_pos)), thrs)
            h = [x + beg for x in candidats if x+beg not in afegits]            
            hunted = hunted + h
            nous_batecs = nous_batecs + h
            afegits = afegits + h
        
        
        
        nous_batecs = sorted(list(set(nous_batecs)))
        change = True
        while change:
            change = False
            to_b_deleted = [nous_batecs[i+1] for i in range(len(nous_batecs)-1) if nous_batecs[i+1]-nous_batecs[i] < 50]
            for x in to_b_deleted:       
                nous_batecs.remove(x)
                change = True

        good_labels = ['N' for x in nous_batecs]
        for ventr,x in enumerate(nous_batecs):
            if x in ventriculars:
                good_labels[ventr] = 'V'


        return  nous_batecs, [good_labels,afegits],[afegits] , quant_entorn, entorn
    else:
        beats_finals = sorted(list(set([x for x in batecs_bons if x >= begin and x <= end]+in_ventriculars)))
        good_labels = ['N' for x in beats_finals]
        for ventr,x in enumerate(beats_finals):
            if x in beats_finals:
                good_labels[ventr] = 'V'
        return sorted(list(set([x for x in batecs_bons if x >= begin and x <= end]+in_ventriculars))) , [good_labels,[]], [], [],[]
