from error.stats import findSimilarInSequence

def _work_beat(beat, beats_pos, beats_info, max_distance_accepted, deep, lead, dict_base):
    x = findSimilarInSequence(beats_pos, beat)
    from numpy import abs

    if len(beats_pos)==0:
        nou_dict = dict(dict_base)
        print str(lead)
        nou_dict[lead] = deep
        beats_info.append(nou_dict)
        beats_pos.append(beat)
    
    elif abs(beats_pos[x] - beat) < max_distance_accepted:
        beats_info[x][lead] = deep
    else:
        insert_index = x
        if beats_pos[x] < beat:
            insert_index = insert_index + 1
        
        nou_dict = dict(dict_base)
        nou_dict[lead] = deep
        
        
        beats_info.insert(insert_index,nou_dict)
        beats_pos.insert(insert_index,beat)
        
        
'''
example:
    z1,z2 = lj.sample_lead_joiner([aa,bb,cc], 4, decisor, accepted_beat_score = 6, weight_per_deep = [0,1,2,3,3,3,3,3,3,3,3,3,3,3,3])
'''

def sample_lead_joiner(leads, max_deep, decisor, accepted_beat_score = 6, max_distance_accepted = 10, weight_per_deep = []):
        
    
    beats_pos = []
    beats_info = []
    dict_base = {}
    for i in range(len(leads)):
        dict_base.__setitem__(i,max_deep+1)
    for i,lead in enumerate(leads):
        for interval in lead:
            for beat in interval[2]:
                _work_beat(beat, beats_pos, beats_info, max_distance_accepted,interval[-1], i,dict_base)
            #for resc in interval[3][0]:
            #    for beat in resc:
            #        _work_beat(beat, beats_pos, beats_info, max_distance_accepted,interval[-1], i,dict_base)
    
    from numpy import sum
    return decisor(beats_pos, beats_info, max_deep = max_deep, accepted_beat_score = accepted_beat_score, weight_per_deep = weight_per_deep)
    