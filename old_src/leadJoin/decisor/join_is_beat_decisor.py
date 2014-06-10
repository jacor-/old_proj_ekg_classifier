

def weighted_decisor(beats_pos, beats_info, **kwargs):
    max_deep = kwargs['max_deep']
    accepted_beat_score = kwargs['accepted_beat_score']
    weight_per_deep = kwargs['weight_per_deep']
    if len(weight_per_deep) == 0:
        weight_per_deep = [i for i in range(max_deep+2)]

    return [beats_pos[i] for i,x in enumerate(beats_info) if sum([weight_per_deep[x[l]] for l in x.keys()]) <= accepted_beat_score], [sum([weight_per_deep[x[l]] for l in x.keys()]) for i,x in enumerate(beats_info) if sum([weight_per_deep[x[l]] for l in x.keys()]) <= accepted_beat_score]
    
def just_one_decisor(beats_pos, beats_info, **kwargs):    
    from numpy import min
    accepted_beat_score = kwargs['accepted_beat_score']
    return [beats_pos[i] for i,x in enumerate(beats_info) if min(x.values()) < accepted_beat_score], [min(x.values()) for i,x in enumerate(beats_info) if min(x.values()) < accepted_beat_score]
