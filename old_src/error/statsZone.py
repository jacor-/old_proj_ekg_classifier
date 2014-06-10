from stats import compareLists, findSimilarInSequence;

def compareZones(refference, zones, max_error_margin = 10, points = False):
    ret_comparison = []
    for interval in zones:
        ini = interval[0]
        fin = interval[1]
        
        index_ini = findSimilarInSequence(refference, ini)
        index_fin = findSimilarInSequence(refference, fin)+1
        real_ref = refference[index_ini:index_fin]
        
        
        tot_beats = list(interval[2])

        
        tot_beats = sorted(list(set(tot_beats)))
        
        aux1 = compareLists(real_ref, interval[2],max_error_margin, points)
        aux2 = compareLists(real_ref, tot_beats,max_error_margin, points)
        
        ret_comparison.append((ini,fin, aux1, aux2,interval[-1]))
        
    from  numpy import sum, arange, max
    levels = max([x[-1] for x in ret_comparison])
    res = {}
    for level in arange(levels+1):
        res[level] = [sum([x[3][0] for x in ret_comparison if x[-1] == level]),sum([x[3][1] for x in ret_comparison if x[-1] == level]),sum([x[3][2] for x in ret_comparison if x[-1] == level])]
        res[level].append(float(res[level][0])/(res[level][0]+res[level][2]))
        res[level].append(float(res[level][0])/(res[level][0]+res[level][1]))
        
    return res
