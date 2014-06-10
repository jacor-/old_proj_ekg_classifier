
'''

Group input data using some criteria.

Input:
  - data:  array with info. Each row is information about one beat
  
Output:
  - groups: characterization of each grup
  - g_map:  relation between each row in data and the group in 'groups' it belongs to. 

'''

def sample_make_groups(waveform, thrs):
    
    grups_ini = []
    from numpy import dot, mean, sqrt,diff,median,max,argmax
    print " exmpezando grupos "
    means = []
    for wave in waveform:
        
        done = False
        lkh = [dot(wave, means[j]) for j,base in enumerate(grups_ini)]
        if len(lkh) > 0 and max(lkh) >= thrs:
            grups_ini[argmax(lkh)].append(wave)
            means[argmax(lkh)] = mean(grups_ini[argmax(lkh)],0)
        else:
            grups_ini.append([wave])
            means.append(wave)
    print " acabando grupos "
    return grups_ini, [mean(x,0) for x in grups_ini],[len(x) for x in grups_ini]
    
    '''
    grups_ini = []
    medias = []
    from numpy import dot, mean, sqrt, diff, median, max, argmax
    for r, wave in enumerate(waveform):
        print str(r) + "   " + str(len(waveform))
        lkh = [dot(wave, medias[j]) for j,base in enumerate(grups_ini)]
        #lkh = [dot(wave/sqrt(wave,wave), medias[j]/sqrt(medias[j],medias[j])) for j,base in enumerate(grups_ini)]
        if len(lkh) > 0 and max(lkh) >= thrs:
            grups_ini[argmax(lkh)].append(wave)
            medias[argmax(lkh)] = mean(grups_ini[argmax(lkh)],0)
            z = medias[argmax(lkh)]
            medias[argmax(lkh)] = z / sqrt(dot(z,z))
        else:
            grups_ini.append([wave])
            medias.append(wave)
    return grups_ini, medias,[len(x) for x in grups_ini]
    '''