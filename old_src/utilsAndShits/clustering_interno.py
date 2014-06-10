
from numpy import array,sqrt,dot,argmax,max,argmin,min,mean,median
from system.settings import *
import signalCleaning.sample as cl
from mdp.nodes import JADENode 
from system.settings import *
cases = h_io.get_usable_cases(REDIAGNOSE_DIAGNOSER)
from time import time
import beatWork.characterization.waveform_characterizers as chr

normals_main = {}
ventriculars_main = {}

normals_char = {}
ventriculars_char = {}

ventricular_models = []
ventricular_quants = 0
normal_models = []




def fes_grups_given_models(sig,posicions,thrs,medias):
    from numpy import array,sqrt,dot,argmax,max,argmin,min,mean,median,correlate
    posicions2 = [argmax(sig[posicions[i]-5:posicions[i]+5])+posicions[i]-5 for i in range(len(posicions))]
    beats = [sig[posicions2[i]-10:posicions2[i]+30] for i in range(len(posicions2))]
    beats = [i-mean(i) for i in beats]
    beats = [i/sqrt(dot(i,i)) for i in beats]
    pos = [[] for i in range(len(medias))]
    
    for x in range(len(beats)-1):
        thr_empirical = [max(dot(beats[x+1],zz)) for zz in medias ]
        if max(thr_empirical) > thrs:
            pos[argmax(thr_empirical)].append(posicions[x+1])
    return pos




def fes_grups(sig,posicions,thrs):
    from numpy import array,sqrt,dot,argmax,max,argmin,min,mean,median,correlate
    posicions2 = [argmax(sig[posicions[i]-5:posicions[i]+5])+posicions[i]-5 for i in range(len(posicions))]
    beats = [sig[posicions2[i]-20:posicions2[i]+30] for i in range(len(posicions2))]
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

def separeMonstre(pos,medias,labs):
    vs = [x for i,x in enumerate(labs[0]) if labs[1][i] in 'V']
    ventriculars_main_wave = []
    ventriculars_pos = []
    normals_main_wave = []
    normals_pos = []
    
    
    if len(vs) > 0:
        representativitat_v = [float(sum(array([[ii for ii,s in enumerate(pos) if x in s ][0] for x in vs])==zz))/len(pos[zz]) for zz in range(len(pos))]
        representativitat_v2 = [float(sum(array([[ii for ii,s in enumerate(pos) if x in s ][0] for x in vs])==zz))/len(vs) for zz in range(len(pos))]
        #print str(representativitat_v)
        #print str(representativitat_v2)
        for i,x in enumerate(representativitat_v):
            if x > 0.7:
                ventriculars_main_wave.append(medias[i])
                ventriculars_pos.append(pos[i])
            
    ns = [x for i,x in enumerate(labs[0]) if labs[1][i] not in 'NSV']
    nns = [x for i,x in enumerate(labs[0]) if labs[1][i] not in 'NSV']
    if len(nns) > 0:
        representativitat_n = [1.-float(sum(array([[ii for ii,s in enumerate(pos) if x in s ][0] for x in nns])==zz))/len(pos[zz]) for zz in range(len(pos))]
        representativitat_n2 = [float(len(pos[zz]))*representativitat_n[zz]/(len(beats)-len(ns)) for zz in range(len(pos))]
    
        #print str(representativitat_n)
        for i,x in enumerate(representativitat_n2):
            if x > 0.1:
                normals_main_wave.append(medias[i])
                normals_pos.append(pos[i])
    return ventriculars_main_wave, ventriculars_pos, normals_main_wave, normals_pos


for j,case in enumerate(cases[:1]):
    print "empezando el caso " + case + "      " + str(j) + " of " + str(len(cases))
    ref = time()
    data, labs = h_io.get_complete_exam(case,REDIAGNOSE_DIAGNOSER)
    
    data = data.astype('float')
    
    z = JADENode()
    zz = z.execute(data.transpose())
    
    import signalCleaning.cleaners as cl2 

    sig0 = zz.transpose()[0]
    sig0 = cl.sampleCleaner(sig0,1)[0]
    sig0 = cl2.bandpass(sig0,1.,40.)
    #sig0 = cl.sampleCleaner(sig0,1)[0]
    
    posicions2 = [argmax(sig0[labs[0][i]-5:labs[0][i]+5])+labs[0][i]-5 for i in range(len(labs[0]))]
    beats = [sig0[posicions2[i]-10:posicions2[i]+30] for i in range(len(posicions2))]
    beats = [i-mean(i) for i in beats]
    beats = [i/sqrt(dot(i,i)) for i in beats]    
    dif_beats0 = [dot(beats[i],beats[i+1]) for i in range(len(beats)-1)]
    
    
    sig1 = zz.transpose()[1]
    sig1 = cl.sampleCleaner(sig1,1)[0]
    sig1 = cl2.bandpass(sig1,1.,40.)
    #sig1 = cl.sampleCleaner(sig1,1)[0]
    
    posicions2 = [argmax(sig1[labs[0][i]-5:labs[0][i]+5])+labs[0][i]-5 for i in range(len(labs[0]))]
    beats = [sig1[posicions2[i]-10:posicions2[i]+30] for i in range(len(posicions2))]
    beats = [i-mean(i) for i in beats]
    beats = [i/sqrt(dot(i,i)) for i in beats]    
    dif_beats1 = [dot(beats[i],beats[i+1]) for i in range(len(beats)-1)]
    
    
    sig2 = zz.transpose()[2]
    sig2 = cl.sampleCleaner(sig2,1)[0]
    sig2 = cl2.bandpass(sig2,1.,40.)
    #sig2 = cl.sampleCleaner(sig2,1)[0]
    
    posicions2 = [argmax(sig2[labs[0][i]-5:labs[0][i]+5])+labs[0][i]-5 for i in range(len(labs[0]))]
    beats = [sig2[posicions2[i]-10:posicions2[i]+30] for i in range(len(posicions2))]
    beats = [i-mean(i) for i in beats]
    beats = [i/sqrt(dot(i,i)) for i in beats]    
    dif_beats2 = [dot(beats[i],beats[i+1]) for i in range(len(beats)-1)]
    
    #escogida = argmin([var(dif_beats0),var(dif_beats1),var(dif_beats2)])
    qualitat = [ssaa[1] for ssaa in sorted(zip([mean(dif_beats0),mean(dif_beats1),mean(dif_beats2)],[0,1,2]),key=lambda x:x[0],reverse = True)]
    
    ais = min([mean(dif_beats0),mean(dif_beats1),mean(dif_beats2)])
    
    thrs = min([0.8,ais*2])
    thrs = 0.8

    sigs = [sig0,sig1,sig2]
        
    medias,pos = fes_grups(sigs[qualitat[0]],labs[0],thrs)
    ventriculars_main_wave, ventriculars_pos, normals_main_wave, normals_pos = separeMonstre(pos,medias,labs)
    ventricular_quants = ventricular_quants + sum([len(vp2) for vp2 in ventriculars_pos])

    ventriculars_wave_raro = []
    for grup_pos in normals_pos:
        medias2,pos2 = fes_grups(sigs[qualitat[1]],grup_pos,0.9)
        ventriculars_main_wave2, ventriculars_pos2, normals_main_wave2, normals_pos2 = separeMonstre(pos2,medias2,[[labs[0][i] for i,x in enumerate(labs[0]) if x in grup_pos],[labs[1][i] for i,x in enumerate(labs[0]) if x in grup_pos]])
        ventriculars_wave_raro = ventriculars_wave_raro + ventriculars_main_wave2
        ventricular_quants = ventricular_quants + sum([len(vp2) for vp2 in ventriculars_pos2])
    
    medias,pos = fes_grups(sigs[qualitat[1]],labs[0],thrs)
    ventriculars_main_wave22, ventriculars_pos22, normals_main_wave22, normals_pos22 = separeMonstre(pos,medias,labs)
    ventricular_quants = ventricular_quants + sum([len(vp2) for vp2 in ventriculars_pos22])

    
    ventriculars_main_wave = ventriculars_main_wave + ventriculars_wave_raro + ventriculars_main_wave22
    
    ventricular_models = ventricular_models + ventriculars_main_wave
    
    normal_models = normal_models + normals_main_wave

    
    # representativitat dels ventriculars en aquest puto merder
    
    #normals_main[case] = normals_main_wave
    #ventriculars_main[case] = ventriculars_main_wave
    
    
    #reload(chr)
    #vents = []
    #for set_pos in ventriculars_pos:
    #    vents = vents + list(chr.__transform_to_second_round__(sig,set_pos))
    #ventriculars_char[case] = vents
    # 
    # norms = []
    #for set_pos in normals_pos:
    #    norms = norms + list(chr.__transform_to_second_round__(sig,set_pos))
    #normals_char[case] = norms

                
    #import cPickle
    #f = open("selected_lead/"+case,'wb')
    #cPickle.dump(sig,f)
    #cPickle.dump(labs,f)
    #cPickle.dump(normals_pos,f)
    #cPickle.dump(ventriculars_pos,f)
    #cPickle.dump(ventriculars_main_wave,f)
    #cPickle.dump(normals_main_wave,f)
    #f.close()
    
    print "    done  in " + str(time()-ref) + "  seconds"
#    except:
#        print "    falla el caso " + case
