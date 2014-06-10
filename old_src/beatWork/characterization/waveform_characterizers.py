
from estocastic_characterizers import *

 
def __transform_to_second_round__(signal,pos_beats):
    from numpy import ones,median
    #aux = todo_un_poco(signal, pos_beats)

    #c2 = polarityModule(signal,pos_beats)
    #print str(len(c2))
    #print "done3"
    c3 = polarityPhase(signal,pos_beats)
    c3 = c3/median(c3)
    c9 = caracterize_beats_MIT(signal, pos_beats)
    c9 = c9/median(c9)
    #c10 = caracterize_beats_Altura2(signal, pos_beats)
    c12 = area(signal,pos_beats)
    c12 = c12/median(c12)

    c4 = ampladaNormal(signal, pos_beats)
    c4 = c4/median(c4)

    #print str(len(c3))
        
    
    from numpy import array
    return array([[c4[i],c12[i],c3[i],c9[i]] for i in range(len(pos_beats))])

def __transform_2_(signal,pos_beats):
    from numpy import ones
    aux = todo_un_poco(signal, pos_beats)
    l1 = mi_paranoia(ones(10000),[1000])
    from numpy import array
    return array([aux[i][:6] + aux[i][7:7+len(l1)]for i in range(len(aux))])

def todo_un_poco(signal,pos_beats):
    #print "done1"
    l1 = mi_paranoia(signal,pos_beats)
    #print str(len(l1))
    #print "done2"
    c2 = polarityModule(signal,pos_beats)
    #print str(len(c2))
    #print "done3"
    c3 = polarityPhase(signal,pos_beats)
    #print str(len(c3))
    #print "done4"
    l4 = haarRaro(signal,pos_beats,1)
    #print str(len(l4))
    #print "done5"
    l5 = haarRaro(signal,pos_beats,2)
    #print str(len(l5))
    #print "done6"
    l6 = haarRaro(signal,pos_beats,3)
    #print str(len(l6))
    #print "done7"
    l7 = haarRaro(signal,pos_beats,4)
    #print str(len(l7))
    #print "done8"
    l8 = haarRaro(signal,pos_beats,5)
    #print str(len(l8))

    #c5 = characterizeAFul(signal, pos_beats)
    #print "done9"
    c9 = caracterize_beats_MIT(signal, pos_beats)
    #print str(len(c9))
    #print "done10"
    c10 = caracterize_beats_Jose(signal, pos_beats)
    #print str(len(c10))
    #print "done11"
    c11 = caracterize_beats_MIT2(signal, pos_beats)
    #print str(len(c11))
    #print "done12"
    c12 = caracterize_beats_Jose2(signal, pos_beats)
    #print str(len(c12))
    #print "done13"
    c13 = caracterize_RR(signal, pos_beats)
    #print str(len(c13))
    from numpy import array
    
    
    
    return array([[c2[i],c3[i],c9[i],c10[i],c11[i],c12[i],c13[i]] + list(l1[i]) + list(l4[i]) + list(l5[i]) + list(l6[i]) + list(l7[i]) + list(l8[i]) for i in range(len(pos_beats))]) 
        
    
def todo_un_poco_22222(signal,pos_beats):
    #print "done1"
    #print str(len(l1))
    #print "done2"
    
    
    
    c2 = polarityModule(signal,pos_beats)
    #print str(len(c2))
    #print "done3"
    c3 = polarityPhase(signal,pos_beats)
    #print str(len(c3))
    #print "done4"
    l4 = haarRaro(signal,pos_beats,1)
    #print str(len(l4))
    #print "done5"
    #print str(len(l5))
    #print "done6"
    l6 = haarRaro(signal,pos_beats,3)
    #print str(len(l6))
    #print "done7"
    #print str(len(l7))
    #print "done8"
    l8 = haarRaro(signal,pos_beats,5)
    #print str(len(l8))

    #c5 = characterizeAFul(signal, pos_beats)
    #print "done9"
    c9 = caracterize_beats_MIT(signal, pos_beats)
    #print str(len(c9))
    #print "done10"
    c10 = caracterize_beats_Jose(signal, pos_beats)
    #print str(len(c10))
    #print "done11"
    c11 = caracterize_beats_MIT2(signal, pos_beats)
    #print str(len(c11))
    #print "done12"
    c12 = caracterize_beats_Jose2(signal, pos_beats)
    #print str(len(c12))
    #print "done13"
    c13 = caracterize_RR(signal, pos_beats)
    #print str(len(c13))
    from numpy import array
    
    
    
    return array([[c2[i],c3[i],c9[i],c10[i],c11[i],c12[i],c13[i]] +list(l4[i]) + list(l6[i]) + list(l8[i]) for i in range(len(pos_beats))]) 

    

    
    
def caracterize_RR2(signal, beat_position):
    def _noNan(aux):
        from numpy import isnan,isinf
        if isnan(aux): return 0
        elif isinf(aux): return 10
        else: return aux
        
    from numpy import zeros,convolve,median,diff,std,max,abs,mean,array
    fs = 100
    L1 = round(0.160*fs)
    L2 = (0.240*fs)
    
    L11 = 6
    L22 = 15
    
    interesantes = beat_position
    

    FF8 = zeros(len(interesantes));        
    for i,x in enumerate(interesantes):
        FF8[i] = _noNan(median(diff(array(beat_position[interesantes[i-L1]:interesantes[i+L2]]))))
    return FF8

def mi_paranoia(signal, pos_beats):
    from numpy import array,convolve, correlate
    vents = [signal[z-30:z+50] for z in pos_beats]
    vents = [convolve(x,x,mode='ful') for x in vents]
    vents = [correlate(x,x,mode='full') for x in vents]
    vents = [x/max(x) for x in vents]
    return array(vents)



def haarRaro(signal, pos_beats, ordre = 1):
    from numpy import array,convolve, correlate
    vents = [signal[z-10:z+30] for z in pos_beats]
    
    i = ordre
    haar = array([0]*(i+1)+[1]*(i+1))
    vents = [convolve(x,haar, mode = 'same') for x in vents]
    #vents = [correlate(x,x,mode='full') for x in vents]
    #vents = [x/max(x) for x in vents]
    return array(vents)

def polarityVert(signal, pos_beats):
    from numpy import abs,min,max,array
    vents = [signal[z-10:z+30] for z in pos_beats]
    vents = [abs(max(z)/min(z)) for z in vents]
    return vents

def polarityHoriz(signal, pos_beats):
    from numpy import abs,min,max,array,argmax,argmin
    vents = [signal[z-10:z+30] for z in pos_beats]
    vents = [abs(float(argmax(z))/argmin(z)) for z in vents]
    return vents

def polarityPhase(signal, pos_beats):
    from numpy import abs,min,max,array,argmax,argmin,sqrt
    vents = [signal[z-10:z+30] for z in pos_beats]
    a = array([(max(z)-min(z)) for z in vents])
    b = array([(float(argmax(z))-argmin(z)) for z in vents])
    return a/b

def area(signal, pos_beats):
    from numpy import abs,min,max,array,argmax,argmin,sqrt
    vents = [signal[z-20:z+30] for z in pos_beats]
    a = array([sum(z) for z in vents])
    return a


def ampladaNormal(signal, pos_beats):
    from numpy import abs,min,max,array,argmax,argmin,sqrt
    sign = [signal[z]<0 for z in pos_beats]
    vents = [abs(signal[z-50:z+50]*[signal[z-50:z+50]>0,signal[z-50:z+50]<0][sign[i]]) for i,z in enumerate(pos_beats)]
    vents = [(z>max(z)*0.2).astype('float') for z in vents]
    vents_dre = []
    vents_esq = []
    for z in vents:
        for i in range(49):
            if z[50+i]-z[50+i+1] == 1:
                break
        for j in range(49):
            if z[49-j]-z[49-j+1] == -1:
                break
        if i == 48:
            i = 0
        if j == 48:
            j = 0
        vents_dre.append(i)
        vents_esq.append(j)
    
    return array(vents_dre)+array(vents_esq)

def ampladaCorrel(signal, pos_beats):
    from numpy import abs,min,max,array,argmax,argmin,sqrt
    
    vents = mi_paranoia(signal,pos_beats)
    vents = [z-3 for z in vents]
    vents = array([z>0 for z in vents])
    vents_dre = [[i for i in range(len(z)/2-1) if abs(int(z[len(z)/2+i]) - int(z[len(z)/2+i+1])) != 0 ] for z in vents]
    vents_esq = [[i for i in range(len(z)/2-1) if abs(int(z[len(z)/2-i]) - int(z[len(z)/2-i+1])) != 1 ] for z in vents]
    
    for i,z in enumerate(vents_dre):
        if len(z) > 0:
            vents_dre[i] = z[0]
        else:
            vents_dre[i] = 60
    for i,z in enumerate(vents_esq):
        if len(z) > 0:
            vents_esq[i] = z[0]
        else:
            vents_esq[i] = 40
    return array(vents_dre)+array(vents_esq)

def polarityModule(signal, pos_beats):
    from numpy import abs,min,max,array,argmax,argmin,sqrt
    vents = [signal[z-20:z+30] for z in pos_beats]
    a = array([float(max(z))-float(min(z)) for z in vents])
    b = array([(float(argmax(z))-argmin(z)) for z in vents])
    return a*a+b*b


def ampladaBatec(signal, pos_beats):
    from numpy import abs,min,max,array,argmax,argmin,sqrt,diff
    sigaux = signal > 0
    canvis_sigaux = abs(array([0]+list(diff(sigaux))))
    posss = [argmax(abs(signal[z-10:z+10]))+z+10 for z in pos_beats]

    res_ = []
    from mdp.utils import progressinfo
    for z in progressinfo(range(len(posss))):
        minim_index = argmax([posss[z]-100,-1])
        maxim_index = argmin([posss[z]+100,len(signal)-1])    
        for i in range([100,0][minim_index]):
            if canvis_sigaux[posss[z]-i] == 1:
                break
        for j in range([100,1][maxim_index]):
            if canvis_sigaux[posss[z]+j] == 1:
                break
        
        if minim_index == 1:
            i = 0
            
        
        if i+1 == [100,0][minim_index]:
            i = 0
        if j+1 == [100,0][maxim_index]:
            j = 0
        res_.append([i,j])            
        
    
    return res_


def sampleCharacterizator_JustCenter(signal, pos_beats):
    
    beat_position = pos_beats
    from numpy import sqrt, dot,argmax,argmin,zeros

    centroids = beat_position    
    
    from numpy import min, max, zeros
    anteriors = [ zeros(30) for i,x in enumerate(beat_position) if centroids[i]-10 < 0]
    posteriors = [ zeros(30) for i,x in enumerate(beat_position) if centroids[i]+20 > len(signal)]
    batecs_waveform = [signal[centroids[i]-10:centroids[i]+20] for i,x in enumerate(beat_position) if centroids[i]-10 >= 0 and centroids[i]+20 <= len(signal)]
    
    
    b_wave = anteriors + batecs_waveform + posteriors

    from numpy import isnan
    for i,x in enumerate(b_wave):
        for j,y in enumerate(x):
            if isnan(y):
                b_wave[i][j] = 0
    
    return b_wave


def autocorrelation(signal, pos_beats, samples = [-4,-3,-2,-1,1,2,3,4]):
    
    from numpy import arange
    samples = arange(50)
    
    '''
from numpy import array, dot, sqrt,convolve,mean,correlate
from system.settings import *
cases = h_io.get_usable_cases(REDIAGNOSE_DIAGNOSER)
import beatWork.characterization.waveform_characterizers as chr
import extractBeats.extractOneSignalBeats as ext
reload(ext)
reload(chr)
from pylab import figure,plot,show,subplot,savefig,close
#datatot,labs = h_io.get_complete_exam("107-00584",REDIAGNOSE_DIAGNOSER)
#datatot,labs = h_io.get_complete_exam(cases[10],REDIAGNOSE_DIAGNOSER)

chr_v2 = []
chr_n2 = []
zzzzzz = 0



res = []
for case in cases[:]:
    try:
        print str(zzzzzz)
        zzzzzz = zzzzzz+1
        datatot,labs = h_io.get_complete_exam(case,REDIAGNOSE_DIAGNOSER)
        
        import signalCleaning.cleaners as cl
        reload(cl)
        data0 = cl.bandpass(cl.hp(array(datatot[0]).astype('float'),1),0.5,40.)
        from numpy import fft,convolve
        
        #hilbert = lambda x: fft.ifft(-1j*array( list(ones(len(x)/2))+list(ones(len(x)/2)*-1))*fft.fft(x))    
        #if mod(len(data0),2) == 0:
        #    data0 = hilbert(data0)
        #else:
        #    data0 = hilbert(data0[:-1])
        #haar = array([1,1,-1,-1]).astype('float')
        #data0 = convolve(data0, haar)
        
        data1 = cl.bandpass(cl.hp(array(datatot[1]).astype('float'),1),1.,35.)
        data2 = cl.bandpass(cl.hp(array(datatot[2]).astype('float'),1),1.,35.)
        data = mean(array([data0,data1,data2]),0)
        data0 = data
        vent = [i for i,x in enumerate(labs[1]) if x in "V"]
        norm = [i for i,x in enumerate(labs[1]) if x in "NS"]
        
        haar = array([1,-1]).astype('float')
        
        vents = [data0[labs[0][z]-10:labs[0][z]+10] for z in vent]
        vents = [convolve(x,x,mode='full') for x in vents]
        vents = [correlate(x,x,mode='full') for x in vents]
        vents = [x/max(x) for x in vents]
        
        
        
        #normal_indexs = [labs[0][z] for z in norm[:20]] + [labs[0][z] for z in norm[1000:1120]] + [labs[0][z] for z in norm[50000:50120]]
        normal_indexs = [labs[0][z] for z in norm]
        
        norms = [data0[x-10:x+10] for x in normal_indexs]
        norms = [convolve(x,x,mode='full') for x in norms]
        norms = [correlate(x,x,mode='full') for x in norms]
        norms = [x/max(x) for x in norms]
        
        
        pn = clf.predict(norms)
        pv = clf.predict(vents)
        
        print "fallan normales      " + str(sum(pn))           + "        " + str(len(norms)-sum(pn))
        print "fallan ventriculares " + str(len(vents)-sum(pv))+ "        " + str(sum(pv))
        res.append([len(norms)-sum(pn),sum(pv),sum(pn),len(vents)-sum(pv)])
    except:
        pass
        
        
        
        
        
        
        
        
        
        
        
        from scipy.cluster.vq import kmeans,kmeans2
        s = [1]*len(vents)+[0]*len(norms)
        a,b = kmeans2(array(vents+norms),4)
        figure()
        
        from pylab import title
        
        
        subplot(3,1,1)
        for x in vents:
            plot(x, 'r')
        for x in norms:
            plot(x, 'b')
        
        for cc in range(4):
            subplot(323+cc)
            grr = [s[i] for i in range(len(b)) if b[i] == cc]
            if len(grr)>0:
                inddd = float(sum(grr))/len(grr)
                if inddd > 0.7:
                    plot(a[cc],'r')
                elif inddd < 0.3:
                    plot(a[cc],'b')
                else:
                    plot(a[cc],'k')
                    
                
        savefig("res_autoconv/grups/" +case)
        
        for cc in range(4):
            grr = [s[i] for i in range(len(b)) if b[i] == cc]
            if len(grr) > 0:
                index = float(sum(grr))/len(grr)
                if index > 0.9:
                    globals.append(a[cc])
                    globals_labels.append(1)
                if index < 0.1:
                    globals.append(a[cc])
                    globals_labels.append(0)

    except:
        print "merda"
        pass
    


        figure()
        subplot(321)
        for v in vent:
            plot(data0[labs[0][v]-20:labs[0][v]+40],'r')
        subplot(322)
        
        for v in normal_indexs:
            plot(data0[v-20:v+40],'b')
        
        
        
        
        subplot(312)
        #subplot(121)
        for zz in norms:
            plot(zz,'b')
        
        for zz in vents:
            plot(zz,'r')
        #subplot(122)     
            
        subplot(313)
        
        #plot(data0,'k')   
        #plot([labs[0][x] for x in norm], [0 for x in norm],'ob')
        #plot([labs[0][x] for x in vent], [0 for x in vent],'or')
        
        
        
        
        
        #def z_c(x):
        #    xx = [j for j in range(len(x)-1) if x[j]*x[j+1]<=0]
        #    if len(xx) >= 2:
        #        return xx[1]-xx[0]
        #    else:
        #        return len(xx)
        #
        #def my_min(x):
        #    from numpy import argmin
        #    return argmin(x)

                    
                    
        #chr_v = []
        #for v in vents:
        #    aux = v[len(v)/2:]
        #    chr_v.append([z_c(aux),my_min(aux)])
        #    chr_v2.append(chr_v[-1])
            
        #chr_n = []
        #for v in norms:
        #    aux = v[len(v)/2:]
        #    chr_n.append([z_c(aux),my_min(aux)])
        #    chr_n2.append(chr_n[-1])
        
        #subplot(313)
        #plot([x[0] for x in chr_n], [x[1] for x in chr_n], 'og')
        #plot([x[0] for x in chr_v], [x[1] for x in chr_v], 'or')
        
        
        savefig("res_autoconv/" +case)
        close('all')
    except:
        print "merda!!"
        pass

    '''
    
    
    
    
    
    
    
    
    
    
    interval = 20
    center = 150
    min_s = samples[0]
    max_s = samples[-1]
    
    beat_position = pos_beats
    from numpy import sqrt, dot,argmax,argmin,zeros

    centroids = beat_position    
    
    from numpy import min, max, zeros,dot,array
    
    
    anteriors = [ zeros(2*center) for i,x in enumerate(beat_position) if centroids[i]-50-interval+min_s-1 < 0]
    posteriors = [ zeros(2*center) for i,x in enumerate(beat_position) if centroids[i]+50+interval+max_s > len(signal)]
    batecs_waveform = [signal[centroids[i]-center:centroids[i]+center] for i,x in enumerate(beat_position) if centroids[i]-50-interval+min_s-1 >= 0 and centroids[i]+50+interval+max_s <= len(signal)]
    
    
    b_wave = anteriors + batecs_waveform + posteriors
    #b_wave = [array(wave)/sqrt(dot(wave,wave)) for wave in b_wave]
    b_wave = [[dot(wave[center+s-interval:center+s+interval],wave[center-interval:center+interval]) for s in samples   ]for wave in b_wave]

    from numpy import isnan
    for i,x in enumerate(b_wave):
        for j,y in enumerate(x):
            if isnan(y):
                b_wave[i][j] = 0
    
    return b_wave

