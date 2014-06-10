    
    
from system.settings import *
from numpy import array
import signalCleaning.cleaners as cl2
import beatWork.characterization.caracteritzacioAlternativa as chr
import cPickle

reload(cl2)
reload(chr)

def characterizeCosaCovAll(Xs_C1,Ys):
    normals = [x for i,x in enumerate(Xs_C1) if Ys[i] == 0]
    Ns = 100000
    Cl = 40
    from numpy import zeros, log10,array,mean,array
    from scipy.cluster.vq import kmeans, kmeans2
    
    
    from numpy import zeros


    a,b = kmeans2(array(normals), Cl)
    quant = [int(round(float(sum(b==i))/len(normals)*Ns)) for i in range(Cl)]
    quant2 = [max([log10(x),0]) for x in quant]
    quant3n = [int(round(float(x)/sum(quant2)*Ns))  for x in quant2]
    sels = zeros(Cl)
    subsels_norm = []
    for i,x in enumerate(b):
        if sels[x] <= quant3n[x]:
            subsels_norm.append(normals[i])
            sels[x] = sels[x] + 1
        if len(subsels_norm) >= Ns:
            break
    
    
    
    ventriculars = [x for i,x in enumerate(Xs_C1) if Ys[i] == 1]
    
        
    from numpy.linalg import inv
    from scipy.spatial.distance import mahalanobis
    from numpy import cov,mean
    
    from numpy.linalg import inv
    from scipy.spatial.distance import mahalanobis, euclidean
    from numpy import cov,mean,array
    me = mean(normals,0)

    covar = inv(cov(array(ventriculars+subsels_norm).transpose()))
    #covar = inv(cov(array(subsels_norm).transpose()))
    return [mahalanobis(x,me,covar) for x in Xs_C1]


def characterizeDistances(Xs_C1,Ys):
    normals = [x for i,x in enumerate(Xs_C1) if Ys[i] == 0]
    Ns = 100000
    Cl = 40
    from numpy import zeros, log10,array,mean,array
    from scipy.cluster.vq import kmeans, kmeans2
    
    
    from numpy import zeros


    a,b = kmeans2(array(normals), Cl)
    quant = [int(round(float(sum(b==i))/len(normals)*Ns)) for i in range(Cl)]
    quant2 = [max([log10(x),0]) for x in quant]
    quant3n = [int(round(float(x)/sum(quant2)*Ns))  for x in quant2]
    sels = zeros(Cl)
    subsels_norm = []
    for i,x in enumerate(b):
        if sels[x] <= quant3n[x]:
            subsels_norm.append(normals[i])
            sels[x] = sels[x] + 1
        if len(subsels_norm) >= Ns:
            break
    
    
    
    ventriculars = [x for i,x in enumerate(Xs_C1) if Ys[i] == 1]
    
        
    from numpy.linalg import inv
    from scipy.spatial.distance import mahalanobis
    from numpy import cov,mean
    
    from numpy.linalg import inv
    from scipy.spatial.distance import mahalanobis, euclidean,cosine
    from numpy import cov,mean,array
    me = mean(normals,0)


    covar = inv(cov(array(subsels_norm).transpose()))
    mahalanobis_normals = [mahalanobis(x,me,covar) for x in Xs_C1]


    
    return mahalanobis_normals



cases = h_io.get_usable_cases(REDIAGNOSE_DIAGNOSER)

for zz, case in enumerate(cases):
    try:
        print str(zz) + "  of  " + str(len(cases))
        data,labs = h_io.get_complete_exam(case, REDIAGNOSE_DIAGNOSER)
        valid_labs = [[pos for i,pos in enumerate(labs[0]) if labs[1][i] in 'SVN'],[labs[1][i] for i,pos in enumerate(labs[0]) if labs[1][i] in 'SVN']]
        data[0] = cl2.hp(data[0],1.); data[1] = cl2.hp(data[1],1.); data[2] = cl2.hp(data[2],1.)
        data[0] = cl2.bandpass(data[0],1.,45.); data[1] = cl2.bandpass(data[1],1.,45.); data[2] = cl2.bandpass(data[2],1.,45.)
        #data[0] = cl2.normalizer(data[0], valid_labs[0], []); data[1] = cl2.normalizer(data[1], valid_labs[0], []); data[2] = cl2.normalizer(data[2], valid_labs[0], [])
        cars0 = chr.getCharacterization(data[0], valid_labs)
        cars1 = chr.getCharacterization(data[1], valid_labs)
        cars2 = chr.getCharacterization(data[2], valid_labs)
        
            
        f = open('clasificacionPFC/'+case+'_lead1_4_intentoDesesperado','wb')
        cPickle.dump(cars0,f)
        del cars0
        f.close()
        
        f = open('clasificacionPFC/'+case+'_lead2_4_intentoDesesperado','wb')
        cPickle.dump(cars1,f)
        del cars1
        f.close()
        
        f = open('clasificacionPFC/'+case+'_lead3_4_intentoDesesperado','wb')
        cPickle.dump(cars2,f)
        del cars2
        f.close()
        
        f = open('clasificacionPFC/'+case+'_valid_labs_4_intentoDesesperado','wb')
        cPickle.dump(valid_labs,f)
    except:
        continue

Xs_C1 = []
Xs_C2 = []
Xs_C3 = []
Ys_tot = []
indexs = [0]
tr = {'V':1,'S':0,'N':0}
casos_bons = []
for case in cases:
    try:
        print "lead 1"
        f = open('clasificacionPFC/'+case+'_lead1_4_intentoDesesperado','rb')
        cars0 = cPickle.load(f)
        f.close()
        Xs_C1 = Xs_C1 + list(cars0)
        
        print "lead 2"
        f = open('clasificacionPFC/'+case+'_lead2_4_intentoDesesperado','rb')
        cars1 = cPickle.load(f)
        f.close()     
        Xs_C2 = Xs_C2 + list(cars1)
        
        print "lead 3"
        f = open('clasificacionPFC/'+case+'_lead3_4_intentoDesesperado','rb')
        cars2 = cPickle.load(f)
        f.close()
        Xs_C3 = Xs_C3 + list(cars2)
        
        print "lead labels"
        f = open('clasificacionPFC/'+case+'_valid_labs_4_intentoDesesperado','rb')
        Ys = map(lambda x: tr[x],list(cPickle.load(f))[1])
        Ys_tot = Ys_tot + Ys
        
        
        print str(len(cars0)) + "   " + str(len(Ys))
        indexs.append(len(Ys_tot))
        casos_bons.append(case)
    except:
        continue


dist0 = characterizeDistances(array(Xs_C1), Ys_tot)
aux = list(array(Xs_C1).transpose())
aux.append(list(dist0))
Xs_C1 = list(array(aux).transpose())


dist0 = characterizeDistances(array(Xs_C2), Ys_tot)
aux = list(array(Xs_C2).transpose())
aux.append(list(dist0))
Xs_C2 = list(array(aux).transpose())


dist0 = characterizeDistances(array(Xs_C3), Ys_tot)
aux = list(array(Xs_C3).transpose())
aux.append(list(dist0))
Xs_C3 = list(array(aux).transpose())

f = open('clasificacionPFC/cases_intentoDesesperado','wb')
cPickle.dump(casos_bons,f)
f.close()

f = open('clasificacionPFC/indexs_intentoDesesperado','wb')
cPickle.dump(indexs,f)
f.close()
del cars0

for z,case in enumerate(cases):
    try:
        f = open('clasificacionPFC/'+case+'_lead1_4_intentoDesesperado','wb')
        cars0 = cPickle.dump(Xs_C1[indexs[z]:indexs[z+1]],f)
        f.close()
        del cars0
        
        
        f = open('clasificacionPFC/'+case+'_lead2_4_intentoDesesperado','wb')
        cars0 = cPickle.dump(Xs_C2[indexs[z]:indexs[z+1]],f)
        f.close()
        del cars0
        
        f = open('clasificacionPFC/'+case+'_lead3_4_intentoDesesperado','wb')
        cars0 = cPickle.dump(Xs_C3[indexs[z]:indexs[z+1]],f)
        f.close()
        del cars0    
    except:
        continue

