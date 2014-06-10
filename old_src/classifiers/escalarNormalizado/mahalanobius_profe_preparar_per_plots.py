from numpy import array,sqrt,dot,argmax,max,argmin,min,mean,median
from system.settings import *
import signalCleaning.sample as cl
from mdp.nodes import JADENode 
from system.settings import *

cases = h_io.get_usable_cases(REDIAGNOSE_DIAGNOSER)
from time import time
import beatWork.characterization.waveform_characterizers as chr

from numpy import array


#normalizado
#path = 'clasificacionPFC/'
#norm = True
#base_filename = 'components'

#no_normalizado
path = 'clasificacionPFC/'
norm = False
base_filename = 'components_no_norm'


def getResults(base_filename, path):
    cases = h_io.get_usable_cases(REDIAGNOSE_DIAGNOSER)
    a = []
    b = []
    
    Xs_C1 = []
    Xs_C2 = []
    Xs_C3 = []
    Xs_Combined = []
    
    Ys = []
    indexs = [0]
    case_index = {}
    
    for num_case, case in enumerate(cases[:]):
        try:
            print case
            import cPickle
            #f = open('clasificacionPFC/'+case+'_lead1_4_components','rb')
            #Ys = Ys + list(cPickle.load(f))
            #f.close()
            
            
            f = open(path+case+'_lead1_4_'+base_filename,'rb')
            lead1 = cPickle.load(f)
            f.close()
            Xs_C1 = Xs_C1 + list(lead1)
        
            
            f = open(path+case+'_lead2_4_'+base_filename,'rb')
            lead2 = cPickle.load(f)
            f.close()
            Xs_C2 = Xs_C2 + list(lead2)
        
            
            f = open(path+case+'_lead3_4_'+base_filename,'rb')
            lead3 = cPickle.load(f)
            f.close()
            Xs_C3 = Xs_C3 + list(lead3)
            
            f = open(path+case+'_lead_all_4_components','rb')
            lead_tot = cPickle.load(f)
            f.close()
            Xs_Combined = Xs_Combined + list(lead_tot)
            
            f = open(path+case+'_valid_labs_4_components','rb')
            valid_labs = cPickle.load(f)
            f.close()
            tr = {'V':1,'N':0,'S':0}    
            Z = map(lambda x:tr[x],valid_labs[1])
            Ys = Ys + Z
            
            indexs.append(len(Ys))
            case_index[num_case] = case
            
        except:
            print "   error " + case
            pass
    return Xs_C1,Xs_C2,Xs_C3,Ys,indexs,case_index


def plotaDensitats(Yss,dataa, norm = True):
    from pylab import figure,show,plot,subplot,histogram,title,xticks
    
    def plotaDensitatsRares(data_struct, Ys, l=-10,r=10, titol = '',z=1,reverse = False):
    
        if reverse == True:
            data_struct = data_struct * -1
            base_left = -r
            base_right = -l
        else:
            base_left = l
            base_right = r
    
    
        v = array([data_struct[j] for j in [i for i,x in enumerate(Ys) if x == 1]])
        n = array([data_struct[j] for j in [i for i,x in enumerate(Ys) if x == 0]])
    
    
    
        #base_left =  float(median(v[:,zzzz])+mean(v[:,zzzz]))/2 - 2*sqrt(std(v[:,0]))
        #base_right = float(median(n[:,zzzz])+mean(n[:,zzzz]))/2 + 2*sqrt(std(n[:,0]))
        nn = [x for x in n if  x > base_left and x < base_right]
        vv = [x for x in v if  x > base_left and x < base_right]
    
    
        def getFx(distribucion):
            return array([sum(distribucion[:i+1]) for i in range(len(distribucion))]).astype('float')/abs(sum(distribucion))
    
    
        aux=histogram(vv,200)
        axis_vv = aux[1][:-1]
        fdp_vv = array(aux[0]).astype('float')/max([max(aux[0]),1])
        fx_vv = getFx(fdp_vv)
    
    
        aux=histogram(nn,200)
        fdp_nn = array(aux[0]).astype('float')/max([max(aux[0]),1])
        axis_nn = aux[1][:-1]
        fx_nn = getFx(fdp_nn)
    
    
        
        subplot(2,1,1)
        title('f.d.p.    ' + titol)
        plot(axis_nn,fdp_nn,'k',linewidth=1.4)
        plot(axis_vv,fdp_vv,'k--',linewidth=0.6)
    
        xticks([base_left,float(base_left+base_right)/2,base_right])
        subplot(2,1,2)
        title('F.p.   ' + titol)
        plot(axis_nn,fx_nn,'k',linewidth=1.4)
        plot(axis_vv,fx_vv,'k--',linewidth=0.6)
        xticks([base_left,float(base_left+base_right)/2,base_right])

    Yss = array(Yss)
    
    
    
    plotaDensitatsRares(dataa,Yss,-15,15,'morphology',1,False)

        


from numpy import zeros, log10,array,mean
from scipy.cluster.vq import kmeans, kmeans2


from numpy import zeros

def characterizeCosaCovNormals(Xs_C1,Ys):
    normals = [x for i,x in enumerate(Xs_C1) if Ys[i] == 0]
    Ns = 100000
    Cl = 40
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

    #covar = inv(cov(array(subsels_vent+subsels_norm).transpose()))
    covar = inv(cov(array(subsels_norm).transpose()))
    distances_normals = [mahalanobis(x,me,covar) for x in normals]
    distances_ventriculars = [mahalanobis(x,me,covar) for x in ventriculars]
    return distances_normals, distances_ventriculars


Xs_C1,Xs_C2,Xs_C3,Ys,indexs,case_index = getResults(base_filename,path)

nous_C1 = characterizeCosaCovNormals(Xs_C1,Ys)
for i,z in enumerate(Xs_C1):
    Xs_C1[i].append(nous_C1[i])
del nous_C1



#plotaDensitats([0]*len(distances_normals)+[1]*len(distances_ventriculars),array(distances_normals + distances_ventriculars), norm = True)


'''
jugar amb la covariancia -- 

aux = sqrt(diag(covar))
maxs = aux.reshape(len(aux),1)*aux
abs(covar/maxs)
'''

