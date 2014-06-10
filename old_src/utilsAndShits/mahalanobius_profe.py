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
#base_filename = 'mahalanobius_no_normalized'
#base_filename = 'componentesPruebaJoseDef_covNorm_no_norm_signal'
base_filename = 'intentoDesesperado'

from numpy import zeros, log10,array,mean
from scipy.cluster.vq import kmeans, kmeans2


from numpy import zeros

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

    covar = inv(cov(array(ventriculars+subsels_norm).transpose()))
    #covar = inv(cov(array(subsels_norm).transpose()))
    mahalanobis_tots = [mahalanobis(x,me,covar) for x in Xs_C1]

    covar = inv(cov(array(subsels_norm).transpose()))
    mahalanobis_normals = [mahalanobis(x,me,covar) for x in Xs_C1]

    covar = inv(cov(array(ventriculars).transpose()))
    mahalanobis_ventriculars = [mahalanobis(x,me,covar) for x in Xs_C1]
    
    euclidean__ = [euclidean(x,me) for x in Xs_C1]
    cosine__ = [cosine(x,me) for x in Xs_C1]
    
    return [mahalanobis_tots,mahalanobis_normals, mahalanobis_ventriculars, euclidean__, cosine__]
    

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

def characterizeCosaCovNormals(Xs_C1,Ys):
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

    #covar = inv(cov(array(ventriculars+subsels_norm).transpose()))
    covar = inv(cov(array(subsels_norm).transpose()))
    return [mahalanobis(x,me,covar) for x in Xs_C1]

def characterizeCosaCovVentriculars(Xs_C1,Ys):
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

    #covar = inv(cov(array(ventriculars+subsels_norm).transpose()))
    covar = inv(cov(array(ventriculars).transpose()))
    return [mahalanobis(x,me,covar) for x in Xs_C1]

def characterizeCosaEuclidean(Xs_C1,Ys):
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

    #covar = inv(cov(array(ventriculars+subsels_norm).transpose()))
    #covar = inv(cov(array(ventriculars).transpose()))
    return [euclidean(x,me) for x in Xs_C1]

def characterizeCosaCosine(Xs_C1,Ys):
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

    #covar = inv(cov(array(ventriculars+subsels_norm).transpose()))
    #covar = inv(cov(array(ventriculars).transpose()))
    return [cosine(x,me) for x in Xs_C1]


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
            
            #f = open(path+case+'_lead_all_4_components','rb')
            #lead_tot = cPickle.load(f)
            #f.close()
            #Xs_Combined = Xs_Combined + list(lead_tot)
            
            #f = open(path+case+'_valid_labs_4_componentesPruebaJoseDef','rb')
            f = open(path+case+'_valid_labs_4_components','rb')
            valid_labs = cPickle.load(f)
            f.close()
            tr = {'V':1,'N':0,'S':0,'0':0,'1':1,0:0,1:1}    
            Z = map(lambda x:tr[x],valid_labs)
            Ys = Ys + list(Z)
            
            print str(len(Ys)) + "    " + str(len(Xs_C1))+"     " + str(len(lead1))
            
            indexs.append(len(Ys))
            case_index[num_case] = case
            
        except:
            print "   error " + case
            pass
    return Xs_C1,Xs_C2,Xs_C3,Ys,indexs,case_index


def fes_la_prediccio(Xs_C1,Xs_C2,Xs_C3,Ys,indexs, index_cases,norm, excluded, text_clf, text_result):

    def doTestSVM(data, labels, data_train, label_train, indexs,texto):
        import beatWork.classification.basic_classifiers as cl
        reload(cl)
        import cPickle
        try:
            f = open('resultatsEntrenament/'+texto,'rb')
            clf = cPickle.load(f)
            f.close()
            Z = clf.predict(data)
        except:
            f = open('resultatsEntrenament/'+texto,'wb')
            clf = cl.clf_SVM(data_train,label_train)
            cPickle.dump(clf,f)
            f.close()
            Z = clf.predict(data)
        del clf
        return Z
        #return getMatrixResult(indexs,labels,Z)

    def doTestNearestNeighbors(K, data, labels, data_train, label_train, indexs):
        import beatWork.classification.basic_classifiers as cl
        reload(cl)
        clf = cl.clf_NearestNeighbors(K,data_train,label_train)
        Z = clf.predict(data)
        del clf
        return Z

    def getMatrixResult(indexs,Z_ref,Z_res):
        res = []
        for i in range(len(indexs)-1):
            Z_ref_loc = Z_ref[indexs[i]:indexs[i+1]]
            Z_res_loc = Z_res[indexs[i]:indexs[i+1]]
            FP = sum(Z_res_loc-Z_ref_loc==1)
            FN = sum(Z_res_loc-Z_ref_loc==-1)
            res.append([len(Z_ref_loc),sum(Z_ref_loc),len(Z_ref_loc)-sum(Z_ref_loc),sum(Z_ref_loc)-FN,len(Z_ref_loc)-sum(Z_ref_loc)-FP,FP,FN])
        return res
    
    Ys = array(Ys)
    
    txt = ''
    if norm == False:
        txt = 'no_'
    import cPickle


    def subsel_train(Ys,Xs):
        from numpy import cov
        from numpy.linalg import inv
        #whitened = dot(inv(cov(data1.transpose())),data1.transpose())
        whitened = Xs
        
        from numpy import mod
        
        train_data1 = []
        Y_train  = []
        cont = 0
        for i,x in enumerate(whitened):
            if Ys[i] == 1:
                Y_train.append(1)
                train_data1.append(x)
            else:
                if mod(cont,4) == 0:
                    Y_train.append(0)
                    train_data1.append(x)
                cont = cont +1
        return array(train_data1),array(Y_train)

    def ignore_excluded(Ys,Xs,excluded,index_cases, index):
        from numpy import cov
        from numpy.linalg import inv
        #whitened = dot(inv(cov(data1.transpose())),data1.transpose())
        Xs_tot = list(Xs)
        Ys_tot = list(Ys)
        for i,case in enumerate(index_cases):
            if case in excluded:
                Xs_tot = Xs_tot[index[:i]]+Xs_tot[index[i+1:]]
                Ys_tot = Ys_tot[index[:i]]+Ys_tot[index[i+1:]]
        return array(Xs_tot),array(Ys_tot)
                
                
            
    #train_data1 = array([x  if mod(i,4)==0 and Ys[i] == 0]+[x for i,x in enumerate(whitened) if Ys[i] == 1])
    #Y_train = array([Ys[i] for i,x in enumerate(whitened) if mod(i,4)==0 and Ys[i] == 0]+[Ys[i] for i,x in enumerate(whitened) if Ys[i] == 1])
    
    print str("haciendo 1")
    f=open('predict_'+txt+text_result+'_C1','wb')
    data1 = array(Xs_C1)
    tr_X,tr_Y = ignore_excluded(Ys,data1,excluded,index_cases,indexs)
    #tr_X = data1
    #tr_Y = Ys
    Z = doTestSVM(data1,Ys,tr_X,tr_Y,indexs,text_clf+'C1')
    cPickle.dump(Z,f)
    cPickle.dump(getMatrixResult(indexs,Ys,Z),f)
    f.close()
    print str("   acabando 1")
    
    print str("haciendo 2")
    f=open('predict_'+txt+text_result+'_C2','wb')
    data1 = array(Xs_C2)
    Z = doTestSVM(data1,Ys,data1,Ys,indexs,text_clf+'C2')
    #tr_X,tr_Y = subsel_train(Ys,data1)
    tr_X,tr_Y = ignore_excluded(Ys,data1,excluded,index_cases,indexs)
    #tr_X = data1
    #tr_Y = Ys
    cPickle.dump(Z,f)
    cPickle.dump(getMatrixResult(indexs,Ys,Z),f)
    f.close()
    print str("   acabando 2")

    print str("haciendo 3")
    f=open('predict_'+txt+text_result+'_C3','wb')
    data1 = array(Xs_C3)
    #tr_X,tr_Y = subsel_train(Ys,data1)
    tr_X,tr_Y = ignore_excluded(Ys,data1,excluded,index_cases,indexs)
    #tr_X = data1
    #tr_Y = Ys
    Z = doTestSVM(data1,Ys,tr_X,tr_Y,indexs,text_clf+'C3')
    cPickle.dump(Z,f)
    cPickle.dump(getMatrixResult(indexs,Ys,Z),f)
    f.close()
    print str("   acabando 3")
    

    #print str("haciendo 1")
    #f=open('predict_'+txt+'norm_all_C1','wb')
    #data1 = array(Xs_C1)
    #Z = doTestNearestNeighbors(5,data1,Ys,data1,Ys,indexs)
    #cPickle.dump(Z,f)
    #return getMatrixResult(indexs,Ys,Z)
    #f.close()
    #print str("   acabando 1")





def plotaDensitats(Ys,Xs_C1,Xs_C2,Xs_C3, norm = True):
    from pylab import figure,show,plot,subplot,histogram,title,xticks
    
    def plotaDensitatsRares(data_struct, Ys, zzzz,l=-10,r=10, titol = '',z=1,reverse = True):
    
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
        nn = [x for x in n[:,zzzz] if  x > base_left and x < base_right]
        vv = [x for x in v[:,zzzz] if  x > base_left and x < base_right]
    
    
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
    
    
        
        subplot(2,6,z)
        title('hist.    ' + titol)
        plot(axis_nn,fdp_nn,'k',linewidth=1.4)
        plot(axis_vv,fdp_vv,'k--',linewidth=0.6)
    
        xticks([base_left,float(base_left+base_right)/2,base_right])
        subplot(2,6,z+6)
        title('hist.   ' + titol)
        plot(axis_nn,fx_nn,'k',linewidth=1.4)
        plot(axis_vv,fx_vv,'k--',linewidth=0.6)
        xticks([base_left,float(base_left+base_right)/2,base_right])
    
    
    from numpy import array
    Ys = array(Ys)
    
    
    def plotCosaNoNorm(index , dataa , Yss):
        figure(index)
        #[mahalanobis_tots,mahalanobis_normals, mahalanobis_ventriculars, euclidean__, cosine__]
        plotaDensitatsRares(dataa,Yss,0,-1,13,'morphology',1,False)
        plotaDensitatsRares(dataa,Yss,1,-400,100, 'amplitude',2,False)
        plotaDensitatsRares(dataa,Yss,2,-1,50, 'width',3,False)
        plotaDensitatsRares(dataa,Yss,3,-1,2,'variance distribution',4,False)
        plotaDensitatsRares(dataa,Yss,4,-1.6,0.8,'rr',5,False)
        plotaDensitatsRares(dataa,Yss,5,-0.2,40,'mahalanobis',6,False)
        
    def plotCosaNorm(index , dataa , Yss):
        figure(index)
        plotaDensitatsRares(dataa,Yss,0,-20,20,'morphology',1,False)
        plotaDensitatsRares(dataa,Yss,1,-10,75, 'amplitude',2,False)
        plotaDensitatsRares(dataa,Yss,2,-1,3, 'width',3,False)
        plotaDensitatsRares(dataa,Yss,3,-0.5,1,'variance distribution',4,False)
        plotaDensitatsRares(dataa,Yss,4,-0.75,0.75,'rr',5,False)
    
    if norm == False:    
        plotCosaNoNorm(0,array(Xs_C1), Ys)
        #plotCosaNoNorm(1,array(Xs_C2), Ys)
        #plotCosaNoNorm(2,array(Xs_C3), Ys)
        #data1 = list(Xs_C2)+list(Xs_C3)+list(Xs_C1)
        #Ys2 = array(list(Ys)*3)
        #plotCosaNoNorm(3,array(data1), Ys2)
    else:
        plotCosaNorm(0,array(Xs_C1), Ys)
        plotCosaNorm(1,array(Xs_C2), Ys)
        plotCosaNorm(2,array(Xs_C3), Ys)
        data1 = list(Xs_C2)+list(Xs_C3)+list(Xs_C1)
        Ys2 = array(list(Ys)*3)
        plotCosaNorm(3,array(data1), Ys2)

def plotaDensitatPerCase(Ys,Xs_C1,Xs_C2,Xs_C3, index_cases, indexs,norm = True):
    from pylab import figure,show,plot,subplot,histogram,title,xticks,yticks
    from numpy import array
    def plotaDensitatsRares(data_struct, case, Ys, zzzz,l=-10,r=10, titol = '',z=1,reverse = True):
    
        if reverse == True:
            data_struct = data_struct * -1
            base_left = -r
            base_right = -l
        else:
            base_left = l
            base_right = r
    
    

        n = array([data_struct[j] for j in [i for i,x in enumerate(Ys) if x == 0]])
    
    
    
        #base_left =  float(median(v[:,zzzz])+mean(v[:,zzzz]))/2 - 2*sqrt(std(v[:,0]))
        #base_right = float(median(n[:,zzzz])+mean(n[:,zzzz]))/2 + 2*sqrt(std(n[:,0]))

        nn = [x for x in n[:,zzzz] if  x > base_left and x < base_right]
    
    
        def getFx(distribucion):
            return array([sum(distribucion[:i+1]) for i in range(len(distribucion))]).astype('float')/abs(sum(distribucion))
    
    

    
    
        aux=histogram(nn,200)
        fdp_nn = array(aux[0]).astype('float')/max([max(aux[0]),1])
        axis_nn = aux[1][:-1]
    
    
        
        #subplot(Y_max,6,z+6*Y_general)
        title('hist. ' + case)
        plot(axis_nn,fdp_nn,'k',linewidth=1.4)
        #plot(axis_vv,fdp_vv,'k--',linewidth=0.6)
    
        xticks([base_left,base_right])
        yticks([])
    from numpy import array
    Ys = array(Ys)
    
    
    def plotCosaNoNorm(case, dataa , Yss):
        #[mahalanobis_tots,mahalanobis_normals, mahalanobis_ventriculars, euclidean__, cosine__]
        plotaDensitatsRares(dataa,case,Yss,5,-0.2,40,'mahalanobis',6,False)


    from numpy import mod
    from pylab import savefig,figure,show,close,plot
    cont = 0
    last = True
    z = 0
    for i in index_cases:
        if mod(cont,16) == 0:
            if last == True:
                figure(figsize=(16, 12))
                last = False
            else:
                pass
        else:
            last = True
            
        
        case = index_cases[i]
        if indexs[i+1] != indexs[i]:
            subplot(4,4,1+mod(cont,16))
            plotCosaNoNorm(case, array(Xs_C2[indexs[i]:indexs[i+1]]) , array(Ys[indexs[i]:indexs[i+1]]))
            cont = cont + 1

        
        if mod(cont,16) == 0 or i == len(index_cases)-1:
            if i == len(index_cases)-1:
                savefig('mahalanobis_normals_hist_'+str(z+1))
                z = z + 1
                print "saving!"
                close()
            else:
                savefig('mahalanobis_normals_hist_'+str(z))
                z = z + 1
                print "saving!"
                close()

Xs_C1,Xs_C2,Xs_C3,Ys,indexs,case_index = getResults(base_filename,path)
print str(len(Xs_C1)) + "      " + str(len(Ys))

'''
nous_C1 = characterizeCosaCovNormals(Xs_C1,Ys)
for i,z in enumerate(Xs_C1):
    Xs_C1[i][-2] = nous_C1[i] 
print "fet C1"

nous_C2 = characterizeCosaCovNormals(Xs_C2,Ys)
for i,z in enumerate(Xs_C2):
    Xs_C2[i][-2] = nous_C2[i]

print "fet C2"
nous_C3 = characterizeCosaCovNormals(Xs_C3,Ys)
for i,z in enumerate(Xs_C3):
    Xs_C3[i][-2] = nous_C3[i]

mahalanobius_distance = [nous_C1,nous_C2,nous_C3]
'''

#print "fet C3"
#


#plotaDensitats([0]*len(distances_normals)+[1]*len(distances_ventriculars),array(distances_normals + distances_ventriculars), norm = True)

'''
Calcule y append nueva distancia!!!
\\\\
nous_dists_C1 = characterizeDistances(Xs_C1,Ys)
nous_dists_C2 = characterizeDistances(Xs_C2,Ys)
nous_dists_C3 = characterizeDistances(Xs_C3,Ys)

#plotaDensitats(Ys,array(nous_dists).transpose(),[],[], norm = False)



Xs_C1 = array(list(array(Xs_C1).transpose())+[nous_dists_C1[1]]).transpose()
Xs_C2 = array(list(array(Xs_C2).transpose())+[nous_dists_C2[1]]).transpose()
Xs_C3 = array(list(array(Xs_C3).transpose())+[nous_dists_C3[1]]).transpose()

import cPickle
for i in case_index:
    case = case_index[i]
    print str("guardant C1  " + str(i) + "  of "  + str(len(case_index)))
    f = open(path+case+'_lead1_4_'+'mahalanobius_no_normalized','wb')
    cPickle.dump(Xs_C1[indexs[i]:indexs[i+1]],f)
    f.close()
for i in case_index:
    case = case_index[i]
    print str("guardant C2  " + str(i) + "  of "  + str(len(case_index)))
    f = open(path+case+'_lead2_4_'+'mahalanobius_no_normalized','wb')
    cPickle.dump(Xs_C2[indexs[i]:indexs[i+1]],f)
    f.close()
for i in case_index:
    case = case_index[i]
    print str("guardant C3  " + str(i) + "  of "  + str(len(case_index)))
    f = open(path+case+'_lead3_4_'+'mahalanobius_no_normalized','wb')
    cPickle.dump(Xs_C3[indexs[i]:indexs[i+1]],f)
    f.close()
import cPickle
for i in case_index:
    case = case_index[i]
    f = open(path+case+'_valid_labs_4_components','wb')
    cPickle.dump(Ys[indexs[i]:indexs[i+1]],f)
    f.close()
    
'''
    


#print "perdicciendo"
excluded = []#['107-00584','107-00679','107-00698','107-00652','107-00697','107-00636']
fes_la_prediccio(Xs_C1,Xs_C2,Xs_C3,Ys,indexs,case_index, norm, excluded, 'intent_desesperat-26-06-2012-clf','intent_desesperat-26-06-2012')











#plotaDensitatPerCase(array(Ys),array(Xs_C1),array(Xs_C2),array(Xs_C3), case_index, indexs,norm = False)


'''
jugar amb la covariancia -- 

aux = sqrt(diag(covar))
maxs = aux.reshape(len(aux),1)*aux
abs(covar/maxs)
'''




























