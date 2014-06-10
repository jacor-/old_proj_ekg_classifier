from numpy import array,sqrt,dot,argmax,max,argmin,min,mean,median
from system.settings import *
import signalCleaning.sample as cl
from mdp.nodes import JADENode 
from system.settings import *

cases = h_io.get_usable_cases(REDIAGNOSE_DIAGNOSER)
from time import time
import beatWork.characterization.waveform_characterizers as chr

from numpy import array




import dimensionality_reductor.reductor as red
import dimensionality_reductor.reductors.basicMethods as redMethods
import dimensionality_reductor.reductor as red
import beatWork.classification.basic_classifiers as clf

#metode_reductor = [{'algorithm':redMethods.PCA, 'param':14,'req_labels':True},{'algorithm':redMethods.FischerScore, 'param':6,'req_labels':True}]
metode_reductor = [{'algorithm':redMethods.PCA, 'param':6,'req_labels':True}]
filename = "waveform_norm_FS6"






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
		
		
		f = open('clasificacionPFC/'+case+'_lead1_4_waveform_norm','rb')
		lead1 = cPickle.load(f)
		f.close()
		Xs_C1 = Xs_C1 + list(lead1)
	
		
		f = open('clasificacionPFC/'+case+'_lead2_4_waveform_norm','rb')
		lead2 = cPickle.load(f)
		f.close()
		Xs_C2 = Xs_C2 + list(lead2)
	
		
		f = open('clasificacionPFC/'+case+'_lead3_4_waveform_norm','rb')
		lead3 = cPickle.load(f)
		f.close()
		Xs_C3 = Xs_C3 + list(lead3)
		
		#f = open('clasificacionPFC/'+case+'_lead_all_4_components','rb')
		#lead_tot = cPickle.load(f)
		#f.close()
		#Xs_Combined = Xs_Combined + list(lead_tot)
		
		f = open('clasificacionPFC/'+case+'_valid_labs_4_components','rb')
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


def getMatrixResult(indexs,Z_ref,Z_res):
	res = []
	for i in range(len(indexs)-1):
		Z_ref_loc = Z_ref[indexs[i]:indexs[i+1]]
		Z_res_loc = Z_res[indexs[i]:indexs[i+1]]
		FP = sum(Z_res_loc-Z_ref_loc==1)
		FN = sum(Z_res_loc-Z_ref_loc==-1)
		res.append([len(Z_ref_loc),sum(Z_ref_loc),len(Z_ref_loc)-sum(Z_ref_loc),sum(Z_ref_loc)-FN,len(Z_ref_loc)-sum(Z_ref_loc)-FP,FP,FN])
	return res



def getReductor(reductor_method, trainBeats2, trainLabels2):
	reductor = red.dimensionalityReductor(reductor_method, trainBeats2, trainLabels2)
	return reductor

def getClassifier(clasificador, reduced_trainBeats, trainLabels2):
	if 'params' in clasificador.keys():
		identificador = clasificador['algorithm'](clasificador['params'],reduced_trainBeats,trainLabels2)
	else:
		identificador = clasificador['algorithm'](reduced_trainBeats,trainLabels2)
	return identificador


#clasificador = [
#                  {'algorithm':clf.clf_SVM},
#                  {'algorithm':clf.clf_NearestNeighbors,'params':7},
#                  {'algorithm':clf.clf_NearestNeighbors,'params':3},
#               ]





Ys = array(Ys)

print str("haciendo 3")
f=open('predict_'+filename+'_C3','wb')
data1 = array(Xs_C3)
reductor = getReductor(metode_reductor, data1, Ys)
data1 = reductor.reduce(data1)
clf_f = getClassifier({'algorithm':clf.clf_SVM}, data1, Ys)
Z = clf_f.predict(data1)
cPickle.dump(Z,f)
cPickle.dump(getMatrixResult(indexs,Ys,Z),f)
f.close()
print str("   acabando 3")

print str("haciendo 2")
f=open('predict_'+filename+'_C2','wb')
data1 = array(Xs_C2)
reductor = getReductor(metode_reductor, data1, Ys)
data1 = reductor.reduce(data1)
clf_f = getClassifier({'algorithm':clf.clf_SVM}, data1, Ys)
Z = clf_f.predict(data1)
cPickle.dump(Z,f)
cPickle.dump(getMatrixResult(indexs,Ys,Z),f)
f.close()
print str("   acabando 2")


#print str("haciendo 1")
#f=open('predict_'+filename+'_C1','wb')
#data1 = array(Xs_C1)
#reductor = getReductor(metode_reductor, data1, Ys)
#data1 = reductor.reduce(data1)
#clf_f = getClassifier({'algorithm':clf.clf_SVM}, data1, Ys)
#Z = clf_f.predict(data1)
#cPickle.dump(Z,f)
#cPickle.dump(getMatrixResult(indexs,Ys,Z),f)
#f.close()
#print str("   acabando 1")





#print str("haciendo conjunt")
#f=open('predict_no_booleana_all_CCombined','wb')
#data1 = array(Xs_Combined)
#cPickle.dump(doTestSVM(data1,Ys,data1,Ys,indexs),f)
#f.close()
#print str("   acabando conjunt")


















'''
def plotaDensitatsRares(zzzz,l=-10,r=10):

    v = array([Xs_C1[j] for j in [i for i,x in enumerate(Ys) if x == 1]])
    n = array([Xs_C1[j] for j in [i for i,x in enumerate(Ys) if x == 0]])



    #base_left =  float(median(v[:,zzzz])+mean(v[:,zzzz]))/2 - 2*sqrt(std(v[:,0]))
    #base_right = float(median(n[:,zzzz])+mean(n[:,zzzz]))/2 + 2*sqrt(std(n[:,0]))
    base_left = l
    base_right = r
    nn = [x for x in n[:,zzzz] if  x > base_left and x < base_right]
    vv = [x for x in v[:,zzzz] if  x > base_left and x < base_right]


    def getFx(distribucion):
        return array([sum(distribucion[:i+1]) for i in range(len(distribucion))]).astype('float')/sum(distribucion)


    aux=histogram(vv,200)
    axis_vv = aux[1][:-1]
    fdp_vv = array(aux[0]).astype('float')/max(aux[0])
    fx_vv = getFx(fdp_vv)

    aux=histogram(nn,200)
    fdp_nn = array(aux[0]).astype('float')/max(aux[0])
    axis_nn = aux[1][:-1]
    fx_nn = getFx(fdp_nn)


    figure()
    subplot(211)
    plot(axis_nn,fdp_nn,'k--')
    plot(axis_vv,fdp_vv,'k')
    subplot(212)
    plot(axis_nn,fx_nn,'k--')
    plot(axis_vv,fx_vv,'k')



plotaDensitatsRares(0,-20,20)
plotaDensitatsRares(1,-2,75)
plotaDensitatsRares(2,-1,3)
plotaDensitatsRares(3,-0.5,1)
plotaDensitatsRares(4,-0.75,0.75)
'''