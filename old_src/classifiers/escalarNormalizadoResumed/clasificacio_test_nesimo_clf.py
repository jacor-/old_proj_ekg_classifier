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


def fesGrups(Xs,Ys,Ns,Vs,ClN,CLV):
	normals = [x for i,x in enumerate(Xs) if Ys[i] == 0]
	if Ns != -1:
		from scipy.cluster.vq import kmeans2,kmeans
		from numpy import log10,zeros,mean
	
		Cl = ClN
		a,b = kmeans2(array(normals), Cl)
		quant = [int(round(float(sum(b==i))/len(normals)*Ns)) for i in range(Cl)]
		quant2 = [x for x in quant]
		quant3n = [int(round(float(x)/sum(quant2)*Ns))  for x in quant2]
		sels = zeros(Cl)
		subsels_norm = []
		for i,x in enumerate(b):
			if sels[x] <= quant3n[x]:
				subsels_norm.append(normals[i])
				sels[x] = sels[x] + 1
			if len(subsels_norm) >= Ns:
				break
	else:
		subsels_norm = normals
	
	ventriculars = [x for i,x in enumerate(Xs) if Ys[i] == 1]
	
	if Vs != -1:
		Cl = CLV
		a,b = kmeans2(array(ventriculars), Cl)
		quant = [int(round(float(sum(b==i))/len(ventriculars)*Vs)) for i in range(Cl)]
		quant2 = [x for x in quant]
		quant3v = [int(round(float(x)/sum(quant2)*Vs))  for x in quant2]
		sels = zeros(Cl)
		subsels_vent = []
		for i,x in enumerate(b):
			if sels[x] <= quant3v[x]:
				subsels_vent.append(ventriculars[i])
				sels[x] = sels[x] + 1
			if len(subsels_vent) >= Vs:
				break
	else:
		subsels_vent = ventriculars
	
	return subsels_norm+subsels_vent,[0]*len(subsels_norm)+[1]*len(subsels_vent)


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


def fes_la_prediccio(Xs_C1,Xs_C2,Xs_C3,Ys,indexs, norm):

	def doTestSVM(data, labels, data_train, label_train, indexs):
		import beatWork.classification.basic_classifiers as cl
		reload(cl)
		clf = cl.clf_SVM(data_train,label_train)
		Z = clf.predict(data)
		del clf
		return Z
		#return getMatrixResult(indexs,labels,Z)
	
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



	
	print str("haciendo 3")
	f=open('predict_'+txt+'norm_resumed_balanced_lineal_C3','wb')
	data1 = array(Xs_C3)
	X_gr,Y_gr = fesGrups(data1,Ys,10000,-1,40,40)
	Z = doTestSVM(data1,Ys,array(X_gr),Y_gr,indexs)
	cPickle.dump(Z,f)
	cPickle.dump(getMatrixResult(indexs,Ys,Z),f)
	f.close()
	print str("   acabando 3")
	
	print str("haciendo 2")
	f=open('predict_'+txt+'norm_resumed_balanced_lineal_C2','wb')
	data1 = array(Xs_C2)
	X_gr,Y_gr = fesGrups(data1,Ys,10000,-1,40,40)
	Z = doTestSVM(data1,Ys,array(X_gr),Y_gr,indexs)
	cPickle.dump(Z,f)
	cPickle.dump(getMatrixResult(indexs,Ys,Z),f)
	f.close()
	print str("   acabando 2")
	
	print str("haciendo 1")
	f=open('predict_'+txt+'norm_resumed_balanced_lineal_C1','wb')
	X_gr,Y_gr = fesGrups(data1,Ys,10000,-1,40,40)
	Z = doTestSVM(data1,Ys,array(X_gr),Y_gr,indexs)
	cPickle.dump(Z,f)
	cPickle.dump(getMatrixResult(indexs,Ys,Z),f)
	f.close()
	print str("   acabando 1")






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
	
	
		
		subplot(2,5,z)
		title('f.d.p.    ' + titol)
		plot(axis_nn,fdp_nn,'k',linewidth=1.4)
		plot(axis_vv,fdp_vv,'k--',linewidth=0.6)
	
		xticks([base_left,float(base_left+base_right)/2,base_right])
		subplot(2,5,z+5)
		title('F.p.   ' + titol)
		plot(axis_nn,fx_nn,'k',linewidth=1.4)
		plot(axis_vv,fx_vv,'k--',linewidth=0.6)
		xticks([base_left,float(base_left+base_right)/2,base_right])
	
	
	
	Ys = array(Ys)
	
	
	def plotCosaNoNorm(index , dataa , Yss):
		figure(index)
		plotaDensitatsRares(dataa,Yss,0,-15,15,'morphology',1,True)
		plotaDensitatsRares(dataa,Yss,1,-400,50, 'amplitude',2,True)
		plotaDensitatsRares(dataa,Yss,2,1,50, 'width',3,False)
		plotaDensitatsRares(dataa,Yss,3,-2,2,'variance distribution',4,True)
		plotaDensitatsRares(dataa,Yss,4,-0.75,0.75,'rr',5,True)
		
	def plotCosaNorm(index , dataa , Yss):
		figure(index)
		plotaDensitatsRares(dataa,Yss,0,-20,20,'morphology',1,True)
		plotaDensitatsRares(dataa,Yss,1,-2,75, 'amplitude',2,True)
		plotaDensitatsRares(dataa,Yss,2,-1,3, 'width',3,False)
		plotaDensitatsRares(dataa,Yss,3,-0.5,1,'variance distribution',4,True)
		plotaDensitatsRares(dataa,Yss,4,-0.75,0.75,'rr',5,True)
	
	if norm == False:	
		plotCosaNoNorm(0,array(Xs_C1), Ys)
		plotCosaNoNorm(1,array(Xs_C2), Ys)
		plotCosaNoNorm(2,array(Xs_C3), Ys)
		data1 = list(Xs_C2)+list(Xs_C3)+list(Xs_C1)
		Ys2 = array(list(Ys)*3)
		plotCosaNoNorm(3,array(data1), Ys2)
	else:
		plotCosaNorm(0,array(Xs_C1), Ys)
		plotCosaNorm(1,array(Xs_C2), Ys)
		plotCosaNorm(2,array(Xs_C3), Ys)
		data1 = list(Xs_C2)+list(Xs_C3)+list(Xs_C1)
		Ys2 = array(list(Ys)*3)
		plotCosaNorm(3,array(data1), Ys2)


Xs_C1,Xs_C2,Xs_C3,Ys,indexs,case_index = getResults(base_filename,path)

fes_la_prediccio(Xs_C1,Xs_C2,Xs_C3,Ys,indexs, norm)

#plotaDensitats(Ys,Xs_C1,Xs_C2,Xs_C3, norm = norm)
