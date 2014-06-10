
from trainSystem.trainer import trainer
import beatWork.characterization.production as chrClass
reload(chrClass)
from beatWork.characterization.caracteritzacioAlternativa import getCharacterization2 as chr_criteria
from beatWork.classification.basic_classifiers import clf_SVM
from system.settings import *
import signalCleaning.cleaners as cl
from numpy import array




cases = h_io.get_usable_cases(REDIAGNOSE_DIAGNOSER)
translator = {'V':1,'N':0,'S':0,0:0,1:1}

def test2(lead, name, predict = False):
    train_data = []
    train_labels = []
    for z,case in enumerate(cases[:]):
        print str(z) + "      of      " + str(len(cases))
        data, labs = h_io.get_complete_exam(case, REDIAGNOSE_DIAGNOSER)
        sig = cl.hp(data[lead],1.)
        sig = cl.bandpass(sig,1.,48.)
        valid_labs = [[x for i,x in enumerate(labs[0]) if labs[1][i] in 'SVN'],[translator[labs[1][i]] for i,x in enumerate(labs[0]) if labs[1][i] in 'SVN']]
        ttrain2 = chr_criteria(sig,valid_labs)
        train_data = train_data + list(ttrain2)
        train_labels = train_labels + list(valid_labs[1])
    
    import beatWork.characterization.derivadas as chr
    
    reload(chr)
    S,me,covar = chr.characterizeCosaCovAll(train_data, [train_labels,[]])
    
    train_data = list(array(train_data).transpose())
    train_data.append(list(S))
    train_data = array(train_data).transpose()
    
    
    
    
    
    
    
    c = chrClass.production_characterization_system()
    c.addPipeline(chr_criteria, [])
    c.addPipeline(chr.characterizeCosaCovNormal, [[],[me,covar]])
    
    tr = trainer("instancias_produccion/"+name+"str(lead)")
    tr.setCharacterizer(c)
    tr.trainClassifier(clf_SVM,train_data, train_labels)
    tr.saveInstance()
    
    if predict == True:
        Z = tr.runPrediction(train_data)
        return Z, train_labels
    else:
        return 


name = "oldTry"
#test1(0,name)
Z = test2(1,name, True)

'''
import cPickle
tr = trainer("instancias_produccion/"+ name+"1")
tr.Load()

res = []
ref = []
for z,case in enumerate(cases[:]):
    print str(z) + "      of      " + str(len(cases))
    data, labs = h_io.get_complete_exam(case, REDIAGNOSE_DIAGNOSER)
    sig = cl.hp(data[1],1.)
    sig = cl.bandpass(sig,1.,48.)
    valid_labs = [[x for i,x in enumerate(labs[0]) if labs[1][i] in 'SVN'],[translator[labs[1][i]] for i,x in enumerate(labs[0]) if labs[1][i] in 'SVN']]

    translator = {'V':1,'N':0,'S':0,0:0,1:1}
    
    valid_labs[1] = array(map(lambda x: translator[x], valid_labs[1]))
    Z = tr.runCharAndPrediction(sig,valid_labs)
    
    print "total det " + str(sum(Z))+ "   hit " + str(sum((Z-valid_labs[1])==0)) + "    fn "+ str(sum((valid_labs[1]-Z)==1)) + "    fp " + str(sum((valid_labs[1]-Z)==-1)) + "  total  " + str(sum(valid_labs[1]))

    res = res + list(Z)
    ref = ref + list(valid_labs[1])




'''