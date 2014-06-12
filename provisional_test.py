
from old_src.signalCleaning import cleaners
import old_src.system.holter_io as holter_utils
from numpy import array
from dataset_utils.getData import getData, getDataAndLabels
from dataset_utils.GetTrainAndTestSets import getSignalsTrainTest
#import statsmodels.api as sm
cases = holter_utils.get_usable_cases('cardiosManager')
ids_train, ids_test = getSignalsTrainTest(cases)
#beats_train, beats_test, labels_train, labels_test = getData([ids_train[0]], [ids_test[0]])

def caracterizeBeats(cases):
    order = 3
    beats = []
    labels = []
    dic = {'N'}
    for i,case in enumerate(cases):
        print "Doing " + str(i) + " " + str(len(cases))
        data, labels_ = getDataAndLabels(case)
        for label in zip(labels_[0], labels_[1]):
            try:
                arma_mod30 = sm.tsa.ARMA(data[0][max(0,label[0]-100):min(label[0]+100,len(data[0]))], (order,0)).fit()
                beats.append(arma_mod30.params)
            except:
                beats.append([0]*(order+1))
            
            if label[1] == 'N':
                labels.append(0)
            elif label[1] == 'V':
                labels.append(1)
            else:
                labels.append(2)
    return beats, labels

beats_train, labels_train = caracterizeBeats(ids_train)
beats_test, labels_test = caracterizeBeats(ids_test)

