from sklearn.cross_validation import train_test_split
from numpy import array
import numpy as np

def getSignalsTrainTest(signal_ids, test_size=0.1):
    ids_train, ids_test, labels_train, labels_test = train_test_split(signal_ids, [0 for i in signal_ids], test_size=test_size)    
    return ids_train, ids_test







