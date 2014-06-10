
import old_src.system.holter_io as holter_utils
import numpy as np
import old_src.signalCleaning.cleaners as cleaners

def __randomizeData__(beats):
    np.random.shuffle(beats)
    
def __getDataOfASignal__(case):
    data,labs = holter_utils.get_complete_exam(case,'cardiosManager') 
    data[0] = cleaners.hp(data[0], 0.6)
    data[1] = cleaners.hp(data[1], 0.6)
    data[2] = cleaners.hp(data[2], 0.6)
    now_beats = []
    window = [-30,50]
    for pos, lab in zip(labs[0],labs[1]):
        new_beat = data[:,pos+window[0]:pos+window[1]]
        now_beats.append((new_beat.reshape(3*(window[1]-window[0])), lab))
    return now_beats

def __getData__(cases):
    data = []
    for case in cases:
        data += __getDataOfASignal__(case)
    __randomizeData__(data)
    beats, labels = zip(*data)
    return np.array(beats), labels

def getTrainData(ids_train):
    beats_train, labels_train = __getData__(ids_train[:3])    
    return beats_train, labels_train

def getTestData(ids_test):
    beats_test, labels_test = __getData__(ids_test[:3])
    return beats_test, labels_test

def getData(ids_train, ids_test):
    beats_train, labels_train = getTrainData(ids_train)
    beats_test, labels_test = getTrainData(ids_test)
    return beats_train, beats_test, labels_train, labels_test
  