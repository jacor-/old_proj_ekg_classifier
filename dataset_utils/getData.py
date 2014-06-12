
import old_src.system.holter_io as holter_utils
import numpy as np
import old_src.signalCleaning.cleaners as cleaners

def __randomizeData__(beats):
    np.random.shuffle(beats)
    
def getDataAndLabels(case):
    data,labs = holter_utils.get_complete_exam(case,'cardiosManager') 
    data[0] = cleaners.hp(data[0], 0.6)
    data[1] = cleaners.hp(data[1], 0.6)
    data[2] = cleaners.hp(data[2], 0.6)
    return data, labs

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

import futures
def __getData__(cases):
    data = []
    executor = futures.ThreadPoolExecutor(max_workers=8)
    futs = [executor.submit(__getDataOfASignal__, case) for case in cases]
    current = 0
    total = len(futs)
    for fut in futures.as_completed(futs):
        current += 1
        print str(float(current)/total*100)
        data += fut.result()
    __randomizeData__(data)
    beats, labels = zip(*data)
    return np.array(beats), labels

def getTrainData(ids_train):
    beats_train, labels_train = __getData__(ids_train)    
    return beats_train, labels_train

def getTestData(ids_test):
    beats_test, labels_test = __getData__(ids_test)
    return beats_test, labels_test

def getData(ids_train, ids_test):
    beats_train, labels_train = getTrainData(ids_train)
    beats_test, labels_test = getTrainData(ids_test)
    return beats_train, beats_test, labels_train, labels_test
  
