
from old_src.signalCleaning import cleaners
import old_src.system.holter_io as holter_utils
from numpy import array
from dataset_utils.getData import getData
from dataset_utils.GetTrainAndTestSets import getSignalsTrainTest
 
cases = holter_utils.get_usable_cases('cardiosManager')
ids_train, ids_test = getSignalsTrainTest(cases)
beats_train, beats_test, labels_train, labels_test = getData(ids_train, ids_test)
