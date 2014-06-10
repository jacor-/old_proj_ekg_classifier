
from system.settings import *
from numpy import array
import qrsDetector.setDetectors as dets

#detectores = [dets.detectTomkinsAdaptive,dets.detectTomkinsFix,dets.detectHaar16Adaptive,dets.detectHaar16Fix,dets.detectHilbertAdaptive, dets.detectHilbertFix, dets.detectZeroCrossAdaptive, dets.detectZeroCrossFix][:-1]
#descriptores = ["pantomkinsAdapt","pantomkinsFix","haarAdapt","haarfix","hilbertAdapt", "hilbertFix", "ZeroCrossAdapt", "ZeroCrossFix"][:-1]

detectores = [dets.detectTomkinsAdaptiveExp, dets.detectHaar16AdaptiveExp, dets.detectHilbertAdaptiveExp]#, dets.detectZeroCrossAdaptiveExp]
descriptores = ["panTomkinsAdaptExp","haarAdaptExp","hilbertAdaptExt"]#,"zeroCrossExp"]

cases = h_io.get_usable_cases('cardiosManager')


for index_meth,meth in enumerate(detectores):
    import subprocess
    subprocess.call(['mkdir','resultats/resultatsDeteccio/secondRound/'+descriptores[index_meth]])
        

resultats = {}

offset_index_base = 99
for index_case,case in enumerate(cases[:]):
    
    index_case = index_case
    
    print "cas " + str(index_case)
    data,labs = h_io.get_complete_exam(cases[index_case],REDIAGNOSE_DIAGNOSER) 
    resultats[case] = {}
    for index_meth,meth in enumerate(detectores):
        print "   metode " + descriptores[index_meth]
        for lead in range(3):
            print "       lead " + str(lead)
            import cPickle
            Z = detectores[index_meth](array(data[lead]).astype('float'))
            f = open('resultats/resultatsDeteccio/secondRound/'+descriptores[index_meth]+"/"+case+"_"+str(lead),'wb')
            cPickle.dump(Z, f)
            f.close()

