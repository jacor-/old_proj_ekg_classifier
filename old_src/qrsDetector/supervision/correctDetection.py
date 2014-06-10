import qrsDetector.supervision.detection.conflictiveZones as potErrors 
import qrsDetector.supervision.correction.RR_correction as rrCorr
import beatHunter.sample_beat_hunter as bh
from numpy import array
from scipy.cluster.vq import kmeans2
import beatWork.characterization.waveform_characterizers as wc
import beatHunter.sample_beat_hunter as bh

def corrigeRR(sig, pos_beats,K =10):
    new_dets,zones_chungues,thrss = potErrors.detectConflictiveZones(pos_beats, sig)
    nous_candidats = []
    
    
    beats = wc.sampleCharacterizator_JustCenter(sig,new_dets)
    a,b = kmeans2(array(beats),K)
    b = array(b)
    b = [sum(b==i) for i in range(K)]
    for x in zones_chungues:
        nous_candidats = nous_candidats + bh.sampleBeatHunterLikelihoodPFC(sig,x[0],x[1],[a,b],x[2],0.7)

    return new_dets, nous_candidats
