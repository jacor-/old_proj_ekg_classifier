
from qrsDetector.supervision.detection.conflictFinder    import findTpFn
from qrsDetector.supervision.detection.qualityEstimation import normalizedRRestimator

def detectConflictiveZones(dets, data):
    reload(findTpFn)
    reload(normalizedRRestimator)
    thrss = normalizedRRestimator.estimate(dets, 0, len(data))
    new_dets,zones_chungues = findTpFn.findRRConflict(thrss, dets)
    return new_dets,zones_chungues,thrss

