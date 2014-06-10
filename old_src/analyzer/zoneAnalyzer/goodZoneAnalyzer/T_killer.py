

def Tkiller(clean_sample, begin, end, pos_beats_candidates, ventriculars, refference_beats, thrs, use_own_beats = False):
    import analyzer.zoneAnalyzer.goodZoneAnalyzer.sampleGoodZoneAnalyzer as tk
    reload(tk)
    candidats, labels, quant_entorn, entorn = tk.analyzeCorrectZone(clean_sample, begin, end, pos_beats_candidates, ventriculars, refference_beats, thrs, use_own_beats = False)
            
            
    return candidats, labels, quant_entorn, entorn
