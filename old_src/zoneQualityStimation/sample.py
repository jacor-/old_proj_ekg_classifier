
'''
INPUT:

   - signal, inicio y final determina la zona de la que se estima la calidad

OUTPUT:

The output of a qrsDetector is an instance of "structure_QrsCandidates". The information it contains is:
   - a list of delimiter intervals : the first one is the same as inicio and the last one is final.
   - a list of indexs. Each index is the quality stimation of the zone (size is one less than delimiter's intervals longitude).

   
'''
        
def sampleZoneQualityEstimation(signal, beggin, end, **kwargs):
    ENTORN = 4
    beatsPos = [ x for x in kwargs['beatsPosition'] if x > beggin and x < end]
    from numpy import var, diff, mean,median, array, sqrt,max, arange,min
    RR = diff(beatsPos)
    contorn_rr = [median(RR[max([0,i-ENTORN]):min([i+ENTORN,len(RR)-1])]) for i in range(len(RR))]
    norm_rr = array([RR[i]/contorn_rr[i] for i in range(len(RR))])-1
    
    
    '''
    Old criteri
    '''
    deep_rec = kwargs['deep_rec']
    wind = 5 - max(deep_rec,2)
    ys = [var(norm_rr[max(i-wind,0):min(i+wind,len(norm_rr))])/mean(norm_rr[max(i-wind,0):min(i+wind,len(norm_rr))]) for i in range(len(norm_rr))]
    wind2 = [ENTORN,ENTORN,ENTORN,ENTORN,ENTORN,ENTORN,ENTORN]
    contorn2_rr = [mean(ys[max(0,i-wind2[deep_rec]):min(i+wind2[deep_rec],len(ys)-1)]) for i in range(len(ys))]
            
    if len(contorn2_rr) == 0:
        return [beggin] + [end], [1]
    elif len(contorn2_rr) < 2:
        return [beggin] + [end], [contorn2_rr[0]]
    else:
        return [beggin] + beatsPos + [end], [contorn2_rr[0]] + list(contorn2_rr) + [contorn2_rr[-1]]
