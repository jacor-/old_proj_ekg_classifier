    
def sampleZoneQualityEstimation(signal, beggin, end, **kwargs):
    ENTORN = 3

    beatsPos = [ x for x in kwargs['beatsPosition'] if x > beggin and x < end]
    from numpy import var, diff, mean,median, array, sqrt,max, arange
    RR = diff(beatsPos)
    contorn_rr = [median(RR[max([0,i-ENTORN]):min([i+ENTORN,len(RR)-1])]) for i in range(len(RR))]
    norm_rr = [RR[i]/contorn_rr[i] for i in range(len(RR))]

    '''
    Nou criteri
    '''
    nou_criteri = sqrt(abs(array(norm_rr)-1))*0.2/0.5
    contorn2_rr = array(nou_criteri)

    for index, punt_conflicte in enumerate(nou_criteri > 0.2):
        if punt_conflicte:
            for real_index in arange(max([index-ENTORN,0]), min([index+ENTORN, len(contorn2_rr)]),1): 
                contorn2_rr[real_index] = max([nou_criteri[real_index]+1, contorn2_rr[real_index]])

    #import cPickle
    #f = open('joseCosaRara', 'wb')
    #cPickle.dump(norm_rr,f)
    #cPickle.dump(contorn2_rr,f)    
    #f.close()

    if len(contorn2_rr) == 0:
        return [beggin] + [end], [1]
    elif len(contorn2_rr) < 2:
        return [beggin] + [end], [contorn2_rr[0]]
    else:
        return [beggin] + beatsPos + [end], [contorn2_rr[0]] + list(contorn2_rr) + [contorn2_rr[-1]]




    