
def estimate(beats_pos, begin, end, ENTORN = 6):
    beatsPos = [ x for x in beats_pos if x > begin and x < end]
    from numpy import var, diff, mean,median, array, sqrt,max, arange,min
    RR = diff(beatsPos)
    contorn_rr = [median(RR[max([0,i-ENTORN]):min([i+ENTORN,len(RR)-1])]) for i in range(len(RR))]
    norm_rr = [0]+list(array([RR[i]/contorn_rr[i] for i in range(len(RR))])-1)
    return array(norm_rr)
