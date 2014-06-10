

def crossed_conv(signal,pos_beats):
    from numpy import convolve,array
    zzz = [[signal[0][x-20:x+30],signal[1][x-20:x+30],signal[2][x-20:x+30]] for x in pos_beats]
    a1 = [sum(abs(convolve(convolve(zz[2],zz[1]),zz[0]))) for zz in zzz] 
    return array(a1)
    
def crossed_corr(signal,pos_beats):
    from numpy import correlate,array
    zzz = [[signal[0][x-20:x+30],signal[1][x-20:x+30],signal[2][x-20:x+30]] for x in pos_beats]
    a1 = [sum(abs(correlate(correlate(zz[2],zz[1]),zz[0]))) for zz in zzz]
    return array(a1)
