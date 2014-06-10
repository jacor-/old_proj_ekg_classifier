
def recoloca(signal, candidats):
    from numpy import argmax, max, array, abs,argmax,min
    return [argmax(abs(array(signal[max([candidat-5,0]):min([candidat+15,len(signal)])])))-5+candidat  for candidat in candidats]

def recoloca2(signal, candidats):
    from numpy import argmax, max, array, abs,argmax,min
    #return candidats
    return list(set([argmax(abs(array(signal[max([candidat-30,0]):min([candidat+30,len(signal)])])))-30+candidat  for candidat in candidats]))        