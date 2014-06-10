
def maxs_rel_positions(sig, reverse = False):
    if reverse == True:
        sig = 1./sig
    #for i in range(len(sig)-2):
    #    if sig[i+1] == sig[i+2]:
    #        sig[i+1] = (sig[i]+sig[i+1])/2
    return [i+1 for i in range(len(sig)-2) if sig[i] <= sig[i+1] >= sig[i+2]]
