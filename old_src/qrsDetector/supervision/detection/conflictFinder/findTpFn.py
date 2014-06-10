
def findRRConflict(norm_rr, beats_pos):
    
    from numpy import median,diff,zeros,min,max
    
    FP = [[i+1,median(diff(beats_pos[max([i-1-13,0]):(i-1)]))] for i,x in enumerate(norm_rr) if x < -0.4]
    FN = [[i,median(diff(beats_pos[max([i-1,0]):(i-1)+13]))] for i,x in enumerate(norm_rr) if x >  0.4]
    
    beats_pos2 = list(beats_pos)
    
    
    tot = sorted(FP+FN,key = lambda x: x[0], reverse = False)
    
    zones = []
    cumulative = 0
    for i,interval in enumerate(tot):
        zones.append([interval[0]-2-cumulative, interval[0]-1-cumulative, interval[1]])
        beats_pos2 = beats_pos2[:interval[0]-1-cumulative]+beats_pos2[interval[0]+1-cumulative:]
        cumulative = cumulative + 2
    
    return beats_pos2, zones


'''
Codigo plotting cuadraditos guapos

In [168]: plot(data[0]/max(data[0]),'k')
Out[168]: [<matplotlib.lines.Line2D at 0x2bc45510>]

In [169]: plot(dets, [-0.5 for x in dets], 'ok')
Out[169]: [<matplotlib.lines.Line2D at 0x2bc58750>]

In [170]: plot(new_dets, [-0.75 for x in new_dets], 'xk')
Out[170]: [<matplotlib.lines.Line2D at 0x2bc58110>]

In [171]: 

In [171]: for x in zc:
   .....:     rr = Rectangle((new_dets[x[0]],-1),new_dets[x[1]]-new_dets[x[0]], 2,alpha = 0.05,fill = True,color = 'k')
   .....:     gca().add_patch(rr)
   .....:     
'''