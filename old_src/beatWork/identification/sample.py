
'''

- given a signal and a set of points, this functions has to say if at the given signal in the given point there is a beat or not.

OUTPUT:
- a numpy's array-like filled with booleans. The size of the output must be the same than the size of the input.
- a numpy's array-like filled with an estimation of the quality related to the identification

NOTE: 

- If your method need to be trained, you should think how to save and identify trained information in order to not repeat it each time.5

'''

def identification(beat_charact, pos_beat, thrs = 0.9, **kwargs):
    
    lead_name = kwargs['lead_name']
    import cPickle
    f = open('none_-10_20_' + str(lead_name) + '_serialize_clf.srl','r')
    clf = cPickle.load(f)
    f.close()
    print "carregat!!! Working a saco!!!"
    temp = []
    i = 0
    while i < len(beat_charact):
        i = i + 500            
        temp = temp + list(clf.predict(beat_charact[i-500:i]))
        print "done 500!!!"
    return sorted(list(set([pos_beat[i] for i,x in enumerate(temp) if x == 0 or x == 1]))),[],sorted(list(set([pos_beat[i] for i,x in enumerate(temp) if x == 1]))) 
    
    
    '''
    import cPickle
    f = open('beats','rb')
    beats = cPickle.load(f)  #N
    labels = cPickle.load(f) #labels
    f.close()
    print "carregat!!! Working a saco!!!"
    temp = []
    labs = []
    lkh = []
    i = 0
    from numpy import correlate,max,argmax,mod
    fails = []
    for i,x in enumerate(beat_charact):
        aux = [max(correlate(x,m)) for j,m in enumerate(beats)]
        if  max(aux) > thrs:
            temp.append(i)
            labs.append(labels[argmax(aux)])
        else:
            fails.append(max(aux))
        if mod(i,500)==0:
            print "   van 500 de " + str(len(beat_charact))
            f = open('noentiendoquepasa','wb')
            cPickle.dump(aux,f)
            f.close()
            print str(aux)
            print str(max(aux))
            print str(argmax(aux))

            

    return [pos_beat[x] for x in temp],[],[pos_beat[x] for i,x in enumerate(temp) if labs[i] == 1]
    '''