from numpy import array, zeros,argmax, argmin, abs


def findSimilarInSequence(sequence, item):
    def _find(sequence, item, index, left, right):    
        if index <= 0:
            return 0
        elif index >= len(sequence)-1:
            return len(sequence)-1
        elif sequence[index] == item:
            return index
        elif left +1 == right or left >= right:
            return index+argmin([abs(item-sequence[left]),abs(item-sequence[right])])

        elif sequence[index] > item:
            right = index
            index = (left+right)/2
            return _find(sequence,item,index,left,right)
        else:
            left = index
            index = (left+right)/2
            return _find(sequence,item,index,left,right)
    return _find(sequence, item, len(sequence)/2, 0,len(sequence))

def compareLists(list1, list2, max_error_margin = 10, points = False):
    resultats_in_list1 = zeros(len(list1))
    resultats_in_list2 = zeros(len(list2))
    '''
    tp --> ones in resultats_in_list1
    fn --> zeros in resultats_in_list1
    fp --> zeros in resultats_in_list2
    '''
    if len(list2) > 0:
        for index_list1,element in enumerate(list1):
            index_list2 = findSimilarInSequence(list2, element)
            most_similar = list2[index_list2]
            if abs(most_similar-element) <= max_error_margin and not resultats_in_list2[index_list2]:
                resultats_in_list1[index_list1] = 1
                resultats_in_list2[index_list2] = 1
    tp = resultats_in_list2.sum()
    fp = len(resultats_in_list2) - tp
    fn = len(resultats_in_list1) - resultats_in_list1.sum()
    if points:
        return [list2[i] for i,x in enumerate(resultats_in_list2) if x], [list2[i] for i,x in enumerate(resultats_in_list2) if not x], [list1[i] for i,x in enumerate(resultats_in_list1) if not x]
    return tp, fp, fn, float(tp)/(tp+fn), float(tp)/(tp+fp)

def compareCardios(list1, list_cardios2, max_error_margin = 10, points = False):
    
    tr = {0:0,1:1,'V':1,'N':0,'S':0}
    
    
    list1 = [[list1[0][i] for i,x in enumerate(list1[1]) if x in 'SVN'],[list1[1][i] for i,x in enumerate(list1[1]) if x in 'SVN']]
    labs_ref = [list1[0],map(lambda x:tr[x],list1[1])]
    
    
    list_cardios2 = [[list_cardios2[0][i] for i,x in enumerate(list_cardios2[1]) if x in 'SVN'],[list_cardios2[1][i] for i,x in enumerate(list_cardios2[1]) if x in 'SVN']]
    poss = [findSimilarInSequence(list_cardios2[0],x) for i,x in enumerate(list1[0])]
    valid_cardios = [[list_cardios2[0][x] for x in poss],map(lambda x:tr[x],[list_cardios2[1][x] for x in poss])]
    
    from numpy import array
    Z = array(labs_ref[1])-array(valid_cardios[1])
    
    #tp,fp,fn
    return len(Z),sum(array(labs_ref[1])==0),sum(array(labs_ref[1])==1),sum((Z==0)*(array(labs_ref[1])==0)),sum((Z==0)*(array(labs_ref[1])==1)), sum(Z==-1), sum(Z==1)



def compareTipesList(struct_1, struct_2,groups = [['V'],['N','S'],['X']], max_error_margin = 10, points = False, len_sample = None):
    '''
    Any label in a list of the 'groups' will be considered as a same type group. So, if exist a group like ['A','B'] and two beats are labeled as A and B, they will be considered as labels of the same group
    '''
    results = {}
    for gr in groups:
        aux_1 = []
        aux_2 = []
        for i,x in enumerate(struct_2[1]):
            if x in gr:
                aux_2.append(struct_2[0][i])
        for i,x in enumerate(struct_1[1]):
            if x in gr:
                aux_1.append(struct_1[0][i])
        
        tp, fp, fn, aux_a, aux_b = compareLists(aux_1, aux_2, max_error_margin, points)
        if points:
            results[str(gr)] = [tp, fp, fn]
        else:
            if len_sample != None:
                results[str(gr)] = [tp, fp, fn,  get_sensitivity(tp, fn, fp), get_VPP(tp,fn,fp)]
            else:
                results[str(gr)] = [tp, fp, fn,  get_sensitivity(tp, fn, fp), get_VPP(len_sample,fn,fp)]

    return results





def mi_medida(tp,fn,fp):
    if tp+fn == 0:
        if fp == 0:
            return 100.
        else:
            return 0.
    return float(fp)/float(tp+fn)*100

def get_sensitivity(tp, fn, fp):
    if tp+fn == 0:
        return 100
    return float(tp)/float(tp+fn)*100

def get_specificity(tn, fn, fp):
    if tn != None:
        if tn+fn == 0:
            return 100
    return float(tn)/float(tn+fn)*100

def get_VPP(tp, fn, fp):
    if tp != None:
        if tp+fp == 0:
            return 100
    return float(tp)/float(tp+fp)*100

def qrs_position_not_details(label_ref, pos_ref, label_comp, pos_comp, details = False, error_permitted = 20 ):
    fp = 0
    fn = 0
    tp = 0
    
    accepted = ['V','N','S','I']
    
    hits = zeros(len(pos_comp))
    for i,beat in enumerate(label_ref):
        if beat in accepted:
            pos_beat = pos_ref[i]
            aux = (array(pos_comp) > (pos_beat-error_permitted) )*(array(pos_comp) < (pos_beat+error_permitted) )
            
            quant = 0
            for j,hit in enumerate(aux):
                if hit and label_comp[j] in accepted:
                    if quant == 0:
          
                        tp = tp+1
                    else:
                        fp = fp+1
            
                    
                    hits[j] = hits[j] + 1
                    quant = quant + 1
            
            if quant == 0:
                fn = fn + 1
    
    
    aux = (hits == 0)
    for i,n in enumerate(aux):
        if n:
            fp = fp+1

    
    if not details:                        
        return tp, fp, fn
    else:
        return tp,fp,fn


def qrs_position(label_ref, pos_ref, label_comp, pos_comp, details = False, error_permitted = 20 ):
    fp = 0
    fn = 0
    tp = 0
    
    fp_pos = []
    fn_pos = []
    tp_pos = []
    
    accepted = ['V','N','S','I']
    
    hits = zeros(len(pos_comp))
    for i,beat in enumerate(label_ref):
        if beat in accepted:
            pos_beat = pos_ref[i]
            aux = (array(pos_comp) > (pos_beat-error_permitted) )*(array(pos_comp) < (pos_beat+error_permitted) )
            
            quant = 0
            for j,hit in enumerate(aux):
                if hit and label_comp[j] in accepted:
                    if quant == 0:
                        tp_pos.append(pos_comp[j])
                        tp = tp+1
                    else:
                        fp = fp+1
                        fp_pos.append(pos_comp[j])        
                    
                    hits[j] = hits[j] + 1
                    quant = quant + 1
            
            if quant == 0:
                fn = fn + 1
                fn_pos.append(pos_ref[i])
    
    aux = (hits == 0)
    for i,n in enumerate(aux):
        if n:
            fp = fp+1
            fp_pos.append(pos_comp[i])
    
    if not details:                        
        return tp, fp, fn
    else:
        return tp,fp,fn,(fp_pos,['Y' for x in fp_pos]),(fn_pos,['Y' for x in fn_pos]), (tp_pos,['Y' for x in tp_pos])

def qrs_morphology(label_ref, pos_ref, label_comp, pos_comp, keys, error_permitted = 20):
    accepted = ['V','N','S','I']
    tp       = {'V':[],'N':[],'S':[],'I':[]}
    fn       = {'V':[],'N':[],'S':[],'I':[]}
    fp       = {'V':[],'N':[],'S':[],'I':[]}    
    
    for i,beat in enumerate(label_ref):
        if beat in accepted:
            pos_beat = pos_ref[i]
            aux = (array(pos_comp) > (pos_beat-error_permitted) )*(array(pos_comp) < (pos_beat+error_permitted) )
            index_list = []
            for j,hit in enumerate(aux):
                if hit and label_comp[j] in accepted:
                    index_list.append(j)

            hits = array([1 for ss in index_list if label_comp[ss] == beat]).sum()
            try:
                if hits == 0:
                    fn[beat].append(pos_beat)
                if hits >= 1:
                    tp[beat].append(pos_beat)
                if hits > 1:
                    fp[beat].append(pos_beat)
            except:
                print "perro!"

    for i,beat in enumerate(label_comp):
        if beat in accepted:
            pos_beat = pos_comp[i]
            aux = (array(pos_ref) > (pos_beat-error_permitted) )*(array(pos_ref) < (pos_beat+error_permitted) )
            index_list = []
            for j,hit in enumerate(aux):
                if hit and label_comp[j] in accepted:
                    index_list.append(j)

            misses = array([ss for ss in index_list if label_ref[ss] != beat])
            for miss_pos in misses:
                fp[label_comp[miss_pos]].append(pos_comp[miss_pos])    

    tp_ret = {}
    fp_ret = {}
    fn_ret = {}
    for key in keys:
        tp_ret.__setitem__(key, (tp[key],['Y' for x in tp[key]]))
        fp_ret.__setitem__(key, (fp[key],['Y' for x in fp[key]]))
        fn_ret.__setitem__(key, (fn[key],['Y' for x in fn[key]]))
    return tp_ret, fp_ret, fn_ret
