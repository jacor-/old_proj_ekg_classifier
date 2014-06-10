
def percentile(perc, data_to_group, Z, GROUPS):
    from numpy import array, mean
    beats = [(array([data_to_group[y] for y,x in enumerate(Z) if x == i ])) for i in range(GROUPS)]
    aux = sorted(array([float(len(x)) for x in beats])/sum([len(x) for x in beats]),reverse = True)
    from numpy import isnan
    for i in range(GROUPS):
        if sum(aux[:i+1]) > 0.9:
            ret_aux = map(lambda x: mean(x,0), beats[:i+1])
            return [x for x in ret_aux if not isnan(x).any()]
    return []



import cPickle
data = []
labels = []

from system.settings import *

import beatWork.characterization.sample as charact
reload(charact)
import signalCleaning.sample as cleaner
reload(cleaner)

GROUPS = 10
from sklearn.mixture import GMM
agrupador = GMM()



def getWholeTrainData(cases, charact, cleaner, agrupador, GROUPS, leads):
    import time
    representative = {}
    label_group = {}
    for examen_index, examen in enumerate(cases):
       # if examen < 16:
       #     continue
        representative[examen] = {}
        label_group[examen] = {}
        
        
        
        t_ref = time.time()
        print "caracterizing " + str(examen_index+1) + " of " + str(len(cases))
        try:
            data,my_label_info1 = h_io.get_complete_exam(cases[examen_index], REDIAGNOSE_DIAGNOSER)
        except:
            continue    
        Ns = [my_label_info1[0][i] for i,x in enumerate(my_label_info1[1]) if x == 'N']
        Vs = [my_label_info1[0][i] for i,x in enumerate(my_label_info1[1]) if x == 'V']
        
        for lead_number in leads:
            representative[examen][lead_number] = []
            label_group[examen][lead_number] = []
            
            sample_signal_1, z = cleaner.sampleCleaner(data[lead_number], 1)
                
            import random
            N_seria,N_pos_seria = charact.sampleCharacterizator(sample_signal_1, Ns,[], normalize = False)
            V_seria,V_pos_seria = charact.sampleCharacterizator(sample_signal_1, Vs,[], normalize = False)
            
            data_complete = N_seria + V_seria
            pos_complete = N_pos_seria + V_pos_seria
            labels = map(lambda x: 'N',N_seria)+map(lambda x: 'V',V_seria)
            
            translator = {'N':0,'V':1,'X':2,'A':2,'Z':2}
            labels_translated = map(lambda x: translator[x], labels)
        
            train_data = data_complete
            data_test = data_complete
            labels_train = labels_translated
            labels_test = labels_translated
            
            
        
            if len(train_data) > 0:
                agrupador.fit(train_data)
                
                Z = agrupador.predict(N_seria)
                rep_N = percentile(0.9, N_seria, Z, GROUPS)
                if len(rep_N) > 0:
                    representative[examen][lead_number] = representative[examen][lead_number] + rep_N
                    label_group[examen][lead_number] = label_group[examen][lead_number] + ['N']*len(rep_N)
                
                if len(V_seria) > 0:
                    Z = agrupador.predict(V_seria)
                    rep_V = percentile(0.9, V_seria, Z, GROUPS)
                    if len(rep_V) > 0:
                        representative[examen][lead_number] = representative[examen][lead_number] + rep_V
                        label_group[examen][lead_number] = label_group[examen][lead_number] + ['V']*len(rep_V)
                print "he tardado " + str(time.time() - t_ref) + " ..."
        del data
    return representative, label_group 

def getWholeTrainData_AllVentricular_AllNormals(cases, charact, cleaner, agrupador, GROUPS, leads):
    import time
    representative = {}
    label_group = {}
    
    for examen_index, examen in enumerate(cases):
        representative[examen] = {}
        label_group[examen] = {}
        
        t_ref = time.time()
        print "caracterizing " + str(examen_index+1) + " of " + str(len(cases))
        try:
            data,my_label_info1 = h_io.get_complete_exam(cases[examen_index], REDIAGNOSE_DIAGNOSER)
        except:
            continue    
        Ns = [my_label_info1[0][i] for i,x in enumerate(my_label_info1[1]) if x == 'N']
        Vs = [my_label_info1[0][i] for i,x in enumerate(my_label_info1[1]) if x == 'V']
        
        for lead_number in leads:
            representative[examen][lead_number] = []
            label_group[examen][lead_number] = []
            
            sample_signal_1, z = cleaner.sampleCleaner(data[lead_number], 1)
                
            N_seria,N_pos_seria = charact(sample_signal_1, Ns,[], normalize = False)
            V_seria,V_pos_seria = charact(sample_signal_1, Vs,[], normalize = False)
            
            
            
            data_complete = N_seria + V_seria
            labels = map(lambda x: 'N',N_seria)+map(lambda x: 'V',V_seria)
            
            representative[examen][lead_number] = data_complete
            label_group[examen][lead_number] = labels
            
    return representative, label_group 

def prepareTrainDataOneLeadGroupsMean(case, data_info, leads_info, lead):
    '''
    Just 2 labels: Normal or Ventricular
    '''
    N_totals = []
    V_totals = []
    
    from mdp.utils import progressinfo
    for key in data_info:
        if key == case:
            continue
        elif lead in data_info[key]:
            #import ipdb
            #ipdb.set_trace()
            
            train_data = data_info[key][lead]
            train_labels = leads_info[key][lead]
            
            from sklearn import cluster
            from numpy import max,min,zeros,float,sum
            
            '''
            Normals
            '''
            N = [i for i,x in enumerate(train_labels) if x in 'NS']
            if len(N) > 0:
                from numpy import array

                cls = cluster.MiniBatchKMeans(k=50 ,init ='k-means++')
                
                normals = [train_data[x] for x in N]
                
                cls.fit(array(normals))
                cl_labels = cls.labels_
                hist = zeros(max(cl_labels)+1)
                for i in cl_labels:
                    hist[i] = hist[i] + 1
                hist = sorted(zip(map(lambda x: float(x)/len(normals),hist),range(len(hist))),key=lambda x: x[0],reverse = True)
                
                try:
                    max_gr = min([[i for i,x in enumerate(hist) if sum([z[0] for z in hist[:i]]) > 0.8][0],len(hist)])
                except:
                    import ipdb
                    ipdb.set_trace()    
                N_signals = [cls.cluster_centers_[hist[i][1]] for i in range(max_gr+1)]
                N_totals = N_totals + N_signals
            '''
            Ventriculars
            '''
            V = [i for i,x in enumerate(train_labels) if x in 'V']
            if len(V) > 0:
                from numpy import array
                V_signals = [train_data[i] for i in V]
                V_totals = V_totals + V_signals

    from numpy import array
    return array(N_totals+V_totals), ['N' for x in N_totals]+['V' for x in V_totals]

def prepareTrainDataOneLeadGroupsSelection(case, data_info, leads_info, lead):
    '''
    Just 2 labels: Normal or Ventricular
    '''
    N_totals = []
    V_totals = []
    
    quant_N = 150
    quant_V = 100
    
    from mdp.utils import progressinfo
    for key in data_info:
        if key == case:
            continue
        elif lead in data_info[key]:
            #import ipdb
            #ipdb.set_trace()
            
            train_data = data_info[key][lead]
            train_labels = leads_info[key][lead]
            
            from sklearn import cluster
            from numpy import max,min,zeros,float,sum
            
            
            def __make_groups(N, train_data, beats_to_get):
                if len(N) > 100:
                    from numpy import array,sum,min
                    cls = cluster.MiniBatchKMeans(k=50 ,init ='k-means++')
                    
                    normals = [train_data[x] for x in N]
                    
                    cls.fit(array(normals))
                    cl_labels = cls.labels_
                    hist = zeros(max(cl_labels)+1)
                    for i in cl_labels:
                        hist[i] = hist[i] + 1
                    hist = sorted(zip(map(lambda x: float(x)/len(normals),hist),range(len(hist))),key=lambda x: x[0],reverse = True)
                    
                    try:
                        max_gr = min([[i for i,x in enumerate(hist) if sum([z[0] for z in hist[:i]]) > 0.8][0],len(hist)])
                    except:
                        import ipdb
                        ipdb.set_trace()
                    
                    N_signals = []
                    for i in range(max_gr):
                        this_group = int(round(hist[i][0] *  beats_to_get))
                        
                        poss = [zz for zz,cosa in enumerate(cl_labels) if cosa ==hist[i][1]]
                        from numpy import min
                        N_signals = N_signals + [normals[poss[k]] for k in range(min([this_group, len(poss)]))]
                    return N_signals
                else:
                    return [train_data[x] for x in N]
            '''
            Normals
            '''
            N = [i for i,x in enumerate(train_labels) if x in 'NS']
            N_totals = N_totals + __make_groups(N, train_data, 1000)
            '''
            Ventriculars
            '''
            V = [i for i,x in enumerate(train_labels) if x in 'V']
            V_totals = V_totals + __make_groups(V, train_data, 150)

    from numpy import array
    return array(N_totals+V_totals), ['N' for x in N_totals]+['V' for x in V_totals]


def prepareTrainDataOneLead(case, data_info, leads_info, lead):
    '''
    Just 2 labels: Normal or Ventricular
    '''
    _totals = []
    _labels = []
    
    from mdp.utils import progressinfo
    for key in data_info:
        if key == case:
            continue
        elif lead in data_info[key]:
            #import ipdb
            #ipdb.set_trace()
            
            train_data = data_info[key][lead]
            train_labels = leads_info[key][lead]
            _totals = _totals + train_data
            _labels = _labels + train_labels  
            
    from numpy import array
    return _totals, _labels


def prepareTrainDataMultiLabelOneLead(case, data_info, leads_info, agrupador, lead):
    '''
    Just 2 labels: Multilabel! Agrupa y asigna una etiqueta a cada uno!
    '''
    train_labels = []
    train_data = []
    for key in data_info:
        if key == case:
            continue
        elif lead in data_info[key]:
            train_data = train_data + data_info[key][lead]
            train_labels = train_labels + leads_info[key][lead]
    from numpy import array
    
    agrupador.fit(train_data)
    
    
    
    
    test_def = []
    labs_test_def = []
    mapper = {}
    
    Normals = array([train_data[i] for i,x in enumerate(train_labels) if x == 'N' or x == 'S'])
    if len(Normals > 0):
        test_def = test_def + list(Normals)
        Z = agrupador.predict(Normals)
        labs_test_def = labs_test_def + list(Z)
        for x in set(Z):
            mapper[x] = 'N'
    
    Ventriculars = array([train_data[i] for i,x in enumerate(train_labels) if x == 'V'])            
    if len(Ventriculars) > 0:
        test_def = test_def + list(Ventriculars)
        Z = agrupador.predict(Ventriculars)
        Z = Z + max(labs_test_def) + 1
        labs_test_def = labs_test_def + list(Z)
        for x in set(Z):
            mapper[x] = 'V'

    return array(test_def), array(labs_test_def), mapper


def prepareTrainDataJointLeads(case, data_info, leads_info):
    train_labels = []
    train_data = []
    for key in data_info:
        if key == case:
            continue
        else:
            for lead in data_info[key]:
                train_data = train_data + data_info[key][lead]
                train_labels = train_labels + leads_info[key][lead]
    from numpy import array
    return array(train_data), train_labels

def getTestData(case, charact, cleaner, leads):
    import time
    t_ref = time.time()

    try:
        data,my_label_info1 = h_io.get_complete_exam(case, REDIAGNOSE_DIAGNOSER)
    except:
        return []    
    
    Ns = [my_label_info1[0][i] for i,x in enumerate(my_label_info1[1]) if x == 'N']
    Vs = [my_label_info1[0][i] for i,x in enumerate(my_label_info1[1]) if x == 'V']
    
    beats = []
    labels = []
    for lead_number in leads:
        sample_signal_1, z = cleaner.sampleCleaner(data[lead_number], 1)
            
        N_seria,N_pos_seria = charact(data[lead_number],  Ns,[], normalize = False)
        V_seria,V_pos_seria = charact(data[lead_number],  Vs,[], normalize = False)
        
        positions = Ns + Vs
        
        beats_complete = N_seria + V_seria
        labels_ = map(lambda x: 0,N_seria)+map(lambda x: 1,V_seria)
        
        #translator = {'N':0,'V':1,'X':2,'A':2,'Z':2}
        #labels_translated = map(lambda x: translator[x], labels)
        beats.append(beats_complete)
        labels.append(labels_)
    
    return beats, labels, positions


'''
Features extraction
'''

def sampleCharacterizator_JustCenter(signal, my_label_info, normalize = False):
    beat_position = my_label_info[0]
    from numpy import sqrt, dot,argmax,argmin,zeros
    beat_position = [x for x in beat_position if x -5 > 0 and x + 5 <= len(signal)]
    #print str(centroids)
    
    b_wave = []
    b_pos = []
    
    
    centroids = beat_position    
    felt_in_beggining = [my_label_info[1][1] for i,x in enumerate(beat_position) if centroids[i]-10 < 0]
    felt_in_end = [my_label_info[1][1] for i,x in enumerate(beat_position) if centroids[i]-10 >= 0]
    
    
    batecs_waveform = [signal[centroids[i]-10:centroids[i]+20] for i,x in enumerate(beat_position) if centroids[i]-10 >= 0 and centroids[i]+20 <= len(signal)]
    batecs_waveform = [signal[centroids[i]-10:centroids[i]+20] for i,x in enumerate(beat_position) if centroids[i]-10 >= 0 and centroids[i]+20 <= len(signal)]
    new_beat_position = [x for i,x in enumerate(beat_position) if centroids[i]-10 >= 0 and centroids[i]+20 <= len(signal)]
    if normalize == True:
        print "VALGO TRUE!!!"
        batecs_waveform = [x/sqrt(dot(x,x)) for x in batecs_waveform]
    else:
        batecs_waveform = [x.astype(float) for x in batecs_waveform]

    b_wave = b_wave + batecs_waveform
    b_pos = b_pos + new_beat_position

    from numpy import isnan
    for i,x in enumerate(b_wave):
        for j,y in enumerate(x):
            if isnan(y):
                b_wave[i][j] = 0

    Normals = [x for i,x in enumerate(b_wave) if my_label_info[1][i] in 'NS']
    Ventriculars = [x for i,x in enumerate(b_wave) if my_label_info[1][i] in 'V']
    for x in felt_in_beggining:
        if x in 'NS':
            Normals = [Normals[0]] + Normals
        if x in 'V':
            Ventriculars = [Ventriculars[0]] + Ventriculars
    for x in felt_in_end:
        if x in 'NS':
            Normals =  Normals + [Normals[-1]]
        if x in 'V':
            Ventriculars = Ventriculars + [Ventriculars[-1]] 
 
        
    return Ventriculars, Normals

def caracterize_beats(signal, label_info):
    def _noNan(aux):
        from numpy import isnan,isinf
        if isnan(aux): return 0
        elif isinf(aux): return 10
        else: return aux
        
    from numpy import zeros,convolve,median,diff,std
    fs = 100
    L1 = round(0.160*fs)
    L2 = (0.240*fs)
    
    L11 = 6
    L22 = 15
    
    interesantes = [(label_info[0][i],label_info[1][i])for i,x in enumerate(label_info[1]) if x == 'N' or x == 'S' or x == 'V']

    FF1 = zeros(len(interesantes));
    for i,x in enumerate(interesantes):

        segment = signal[interesantes[i][0]-L1:interesantes[i][0]+L2]
        
        s1 = convolve(segment,[1, 0, -1])
        s2 = convolve(s1,[1, 0, -1]);
        s1 = s1[5:35]
        s2 = s2[5:35]
        form_factor = std(s2)*std(segment)/std(s1)/std(s1);
        FF1[i] = _noNan(form_factor)
       
    FF2 = zeros(len(interesantes));    
    for i,x in enumerate(interesantes):

        segment = signal[interesantes[i][0]-L1:interesantes[i][0]+L2]
        
        s1 = convolve(segment,[1,1, 0, -1,-1])
        s2 = convolve(s1,[-1-1, 0, 1,1]);
        s1 = s1[5:35]
        s2 = s2[5:35]
        form_factor = std(s2)*std(segment)/std(s1)/std(s1);
        FF2[i] = _noNan(form_factor)
    
    
    FF3 = zeros(len(interesantes));
    for i,x in enumerate(interesantes):
        segment = signal[interesantes[i][0]-L11:interesantes[i][0]+L22]
        
        s1 = convolve(segment,[1, 0, -1])
        s2 = convolve(s1,[1, 0, -1]);
        s1 = s1[5:35]
        s2 = s2[5:35]
        form_factor = std(s2)*std(segment)/std(s1)/std(s1);
        FF3[i] = _noNan(form_factor)
    
    FF4 = zeros(len(interesantes));
    for i,x in enumerate(interesantes):

        segment = signal[interesantes[i][0]-L11:interesantes[i][0]+L22]
        
        s1 = convolve(segment,[1,1, 0, -1,-1])
        s2 = convolve(s1,[-1-1, 0, 1,1]);
        s1 = s1[5:35]
        s2 = s2[5:35]
        form_factor = std(s2)*std(segment)/std(s1)/std(s1);

        FF4[i] = _noNan(form_factor)
    
    FF5 = zeros(len(interesantes));        
    for i,x in enumerate(interesantes):
        segment = signal[interesantes[i][0]-L1:interesantes[i][0]+L2]
        from numpy import max, min,mean
        form_factor = mean(segment)-median(segment)
        FF5[i] = _noNan(form_factor)

    FF6 = zeros(len(interesantes));        
    for i,x in enumerate(interesantes):
        segment = signal[interesantes[i][0]-L11:interesantes[i][0]+L22]
        from numpy import max, min
        form_factor = std(segment)
        FF6[i] = _noNan(form_factor)

    FF7 = zeros(len(interesantes));        
    for i,x in enumerate(interesantes):
        segment = signal[interesantes[i][0]-L1:interesantes[i][0]+L22]
        s1 = convolve(segment,[1, -1])[5:35]
        from numpy import max, min
        form_factor = std(s1)
        FF7[i] = _noNan(form_factor)
    
    FF8 = zeros(len(interesantes));        
    for i,x in enumerate(interesantes):
        segment = signal[interesantes[i][0]-L1:interesantes[i][0]+L2]
        s1 = convolve(segment,[1,1,1,1,-1,-1-1,-1])[9:-9]
        from numpy import max, min
        form_factor = std(s1)
        FF8[i] = _noNan(form_factor)


    #RR = diff([x[0] for x in interesantes])
    #RR = RR/median(RR)
    #Vs = [(FF2[i],RR[i]-RR[i-1]) for i,x in enumerate(interesantes) if x[1] == 'V' if i < len(RR) and i > 0]
    #Ns = [(FF2[i],RR[i]-RR[i-1]) for i,x in enumerate(interesantes) if x[1] in 'NS' if i < len(RR) and i > 0]
    
    #Vs = [(FF2[i],FF1[i],FF3[i],FF4[i],FF5[i],FF6[i]) for i,x in enumerate(interesantes) if x[1] == 'V' if i < len(interesantes) and i > 0]
    #Ns = [(FF2[i],FF1[i],FF3[i],FF4[i],FF5[i],FF6[i]) for i,x in enumerate(interesantes) if x[1] in 'NS' if i < len(interesantes) and i > 0]
    
    Vs = [(FF1[i],FF2[i],FF3[i],FF4[i],FF5[i],FF6[i],FF7[i],FF8[i]) for i,x in enumerate(interesantes) if x[1] == 'V' if i < len(interesantes) and i > 0]
    Ns = [(FF1[i],FF2[i],FF3[i],FF4[i],FF5[i],FF6[i],FF7[i],FF8[i]) for i,x in enumerate(interesantes) if x[1] in 'NS' if i < len(interesantes) and i > 0]
    
    #Vs = [(FF2[i],FF1[i]) for i,x in enumerate(interesantes) if x[1] == 'V' if i < len(RR) and i > 0]
    #Ns = [(FF2[i],FF1[i]) for i,x in enumerate(interesantes) if x[1] in 'NS' if i < len(RR) and i > 0]
    return Vs,Ns



def FischerScore(n_components, train_data, train_label,test_beats):
    import dimensionality_reductor.reductors.my_methods.fischerScore as fischerScore
    reload(fischerScore)
    fs = fischerScore.fischerScore(n_components)
    fs.fit(train_data,  train_label, True)
    train_data = fs.transform(train_data)
    test_beats = array(fs.transform(test_beats[0]))
    return train_data, test_beats

def PCA(n_components, train_data, test_beats):
    import mdp 
    pca = mdp.nodes.PCANode(output_dim=n_components)
    pca.train(train_data)
    pca.stop_training()
    train_data = pca.execute(train_data)
    test_beats = array(pca.execute(test_beats[0]))
    return train_data, test_beats



def LDA(n_components, train_data, train_label, test_beats):
    from sklearn.lda import LDA
    lda = LDA(n_components)
    lda.fit(train_data, train_label)
    train_data = lda.transform(train_data)
    test_beats = array(lda.transform(test_beats[0]))
    return train_data, test_beats

def QDA(n_components, train_data, train_label, test_beats):
    from sklearn.qda import QDA
    lda = QDA(n_components)
    lda.fit(train_data, train_label)
    train_data = lda.transform(train_data)
    test_beats = array(lda.transform(test_beats[0]))
    return train_data, test_beats


def ICA(n_param, train_data, test_beats):
    import mdp 
    ica = mdp.nodes.FastICANode()
    ica.train(train_data)
    ica.stop_training()
    train_data = ica.execute(train_data)
    test_beats = array(ica.execute(test_beats[0]))
    return train_data, test_beats

'''
Classifiers
'''

def _clf_executor(predict, test_beats, lead):
    Z = []
    j = 0
    from numpy import array
    

    if len(test_beats[lead]) == 0:
        return array([])
    try: 
        for i in range(len(test_beats[lead])/10):
            try:
                Z = Z + list(predict(test_beats[lead][range(i*10,(i+1)*10)]))
                #print str(i) + " of " + str(len(test_beats[0])/10)
            except:
                break
            j = j + 1
    except:
        import ipdb
        ipdb.set_trace()
    if j*10 != len(test_beats[0])-1:
        aux = test_beats[0][j*10:]
        if not len(aux) == 0:
            Z = Z + list(predict(array(aux)))
    return Z

def clf_LDA(n_components, train_data, train_labels, test_data, lead = 0):
    from scikits.learn.lda import LDA
    lda = LDA(n_components)
    lda.fit(train_data, train_labels)
    return _clf_executor(lda.predict,test_data, lead)

def clf_SVM(train_data, train_labels, test_data, lead = 0):
    from sklearn.svm import SVC
    #import ipdb
    clf = SVC()
    clf.fit(train_data, train_labels)
    return _clf_executor(clf.predict,test_data,lead)

def clf_NearestNeighbors(n_neighbors, train_data, train_labels, test_data, lead = 0):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors)
    clf.fit(train_data, train_labels)
    return _clf_executor(clf.predict,test_data, lead)

def clf_NN(train_data, train_labels, test_data, lead = 0):
    '''
    Tal como se explica en  "A Practical Guide to Support Vector Classification":Chih-Wei Hsu, Chih-Chung Chang, and Chih-Jen Lin, Department of Computer Science 2010
    Hay que escalar los datos. Ello se hace en el train y en el test.
    '''
    print "Entrenando con " + str(len(train_data)) + " casos"
    
    from numpy import abs, max,array,min
    
    alpha = max(array(zip(max(abs(train_data),1), max(abs(test_data[lead]),1))),1)
    train_data_norm = [[x/alpha[i] for i,x in enumerate(y)] for y in train_data]
    test_data_norm = [[x/alpha[i] for i,x in enumerate(y)] for y in test_data[lead]]
    
    '''
    Se monta la red neuronal
    '''
    from pybrain.datasets import SupervisedDataSet
    from pybrain.structure import LinearLayer, SigmoidLayer, GaussianLayer
    from pybrain.structure import FeedForwardNetwork
    from pybrain.structure import FullConnection
    
    from pybrain.datasets import ClassificationDataSet
    from pybrain.datasets.unsupervised import UnsupervisedDataSet
    
    cs = UnsupervisedDataSet(len(test_data[lead][0]))
    for i,x in enumerate(test_data_norm):
        cs.addSample(test_data_norm[i])
    
    ds = SupervisedDataSet(len(test_data[lead][0]),1)
    for i,x in enumerate(train_data_norm):
        ds.addSample(tuple(train_data_norm[i]), (train_labels[i],))
    
    n = FeedForwardNetwork()
    
    inLayer = LinearLayer(len(test_data[lead][0]))
    hiddenLayer = GaussianLayer(6)
    #hiddenLayer = LinearLayer(1)
    outLayer = LinearLayer(1)
    
    n.addInputModule(inLayer)
    n.addModule(hiddenLayer)
    n.addOutputModule(outLayer)
    
    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)
    
    n.addConnection(in_to_hidden)
    n.addConnection(hidden_to_out)
        
    n.sortModules()
    
    
    from pybrain.supervised.trainers import BackpropTrainer
    trainer = BackpropTrainer(n, ds,momentum=0.1, verbose=True, weightdecay=0.01)
    trainer.trainUntilConvergence(maxEpochs = 50, continueEpochs = 50, validationProportion = 0.05)
    
    out = n.activateOnDataset(cs)
    out = out.argmax(axis=1)  # the highest output activation gives the class

    return out
'''
Clustering
'''

def clusterNN(n_groups, data):
    from sklearn import cluster

    cls = cluster.MiniBatchKMeans(k= n_groups,init ='k-means++')
    cls.fit(array(data))
    
    cl_labels = cls.labels_
    return cl_labels
       
'''
Joiners
'''
    
def joinerInternalCluster_Voting(data, label_groups, n_neighbors,ref_data, ref_labels):
    from numpy import zeros, array, sum
    Z = clf_NearestNeighbors(n_neighbors, ref_data, ref_labels, data)
    n_groups = max(label_groups)
    res_group = zeros(n_groups+1)
    for i in range(n_groups+1):
        act_gr = array([x for zz,x in enumerate(Z) if label_groups[zz] == i])
        L = len(act_gr)
        Vs = sum(act_gr)
        if Vs >= L/2:
            res_group[i] = 1

    for i in range(len(Z)):
        Z[i] = res_group[label_groups[i]]
    return Z
    
            
        
    
    
    


'''
train_data_aux, train_labels = prepareTrainDataOneLead(cases[2], beats_repr, labels_group,0)
train_data = array(train_data_aux)
del train_data_aux

test_beats_aux, test_labels, beat_position = getTestData(cases[2], charact.sampleCharacterizator_JustCenter, cleaner, [0])
test_beats = array(test_beats_aux)
del test_beats_aux
'''


def getResult(test_labels, train_labels, translator =  lambda x:{'N':0,'S':0,'V':1}[x], inverse_translator = lambda x: x):
    resultat = test_labels
    resultat = map(inverse_translator,resultat)
    differences = [0,0]
    hits = [0,0]
    points = [[],[]]
    for i in range(len(resultat)):
        if resultat[i] + train_labels[i] == 0:
            hits[0] = hits[0] + 1
        elif resultat[i] + train_labels[i] == 2:
            hits[1] = hits[1] + 1
        if resultat[i] - train_labels[i] == 1:
            differences[0] = differences[0] + 1
            points[0].append(i)
        elif resultat[i] - train_labels[i] == -1:
            differences[1] = differences[1] + 1
            points[1].append(i)
    
    from numpy import divide
    statss = [divide(float(hits[0]),hits[0]+differences[0]),divide(float(hits[1]),(hits[1]+differences[1]))]
    return [hits, differences, statss, points]


    
def test7(indexs, train_data,train_labels, test_labels, test_beats):    
    import numpy as np
    
    tr_labels = map(lambda x:{'N':0,'V':1,'S':0}[x],train_labels)
    print "starting PCA"
    
    for output_dim in [15,10,5]:
        try:
            train_data, test_beats = PCA(output_dim,train_data,test_beats)
            break
        except:
            continue

    from numpy import sum
    
    print "   starting Nearest Neighbors"
    Z = clf_NearestNeighbors(10, train_data, tr_labels, test_beats)
    print str(sum(Z))
    res2 = getResult(Z, test_labels[0])

    Z2 = clf_NearestNeighbors(20, train_data, tr_labels, test_beats)
    print str(sum(Z2))
    res3 = getResult(Z2, test_labels[0])

    
    return [res3]


def test8(indexs, train_data,train_labels, test_labels, test_beats):    
    import numpy as np
    
    tr_labels = map(lambda x:{'N':0,'V':1,'S':0}[x],train_labels)
    
    
    from numpy import sum
    from scipy.cluster.vq import whiten
    print "starting PCA"
    train_data, test_beats = PCA(15,train_data,test_beats)
    print "starting FischerScore"
    train_data, test_beats = FischerScore(3,train_data,tr_labels,test_beats)
    print "starting clasificacion"
    Z = clf_NN(0, train_data, tr_labels, test_beats)
    res3 = getResult(Z, test_labels[0])    
    return [res3]

    


from numpy import array,shape

def plotResult(case,res):
    from pylab import figure, plot, show
    data,my_label_info1 = h_io.get_complete_exam(case, REDIAGNOSE_DIAGNOSER)    
    figure(); plot(data[0])
    plot(my_label_info1[0],[data[0][x]+1 for x in my_label_info1[0]],'oy')
    plot([my_label_info1[0][x] for x in res[3][0]],[data[0][my_label_info1[0][x]] for x in res[3][0]],'og')
    plot([my_label_info1[0][x] for x in res[3][1]],[data[0][my_label_info1[0][x]] for x in res[3][1]],'or')
    

    
def netejaRara(case,param = 1):
    from pylab import figure, plot, show
    data,my_label_info1 = h_io.get_complete_exam(case, REDIAGNOSE_DIAGNOSER)    
    figure()

    import signalCleaning.sample as cl
    from numpy import ones, convolve,abs,median
    pond1 = abs(cl.sampleCleaner(data[0],param)[0])
    pulse = ones(12)/12
    pond = convolve(pond1, pulse)[6:-5]
    pond = abs(pond / max(pond[10:-10]))
    
    n = data[0]*pond
    #plot(data[0])
    #plot(n)
    return n
    














'''
Data Getters from DB
'''

def getDataRaraAllLeads(cases, beat_characterizator):
    beats = []
    labels = []
    
    from mdp.utils import progressinfo
    index = [0]
    for case in progressinfo(cases):
        try:
            data,my_label_info1 = h_io.get_complete_exam(case, REDIAGNOSE_DIAGNOSER)
            import signalCleaning.sample as cl
            n1 = cl.sampleCleaner(data[0],1)[0]
            n2 = cl.sampleCleaner(data[1],1)[0]
            n3 = cl.sampleCleaner(data[2],1)[0]
        except:
            index.append(len(labels))
            continue
        
        if len(beat_characterizator['params'].keys()) != 0:
            Vs1,Ns1 = beat_characterizator['algorithm'](n1,my_label_info1, beat_characterizator['params'])
            Vs2,Ns2 = beat_characterizator['algorithm'](n2,my_label_info1, beat_characterizator['params'])
            Vs3,Ns3 = beat_characterizator['algorithm'](n3,my_label_info1, beat_characterizator['params'])
        else:
            Vs1,Ns1 = beat_characterizator['algorithm'](n1,my_label_info1)
            Vs2,Ns2 = beat_characterizator['algorithm'](n2,my_label_info1)
            Vs3,Ns3 = beat_characterizator['algorithm'](n3,my_label_info1)
            
    
        
            
        for mm,x in enumerate(Vs1):
            beats = beats + [Vs1[mm],Vs2[mm],Vs3[mm]]
        for mm,x in enumerate(Ns1):
            beats = beats + [Ns1[mm],Ns2[mm],Ns3[mm]]
        
        labels = labels + ['V']*len(Vs1)*3+['N']*len(Ns1)*3
        index.append(len(labels))
    return beats, labels, index
    

def getDataRara(cases, lead, beat_characterizator):
    beats = []
    labels = []
    
    from mdp.utils import progressinfo
    index = [0]
    for case in progressinfo(cases):
        try:
            data,my_label_info1 = h_io.get_complete_exam(case, REDIAGNOSE_DIAGNOSER)
            import signalCleaning.sample as cl
            n = cl.sampleCleaner(data[lead],1)[0]
            data[lead] = n
        except:
            index.append(len(labels))
            continue
        
        if len(beat_characterizator['params'].keys()) != 0:
            Vs1,Ns1 = beat_characterizator['algorithm'](data[lead],my_label_info1, normalize = beat_characterizator['params']['normalize'])
        else:
            Vs1,Ns1 = beat_characterizator['algorithm'](data[lead],my_label_info1)
        beats = beats + Vs1 + Ns1
        labels = labels + ['V']*len(Vs1)+['N']*len(Ns1)
        index.append(len(labels))
    return beats, labels, index

def getSubData(beats, labels, index, Normals, Ventriculars):
    b = []
    l = []
    i = [0]
    for j in range(len(index)-1):

        f_index = [index[j]+z for z,x in enumerate(labels[index[j]:index[j+1]]) if x == 'V']
        if len(f_index) > 0:
            b = b + list(beats[f_index[0]:min([f_index[0]+Ventriculars,index[j+1],f_index[-1]+1])])
            l = l + list(labels[f_index[0]:min([f_index[0]+Ventriculars,index[j+1],f_index[-1]+1])])
            
        
        N_index = [index[j]+z for z,x in enumerate(labels[index[j]:index[j+1]]) if x == 'N']
        if len(N_index) > 0:
            b = b + list(beats[N_index[0]:min([N_index[0]+Normals,index[j+1],N_index[-1]+1])])
            l = l + list(labels[N_index[0]:min([N_index[0]+Normals,index[j+1],N_index[-1]+1])])
        
        i = i + [len(b)]
    return b, l, i
        
    
        
    



def ale_y_calcula(ttrain,tlabels, index, cas_index,cases, my_classifier_func, filename = '/dev/null',extra_arg = None, QUANT_N = 50, QUANT_V = 100, dimensionality_reduction = []):
    #from pylab import figure, plot, show
    q = len(index)-1
    res = {}
    f = open(filename,'w')
    #info resumed!
    ttrain_, tlabels_, index_ = getSubData(ttrain, tlabels, index, QUANT_N, QUANT_V)
    for actual in cas_index:
        ttest = array(ttrain[index[actual]:index[actual+1]])
        ltest = tlabels[index[actual]:index[actual+1]]
        #
        #Uncomment these lines if you want to resume the test data!
        #
        #Not resumed
        #ttrain2 = list(ttrain[:index[actual]])+list(ttrain[index[actual+1]:])
        #tlabels2 = list(tlabels[:index[actual]])+list(tlabels[index[actual+1]:])
        #
        #resumed
        

        ttrain2 = array(list(ttrain_[:index_[actual]])+list(ttrain_[index_[actual+1]:]))
        tlabels2 = list(tlabels_[:index_[actual]])+list(tlabels_[index_[actual+1]:])
        
        tr_test = map(lambda x:{'N':0,'V':1,'S':0}[x],ltest)
        tr_labels = map(lambda x:{'N':0,'V':1,'S':0}[x],tlabels2)

        if len(ttrain2) > 2 and len(ttest) > 2:
            for reductor in dimensionality_reduction:
                if reductor['req_labels'] == True:
                    ttrain2,ttest = reductor['algorithm'](reductor['param'], ttrain2, tr_labels,[ttest])
                else:
                    ttrain2,ttest = reductor['algorithm'](reductor['param'], ttrain2, [ttest])
    
            if extra_arg != None:
                Z = my_classifier_func(extra_arg,array(ttrain2),tr_labels,array([ttest]), 0)
            else:
                Z = my_classifier_func(array(ttrain2),tr_labels,array([ttest]), 0)
        else:
            Z = []
        #return ttest, ltest, tr_test, Z
        res[cases[actual]] = getResult(Z, tr_test)
        #res[cases[actual]] = res[cases[actual]] + [list(Z)] 
        f.write(cases[actual] + "    " + str(res[cases[actual]][0])+ "     " + str(res[cases[actual]][1]) + "\n")
        f.write(str(res) + "\n")
        del Z
    f.close()
    return res

def ale_y_calcula_por_grupos(ttrain,tlabels, index, cas_index,cases, gr_number = 10, filename = '/dev/null',extra_arg = 3, dimensionality_reduction = None):
    #from pylab import figure, plot, show
    q = len(index)-1
    res = {}
    f = open(filename,'w')
    for actual in cas_index:
        ttest  = ttrain[index[actual]:index[actual+1]]
        ltest = tlabels[index[actual]:index[actual+1]]
        
        
        ttrain2 = list(ttrain[:index[actual]])+list(ttrain[index[actual+1]:])
        tlabels2 = list(tlabels[:index[actual]])+list(tlabels[index[actual+1]:])
    
        tr_test = map(lambda x:{'N':0,'V':1,'S':0}[x],ltest)
        tr_labels = map(lambda x:{'N':0,'V':1,'S':0}[x],tlabels2)




        if extra_arg != None:
            label_groups = clusterNN(gr_number,ttrain2)
            Z = joinerInternalCluster_Voting([ttest], label_groups, extra_arg,ttrain2, tr_labels)
        else:
            pass
        res[cases[actual]] = getResult(Z, tr_test)
        f.write(cases[actual] + "    " + str(res[cases[actual]][0])+ "     " + str(res[cases[actual]][1]) + "\n")
        f.write(str(res) + "\n")
    f.close()
    return res

def launchLotsOfTests(ttrain,tlabels,ind,cases,test_indexs, dimensionality_reduction):
    '''
    dimensionality_reduction es una lista. Cada posicion de la lista tiene un diccionario {algorithm: , param: , req_labels: True}
    '''
    res = {}
    print "tests SVM    1"
    res['SVM-balanced-50'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_SVM, QUANT_N = 50, QUANT_V = 50, dimensionality_reduction = dimensionality_reduction)
    print "tests SVM    2"
    res['SVM-balanced-100'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_SVM, QUANT_N = 100, QUANT_V = 100, dimensionality_reduction = dimensionality_reduction)
    print "tests SVM    3"
    res['SVM-balanced-200'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_SVM, QUANT_N = 200, QUANT_V = 200, dimensionality_reduction = dimensionality_reduction)
    print "tests SVM    4"
    res['SVM-unbalanced-100_200'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_SVM, QUANT_N = 100, QUANT_V = 200, dimensionality_reduction = dimensionality_reduction)
    print "tests SVM    5"
    res['SVM-unbalanced-50_100'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_SVM, QUANT_N = 50, QUANT_V = 100, dimensionality_reduction = dimensionality_reduction)
    
    print "tests 5NN    1"
    res['5NN-balanced-50'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_NearestNeighbors,extra_arg = 5, QUANT_N = 50, QUANT_V = 50, dimensionality_reduction = dimensionality_reduction)
    print "tests 5NN    2"
    res['5NN-balanced-100'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_NearestNeighbors,extra_arg = 5, QUANT_N = 100, QUANT_V = 100, dimensionality_reduction = dimensionality_reduction)
    print "tests 5NN    3"
    res['5NN-balanced-200'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_NearestNeighbors,extra_arg = 5, QUANT_N = 200, QUANT_V = 200,dimensionality_reduction = dimensionality_reduction)
    print "tests 5NN    4"
    res['5NN-unbalanced-100_200'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_NearestNeighbors,extra_arg = 5, QUANT_N = 100, QUANT_V = 200, dimensionality_reduction = dimensionality_reduction)
    print "tests 5NN    5"
    res['5NN-unbalanced-50_100'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_NearestNeighbors,extra_arg = 5, QUANT_N = 50, QUANT_V = 100, dimensionality_reduction = dimensionality_reduction)

    print "tests 3NN    1"
    res['3NN-balanced-50'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_NearestNeighbors,extra_arg = 7, QUANT_N = 50, QUANT_V = 50, dimensionality_reduction = dimensionality_reduction)
    print "tests 3NN    2"
    res['3NN-balanced-100'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_NearestNeighbors,extra_arg = 7, QUANT_N = 100, QUANT_V = 100, dimensionality_reduction = dimensionality_reduction)
    print "tests 3NN    3"
    res['3NN-balanced-200'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_NearestNeighbors,extra_arg = 7, QUANT_N = 200, QUANT_V = 200, dimensionality_reduction = dimensionality_reduction)
    print "tests 3NN    4"
    res['3NN-unbalanced-100_200'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_NearestNeighbors,extra_arg = 7, QUANT_N = 100, QUANT_V = 200,dimensionality_reduction = dimensionality_reduction )
    print "tests 3NN    5"
    res['3NN-unbalanced-50_100'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_NearestNeighbors,extra_arg = 7, QUANT_N = 50, QUANT_V = 100, dimensionality_reduction = dimensionality_reduction)

    print "tests 3NN    1"
    res['3NN-balanced-50'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_NearestNeighbors,extra_arg = 3, QUANT_N = 50, QUANT_V = 50, dimensionality_reduction = dimensionality_reduction)
    print "tests 3NN    2"
    res['3NN-balanced-100'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_NearestNeighbors,extra_arg = 3, QUANT_N = 100, QUANT_V = 100, dimensionality_reduction = dimensionality_reduction)
    print "tests 3NN    3"
    res['3NN-balanced-200'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_NearestNeighbors,extra_arg = 3, QUANT_N = 200, QUANT_V = 200, dimensionality_reduction = dimensionality_reduction)
    print "tests 3NN    4"
    res['3NN-unbalanced-100_200'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_NearestNeighbors,extra_arg = 3, QUANT_N = 100, QUANT_V = 200, dimensionality_reduction = dimensionality_reduction)
    print "tests 3NN    5"
    res['3NN-unbalanced-50_100'] = ale_y_calcula(ttrain,tlabels, ind,  test_indexs,cases, clf_NearestNeighbors,extra_arg = 3, QUANT_N = 50, QUANT_V = 100, dimensionality_reduction = dimensionality_reduction)
    
    return res


def generalTest(cases, beats_char, dimensionality_reduction):

    print "Lead 1"
    ttrain,tlabels,ind = getDataRara(cases[:], 0, beats_char)
    resC1 = launchLotsOfTests(ttrain,tlabels,ind,cases,range(len(cases)),dimensionality_reduction)
    del ttrain,tlabels,ind

    print "Lead 2"    
    ttrain,tlabels,ind = getDataRara(cases[:], 1, beats_char)
    resC2 = launchLotsOfTests(ttrain,tlabels,ind,cases,range(len(cases)), dimensionality_reduction)
    del ttrain,tlabels,ind

    print "Lead 3"    
    ttrain,tlabels,ind = getDataRara(cases[:], 2, beats_char)
    resC3 = launchLotsOfTests(ttrain,tlabels,ind,cases,range(len(cases)), dimensionality_reduction)
    del ttrain,tlabels,ind

    print "Lead All"    
    ttrain,tlabels,ind = getDataRaraAllLeads(cases[:], beats_char)
    resTot = launchLotsOfTests(ttrain,tlabels,ind,cases,range(len(cases)), dimensionality_reduction)
    del ttrain,tlabels,ind
    
    return [resC1,resC2,resC3,resTot]



def testCompleteRareCharacterization(cases):
    import cPickle
    f = open('testCompleteRareCharacterization','wb')
    ob = generalTest(cases, {'algorithm':caracterize_beats,'params':{}}, [])
    cPickle.dump(ob, f)
    f.close()
    return ob

def testCompleteWholeBeat(cases):
    import cPickle
    f = open('testCompleteWholeBeat-PCA14-FS8','wb')
    ob = generalTest(cases, {'algorithm':sampleCharacterizator_JustCenter, 'params':{"normalize": False}}, [{'algorithm':PCA, 'param':14,'req_labels':False},{'algorithm':FischerScore, 'param':8,'req_labels':True}])
    cPickle.dump(ob, f)
    f.close()
    return ob

def testCompleteWholeBeat2(cases):
    import cPickle
    f = open('testCompleteWholeBeat-PCA8','wb')
    ob = generalTest(cases, {'algorithm':sampleCharacterizator_JustCenter, 'params':{"normalize": False}}, [{'algorithm':PCA, 'param':8,'req_labels':False}])
    cPickle.dump(ob, f)
    f.close()
    return ob

def testCompleteWholeBeat3(cases):
    import cPickle
    f = open('testCompleteWholeBeat-FS8','wb')
    ob = generalTest(cases, {'algorithm':sampleCharacterizator_JustCenter, 'params':{"normalize": False}}, [{'algorithm':FischerScore, 'param':8,'req_labels':True}])
    cPickle.dump(ob, f)
    f.close()
    return ob
    
def testCompleteWholeBeat4(cases):
    import cPickle
    f = open('testCompleteWholeBeat-ICA-FS8','wb')
    ob = generalTest(cases, {'algorithm':sampleCharacterizator_JustCenter, 'params':{"normalize": False}}, [{'algorithm':ICA, 'param':14,'req_labels':False},{'algorithm':FischerScore, 'param':8,'req_labels':True}])
    cPickle.dump(ob, f)
    f.close()
    return ob

def testCompleteWholeBeat6(cases):
    import cPickle
    f = open('testCompleteWholeBeat-LDA14-FS8','wb')
    ob = generalTest(cases, {'algorithm':sampleCharacterizator_JustCenter, 'params':{"normalize": False}}, [{'algorithm':LDA, 'param':14,'req_labels':True},{'algorithm':FischerScore, 'param':8,'req_labels':True}])
    cPickle.dump(ob, f)
    f.close()
    return ob


def startCompleteTest():
    cases = h_io.get_usable_cases('cardiosManager')
    #testCompleteRareCharacterization(cases)
    testCompleteWholeBeat(cases)
    testCompleteWholeBeat2(cases)
    testCompleteWholeBeat3(cases)
    testCompleteWholeBeat4(cases)
    testCompleteWholeBeat6(cases)


def __main__():
    cases = h_io.get_usable_cases('cardiosManager')
    leads = [0]
    
    #---------------------------------
    #--------- DIRECT MORPH ----------
    #---------------------------------
    
    #beats_repr, labels_group = getWholeTrainData_AllVentricular_AllNormals(cases, charact.sampleCharacterizator_JustCenter, cleaner, agrupador, GROUPS, leads)
    '''
    f = open('entrenament_a_saco_tothom','rb')
    beats_repr = cPickle.load(f)
    labels_group = cPickle.load(f)
    f.close()
    
    
    from numpy import arange
    
    res = {}
    for i,case in enumerate(cases):
        if i != -1:
            continue
    
        res[case] = []
        print "Case " + str(i+1) + " of " + str(len(cases))
        
        try:
            test_beats_aux, test_labels, beat_position = getTestData(case, charact.sampleCharacterizator_JustCenter, cleaner, [0])
            test_beats = array(test_beats_aux)
            del test_beats_aux
            if len(test_beats[0]) <10:
                continue
        except:
            continue
           
    
        
        
        train_data_aux, train_labels = prepareTrainDataOneLeadGroupsSelection(case, beats_repr, labels_group,0)
        train_data = array(train_data_aux)
        del train_data_aux
    
        nou = test7(arange(5,6), array(list(train_data)+list(train_data)),list(train_labels+train_labels), test_labels, test_beats)
        [nou[resultadillo].append(beat_position) for resultadillo in range(len(nou))]
        res[case] = nou
        
        #del test_beats
        #del train_data
    '''
        
    #---------------------------------
    #--------- CHARACT BEAT ----------
    #---------------------------------
    '''
    Res = testChangeMorphologies(cases)
    '''
    #---------------------------------
    #----------- OLD STUFF -----------
    #---------------------------------
    '''
    ttrain,tlabels,ind = getDataRara(cases)
    tt,lt,ll = getDataRara([cases[2]])
    #plotColoraines2(ttrain,tlabels, ind, each = True)
    tr_test = map(lambda x:{'N':0,'V':1,'S':0}[x],lt)
    tr_labels = map(lambda x:{'N':0,'V':1,'S':0}[x],tlabels)
    
    
    #Z = clf_SVM(array(ttrain),tr_labels,array([tt]))
    Z = clf_NearestNeighbors(3,array(ttrain),tr_labels,array([tt]))
    res3 = getResult(Z, tr_test)
    '''
