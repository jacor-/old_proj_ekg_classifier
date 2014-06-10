import struct
from numpy import array
import settings
from os import listdir
import subprocess
#path_mdb = '/home/jose/.gvfs/temp en srvgm001/Jose/Sint/'

def _get_signal_filename(examId):

    path_mdb = settings.MDB_PATH + 'CardioMg.mdb'
    f = open(path_mdb, 'r')

    linea = ["mdb-export",path_mdb,"Signals", "-H"]
    linea2 = ["mdb-export",path_mdb,"Exams", "-H"]


    f2 = open("aux1.txt", 'w')
    f3 = open("aux2.txt", 'w')

    subprocess.call(linea, stdout=f2)
    subprocess.call(linea2, stdout=f3)

    f.close()
    f2.close()
    f3.close()

    f2 = open("aux1.txt", 'r')
    f3 = open("aux2.txt", 'r')

    signalId = -1
    for l in f3.readlines():
        l2 = l.split(',')
        if(l2[0][1:-1] == examId):
            signalId = l2[8]
            break
    if(signalId == -1):
        raise("La has liado! Ningun examen con esa ID")

    filename_signal = ""
    for l in f2.readlines():
        l2 = l.split(',')
        if(l2[0] == signalId):
            filename_signal = l2[10][1:-1]
            break
    if(filename_signal == ""):
        raise("La has liado ningun senal con esa ID")

    subprocess.call(["rm", "aux1.txt"])
    subprocess.call(["rm", "aux2.txt"])

    return filename_signal

def _get_part_of_signal(exam_id, first_sample = 0, last_sample = 0):
    import os
    filename = settings.FULL_SIGNAL_PATH + _get_signal_filename(exam_id)+'.Hfd'
    
    f1 = open(settings.RESUMED_SIGNAL_PATH + exam_id+'_1.pfc_data','wb')
    f2 = open(settings.RESUMED_SIGNAL_PATH + exam_id+'_2.pfc_data','wb')
    f3 = open(settings.RESUMED_SIGNAL_PATH + exam_id+'_3.pfc_data','wb')
    
    f = open(filename, 'rb')
    aux = f.read(1)

    count = 0;
    while(count < first_sample):
        f.read(3)
        count = count + 1

    while aux != '' and count < last_sample:
        f1.write(aux)
        f2.write(f.read(1))
        f3.write(f.read(1))
        #c1.append(ord(aux)-128)
        #c2.append(ord(f.read(1))-128)
        #c3.append(ord(f.read(1))-128)
        aux = f.read(1)
        count = count+1
    
    f1.close()
    f2.close()
    f3.close()
    f.close() 

def _read_full_labels(exam_id, diagnoser = 'cardiosManager'):
    f = open(settings.FULL_DIAGNOSE_PATH + diagnoser + '/'+exam_id +'.txt','r')
    lab = []
    points = []
    for x in f.readlines():
        aux = x.split(',')
        lab.append(aux[0])
        points.append(int(aux[1])/100)
    return points,lab

def get_positional_refference(exam_id):
    points, lab = _read_full_labels(exam_id, 'jose_diagnose_24:02:2012')
    #seq = ['I','I','N','I','I','N','I','I','A','A','A','A']
    seq = ['I', 'I', 'N', 'I', 'I', 'N', 'I', 'I', 'A', 'A', 'A', 'A']
    k = len(seq)
    it = (lab[i:i+k] for i in range(len(lab)-k))
    
    fids = []
    new_points = []
    new_labels = []
    for i in range(len(lab)-k):    
        aux = it.next()
        if aux == seq:
            if len(fids) == 0:
                fids.append(i)
            else:
                fids.append(i)
    if(len(fids)!=2):
        return -1,-1,-1,-1

    index_inici = fids[0]+k
    index_final = fids[1]
    first_sample = points[index_inici]-50
    last_sample = points[index_final]+50
    return index_inici, index_final, first_sample,last_sample

def prepare_labels(labeler,exam_id, index_inici, index_final, first_sample, last_sample):
    points, lab = _read_full_labels(exam_id, labeler)    
    new_labels = array(lab[index_inici:index_final])
    new_points = array(points[index_inici:index_final])-first_sample
    f = open(settings.RESUMED_DIAGNOSES_PATH+labeler+'/'+exam_id+'.txt', 'w')
    for i in range(len(new_labels)):
        f.write(str(new_labels[i]) + ','+str(new_points[i])+'\n')
    f.close()

def prepare_signal(exam_id, first_sample, last_sample):
    _get_part_of_signal(exam_id, first_sample, last_sample)

def read_labels(exam_id, diagnoser = 'cardiosManager'):
    f = open(settings.RESUMED_DIAGNOSES_PATH + diagnoser + '/'+exam_id +'.txt','r')
    lab = []
    points = []
    for x in f.readlines():
        aux = x.split(',')
        lab.append(aux[0])
        points.append(int(aux[1]))
    return (points,lab)


def read_data(id_exam):
    data = [[],[],[]]
    f1 = open(settings.RESUMED_SIGNAL_PATH + id_exam+'_1.pfc_data','rb')
    f2 = open(settings.RESUMED_SIGNAL_PATH + id_exam+'_2.pfc_data','rb')
    f3 = open(settings.RESUMED_SIGNAL_PATH + id_exam+'_3.pfc_data','rb')
    
    aux = f1.read(1)
    while aux != '':
        data[0].append(ord(aux)-128)
        data[1].append(ord(f2.read(1))-128)
        data[2].append(ord(f3.read(1))-128)
        
        #c1.append(ord(aux)-128)
        #c2.append(ord(f.read(1))-128)
        #c3.append(ord(f.read(1))-128)
        aux = f1.read(1)
    
    f1.close()
    f2.close()
    f3.close()
    return array(data)

def get_complete_exam(id_exam, diagnoser):
    points, labels = read_labels(id_exam, diagnoser)
    from numpy import array,zeros,dot,argmax,abs,mean
    
    from old_src.signalCleaning.cleaners import lowpass
    from old_src.signalCleaning import cleaners
    
    
    
    data = array(read_data(id_exam)).astype('float')    
    return data, (points, labels)

def get_complete_exam_filtrado(id_exam, diagnoser):
    points, labels = read_labels(id_exam, diagnoser)
    
    
    
    import cPickle
    f = open('noves_dates/noves_dates'+id_exam,'rb')
    data = cPickle.load(f)
    f.close()    
    return data, (points, labels)


def import_signals():
    error = []
    usable = []
    for i in listdir(settings.FULL_DIAGNOSE_PATH+'jose_diagnose_24:02:2012/'):
        id_exam = i[:-4]   
        print id_exam
        index_inici, index_final, first_sample, last_sample = get_positional_refference(id_exam)
        print "  ... Finding local zone"
        if(index_inici == -1):
            error.append(id_exam)
            print "  error"
            continue
        else:
            usable.append(id_exam)
        print "  ... Preparing signal"
        prepare_signal(id_exam, first_sample, last_sample)
    f = open(settings.INTERNAL_INFO+"error:signals", 'w')
    f.write(str(error))
    f.close()
    f = open(settings.INTERNAL_INFO+"usable:signals", 'w')
    f.write(str(usable))
    f.close()


'''
def import_diagnoses(new_diagnoser, path_mdb = ''):
'''
'''
    If you are interested in use your own diagnoses with our system, you just have to call this function providing the path where you have stored all you .mdb files and the name with this name is going to be known by.
    If your mdb files have already been imported, you just have to omit the path_mdb parameter.
    
    Usage examples:
        
        import_diagnoses('cardiosManager')
        import_diagnoses('jose_diagnose_24:02:2012')
        
''' 
'''
    if(path_mdb != ''):
        import old_src.mdb_diagnoser_exporter as mdb_diagnoser_exporter
        mdb_diagnoser_exporter.generaNouDiagnosticador(path_mdb, new_diagnoser)
    error = []
    usable = []
    for i in listdir(FULL_DIAGNOSE_PATH+new_diagnoser):
        id_exam = i[:-4]   
        print id_exam
        index_inici, index_final, first_sample, last_sample = get_positional_refference(id_exam)
        print "  ... Found local zone"
        if(index_inici == -1):
            error.append(id_exam)
            print "  error"
            continue
        else:
            usable.append(id_exam)  
        prepare_labels(new_diagnoser,i[:-4], index_inici, index_final, first_sample, last_sample)
        print "  ... Labels " + new_diagnoser + " ready"
    f = open(INTERNAL_INFO+"error:"+new_diagnoser, 'w')
    f.write(str(error))
    f.close()
    f = open(INTERNAL_INFO+"usable:"+new_diagnoser, 'w')
    f.write(str(usable))
    f.close()
'''
def get_usable_cases(diagnoser):
    f = open(settings.INTERNAL_INFO+"usable_"+diagnoser, 'r')
    usables = eval(f.readline())
    f.close()
    f = open(settings.INTERNAL_INFO+"usable_signals", 'r')
    signals = eval(f.readline())
    f.close()
    return [x for x in usables if x in signals]


#import_diagnoses('cardiosManager')
#import_diagnoses('jose_diagnose_24:02:2012')
#import_signals()
#z = get_usable_cases('cardiosManager')