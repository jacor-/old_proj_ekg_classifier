from system.settings import *

from pylab import *

import cPickle
def getReqInfo22(filename):
    import cPickle
    f=open(filename,'rb')
    Z = cPickle.load(f)

    def getResults(base_filename, path):
        cases = h_io.get_usable_cases(REDIAGNOSE_DIAGNOSER)
        a = []
        b = []
        
        Xs_C1 = []
        Xs_C2 = []
        Xs_C3 = []
        Xs_Combined = []
        
        Ys = []
        indexs = [0]
        case_index = {}
        
        for num_case, case in enumerate(cases[:]):
            try:
                print case
                import cPickle
                #f = open('clasificacionPFC/'+case+'_lead1_4_components','rb')
                #Ys = Ys + list(cPickle.load(f))
                #f.close()
                
                
                f = open(path+case+'_lead1_4_'+base_filename,'rb')
                lead1 = cPickle.load(f)
                f.close()
                Xs_C1 = Xs_C1 + list(lead1)
            
                
                f = open(path+case+'_lead2_4_'+base_filename,'rb')
                lead2 = cPickle.load(f)
                f.close()
                Xs_C2 = Xs_C2 + list(lead2)
            
                
                f = open(path+case+'_lead3_4_'+base_filename,'rb')
                lead3 = cPickle.load(f)
                f.close()
                Xs_C3 = Xs_C3 + list(lead3)
                
                f = open(path+case+'_lead_all_4_components','rb')
                lead_tot = cPickle.load(f)
                f.close()
                Xs_Combined = Xs_Combined + list(lead_tot)
                
                f = open(path+case+'_valid_labs_4_components','rb')
                valid_labs = cPickle.load(f)
                f.close()
                tr = {'V':1,'N':0,'S':0}    
                Z = map(lambda x:tr[x],valid_labs[1])
                Ys = Ys + Z
                
                indexs.append(len(Ys))
                case_index[num_case] = case
                
            except:
                print "   error " + case
                pass
        return Xs_C1,Xs_C2,Xs_C3,Ys,indexs,case_index
    path = 'clasificacionPFC/'
    norm = False
    base_filename = 'components_no_norm'
    Xs_C1,Xs_C2,Xs_C3,Ys,indexs,case_index = getResults(base_filename, path)
    return Z,case_index,Ys,indexs

    


cases = h_io.get_usable_cases(REDIAGNOSE_DIAGNOSER)
filename_base = 'predict_no_no_norm_mohoeavious_reduced_covarNorm'
#Z3, case_index, Ys, indexs = getReqInfo22('resultatsEntrenament/'+filename_base+'_C3')
Z2, case_index, Ys, indexs = getReqInfo22('resultatsEntrenament/'+filename_base+'_C2')
#Z1, case_index, Ys, indexs = getReqInfo22('resultatsEntrenament/'+filename_base+'_C1')

import os
os.system("mkdir graphical_reports/"+filename_base)

def prepareReport(casoo,filename_base):
    from pylab import *
    case,labs = h_io.get_complete_exam(casoo,REDIAGNOSE_DIAGNOSER)
    import os
    os.system("mkdir graphical_reports/"+filename_base+"/"+casoo)
    n = [labs[0][i] for i,x in enumerate(labs[1]) if x in 'NS']
    v = [labs[0][i] for i,x in enumerate(labs[1]) if x == 'V']
    from signalCleaning.cleaners import *
    case[0] = hp(case[0],1.)
    case[1] = hp(case[1],1.)
    case[2] = hp(case[2],1.)

    refference_positions = [pos for i, pos in enumerate(labs[0]) if labs[1][i] in 'SVN']

    case[0] = normalizer(case[0],refference_positions,[])
    case[1] = normalizer(case[1],refference_positions,[])
    case[2] = normalizer(case[2],refference_positions,[])


    
    #fig = figure()
    #ax = fig.add_subplot(111)
    #ax.plot(case[0]/max(case[0]),'k')
    #ax.plot(case[1]/max(case[1])-2,'k')
    #ax.plot(case[2]/max(case[2])-4,'k')
    #aux = [annotate('V', xy=(x, 0.5), xytext=(x, 1.1),color='black') for x in v]
    #aux = [annotate('N', xy=(x, 0.5), xytext=(x, 0.7),color='black') for x in n]
    #
    #
    #for actua_zzz, zzz in enumerate(v):
    #    print str(actua_zzz) + " of " + str(len(v))
    #    ax.set_xlim(zzz-500,zzz+500)
    #    ax.set_ylim(-5,1.7)
    #    ax.figure.canvas.draw()
    #    savefig("graphical_reports/"+filename_base+"/"+casoo+'/ventricular_'+str(actua_zzz)+'.pdf',format='pdf')
    from numpy import array,max

    #fig = figure()
    #ax = fig.add_subplot(111)
    #ax.plot(case[0]/max(case[0]),'k')
    #ax.plot(case[1]/max(case[1])-2,'k')
    #ax.plot(case[2]/max(case[2])-4,'k')
    #aux = [annotate('V', xy=(x, 0.5), xytext=(x, 1.1),color='black') for x in v]
    #aux = [annotate('N', xy=(x, 0.5), xytext=(x, 0.7),color='black') for x in n]
    
    close(1)
    from numpy import max, mod
    for actua_zzz, zzz in enumerate(v):
        print str(actua_zzz) + " of " + str(len(v))
        if mod(actua_zzz,4) == 0:
            fig = figure(1)            

        ax = fig.add_subplot(2,2,1+mod(actua_zzz,4))
        ax.plot(case[0][max([zzz-500,0]):zzz+500]/max(case[0][max([zzz-500,0]):zzz+500]),'k')
        ax.plot(case[1][max([zzz-500,0]):zzz+500]/max(case[1][max([zzz-500,0]):zzz+500])-2,'k')
        ax.plot(case[2][max([zzz-500,0]):zzz+500]/max(case[2][max([zzz-500,0]):zzz+500])-4,'k')
        aux = [annotate('V', xy=(x-(max([zzz-500,0])), 0.5), xytext=(x-(max([zzz-500,0])), 1.1),color='black') for x in v if x > zzz-500 and x < zzz + 500]
        #aux = [annotate('N', xy=(x-(zzz-500), 0.5), xytext=(x-(zzz-500), 0.7),color='black') for x in n if x > zzz-500 and x < zzz + 500]

        #ax.set_xlim(zzz-500,zzz+500)
        ax.set_ylim(-5.5,1.7)
            

        if mod(actua_zzz,4) == 3:
            ax.figure.canvas.draw()
            savefig("graphical_reports/"+filename_base+"/"+casoo+'/ventricular_'+str(actua_zzz)+'.pdf',format='pdf')
            close(1)

def prepareErrorReport(casoo,filename_base):
    from pylab import *
    case,labs = h_io.get_complete_exam(casoo,REDIAGNOSE_DIAGNOSER)
    import os
    os.system("mkdir graphical_reports/"+filename_base+"/"+casoo)
    n = [labs[0][i] for i,x in enumerate(labs[1]) if x in 'NS']
    v = [labs[0][i] for i,x in enumerate(labs[1]) if x == 'V']
    from signalCleaning.cleaners import *
    case[0] = hp(case[0],1.)
    case[1] = hp(case[1],1.)
    case[2] = hp(case[2],1.)

    refference_positions = [pos for i, pos in enumerate(labs[0]) if labs[1][i] in 'SVN']

    case[0] = normalizer(case[0],refference_positions,[])
    case[1] = normalizer(case[1],refference_positions,[])
    case[2] = normalizer(case[2],refference_positions,[])


    
    #fig = figure()
    #ax = fig.add_subplot(111)
    #ax.plot(case[0]/max(case[0]),'k')
    #ax.plot(case[1]/max(case[1])-2,'k')
    #ax.plot(case[2]/max(case[2])-4,'k')
    #aux = [annotate('V', xy=(x, 0.5), xytext=(x, 1.1),color='black') for x in v]
    #aux = [annotate('N', xy=(x, 0.5), xytext=(x, 0.7),color='black') for x in n]
    #
    #
    #for actua_zzz, zzz in enumerate(v):
    #    print str(actua_zzz) + " of " + str(len(v))
    #    ax.set_xlim(zzz-500,zzz+500)
    #    ax.set_ylim(-5,1.7)
    #    ax.figure.canvas.draw()
    #    savefig("graphical_reports/"+filename_base+"/"+casoo+'/ventricular_'+str(actua_zzz)+'.pdf',format='pdf')
    from numpy import array,max

    #fig = figure()
    #ax = fig.add_subplot(111)
    #ax.plot(case[0]/max(case[0]),'k')
    #ax.plot(case[1]/max(case[1])-2,'k')
    #ax.plot(case[2]/max(case[2])-4,'k')
    #aux = [annotate('V', xy=(x, 0.5), xytext=(x, 1.1),color='black') for x in v]
    #aux = [annotate('N', xy=(x, 0.5), xytext=(x, 0.7),color='black') for x in n]
    
    ind = -1
    for i,x in enumerate(cases):
        if x == casoo:
            ind = i

    Z = Z2[indexs[ind]:indexs[ind+1]]
    
    os.system("mkdir graphical_reports/"+filename_base+"/"+casoo+"/errors")
    os.system("mkdir graphical_reports/"+filename_base+"/"+casoo+"/errors/FP")
    os.system("mkdir graphical_reports/"+filename_base+"/"+casoo+"/errors/FN")
    from numpy import array,zeros
    Y = zeros(len(n)+len(v))
    for i,x in enumerate([lab for lab in labs[1] if lab in 'SVN']):
        if x in 'V':
            Y[i]=1
    #return Z,Y,refference_positions,case
    errors = array(Z)-array(Y)

    fp = [pos for i,pos in enumerate(refference_positions) if errors[i] ==  1]
    fn = [pos for i,pos in enumerate(refference_positions) if errors[i] == -1]

    close(1)
    from numpy import max
    for actua_zzz, zzz in enumerate(fp):
        print str(actua_zzz) + " of " + str(len(fp))
        if mod(actua_zzz,4) == 0:
            fig = figure(1)            

        ax = fig.add_subplot(2,2,1+mod(actua_zzz,4))
        ax.plot(case[0][max([zzz-500,0]):zzz+500]/max(case[0][max([zzz-500,0]):zzz+500]),'k')
        ax.plot(case[1][max([zzz-500,0]):zzz+500]/max(case[1][max([zzz-500,0]):zzz+500])-2,'k')
        ax.plot(case[2][max([zzz-500,0]):zzz+500]/max(case[2][max([zzz-500,0]):zzz+500])-4,'k')
        aux = [annotate('V', xy=(x-(max([zzz,0])-500), 0.5), xytext=(x-(max([zzz-500,0])), 1.1),color='black') for x in v if x > zzz-500 and x < zzz + 500]
        #aux = [annotate('N', xy=(x-(zzz-500), 0.5), xytext=(x-(zzz-500), 0.7),color='black') for x in n if x > zzz-500 and x < zzz + 500]

        #ax.set_xlim(zzz-500,zzz+500)
        ax.set_ylim(-5.5,1.7)
            

        if mod(actua_zzz,4) == 3:
            ax.figure.canvas.draw()
            savefig("graphical_reports/"+filename_base+"/"+casoo+"/errors/FP/"+str(actua_zzz)+'.pdf',format='pdf')
            close(1)

    close(1)
    for actua_zzz, zzz in enumerate(fn):
        print str(actua_zzz) + " of " + str(len(fn))
        if mod(actua_zzz,4) == 0:
            fig = figure(1)            
        from numpy import max
        ax = fig.add_subplot(2,2,1+mod(actua_zzz,4))
        ax.plot(case[0][max([zzz-500,0]):zzz+500]/max(case[0][max([zzz-500,0]):zzz+500]),'k')
        ax.plot(case[1][max([zzz-500,0]):zzz+500]/max(case[1][max([zzz-500,0]):zzz+500])-2,'k')
        ax.plot(case[2][max([zzz-500,0]):zzz+500]/max(case[2][max([zzz-500,0]):zzz+500])-4,'k')
        aux = [annotate('V', xy=(x-(max([zzz-500,0])), 0.5), xytext=(x-(max([zzz-500,0])), 1.1),color='black') for x in v if x > zzz-500 and x < zzz + 500]
        #aux = [annotate('N', xy=(x-(zzz-500), 0.5), xytext=(x-(zzz-500), 0.7),color='black') for x in n if x > zzz-500 and x < zzz + 500]

        #ax.set_xlim(zzz-500,zzz+500)
        ax.set_ylim(-5.5,1.7)
            

        if mod(actua_zzz,4) == 3:
            ax.figure.canvas.draw()
            savefig("graphical_reports/"+filename_base+"/"+casoo+"/errors/FN/"+str(actua_zzz)+'.pdf',format='pdf')
            close(1)
            



def plotCosa(casoo):
    ssss = '107-00679'
    case,labs = h_io.get_complete_exam(casoo,REDIAGNOSE_DIAGNOSER)
    n = [labs[0][i] for i,x in enumerate(labs[1]) if x in 'NS']
    v = [labs[0][i] for i,x in enumerate(labs[1]) if x == 'V']
    from signalCleaning.cleaners import *
    case[0] = hp(case[0],1.)
    case[1] = hp(case[1],1.)
    case[2] = hp(case[2],1.)

    refference_positions = [pos for i, pos in enumerate(labs[0]) if labs[1][i] in 'SVN']

    case[0] = normalizer(case[0],refference_positions,[])
    case[1] = normalizer(case[1],refference_positions,[])
    case[2] = normalizer(case[2],refference_positions,[])

    plot(case[0]/max(case[0]),'g')
    plot(case[1]/max(case[1])-2,'b')
    plot(case[2]/max(case[2])-4,'y')

    ind = -1
    for i,x in enumerate(cases):
        if x == casoo:
            ind = i



    
    plot(refference_positions,array(Z1[indexs[ind]:indexs[ind+1]])-0.25,'og')
    plot(refference_positions,array(Z2[indexs[ind]:indexs[ind+1]])-0.30,'ob')
    plot(refference_positions,array(Z3[indexs[ind]:indexs[ind+1]])-0.35,'oy')

    #plot(refference_positions,Ys[indexs[ind]:indexs[ind+1]],'or')
    [annotate('V', xy=(x, 0.5), xytext=(x, 1),color='red') for x in v]
    #[annotate('N', xy=(x, 0.5), xytext=(x, 1),color='green') for x in n]

for xase in cases[:]:
    prepareErrorReport(xase,filename_base)
    prepareReport(xase,filename_base)
    
