
cardios_result = {}
res_lead = {}
res_zone = {}
res_joint = {}
for examen in range(100):
    if examen != 1:
        continue
    
    import cPickle
    from numpy import array
    try:
        f = open('cas_' +  str(examen), 'rb')
    except:
        continue
    
    sample_signal_1 = cPickle.load(f)[:100000]
    sample_signal_2 = cPickle.load(f)[:100000]
    sample_signal_3 = cPickle.load(f)[:100000]
    sample_diag = cPickle.load(f)
    cadios_diag = cPickle.load(f)
    f.close()
    
    import analyzer.sample_analyzer as an
    reload(an)
    aa = an.analyze(sample_signal_1, 0.5, 0, [], [[],[]], 55, lead_name = 0)
    bb = an.analyze(sample_signal_2, 0.5, 0, [], [[],[]], 55, lead_name = 1)
    cc = an.analyze(sample_signal_3, 0.5, 0, [], [[],[]], 55, lead_name = 2)
    
    

    colors = ['g','b','r','c','m','y']
    graph = ['s','x','*']
    rescate = 'y'
    from pylab import figure, plot
    figure()
    plot(sample_signal_1+5)
    plot(sample_signal_2)
    plot(sample_signal_3-5)
    from numpy import zeros
    beats = []
    
    
    resASaco = []
    resZone = []
    for index_lead, lead in enumerate([aa,bb,cc]):
        for i,zona in enumerate(lead):
            plot(zona[2],zeros(len(zona[2]))+zona[5],graph[index_lead]+colors[zona[5]])            
            #plot(zona[3][0],zeros(len(zona[3][0]))-1,graph[index_lead]+'y')
            plot([zona[0],zona[1]],[-1-index_lead,-1-index_lead],colors[zona[5]])
    
        import error.stats as st
        import error.statsZone as stZone
        import error.global_result as gr
        reload(st)
        reload(stZone)
        reload(gr)
    
        #ref_beats = [s for i,s in enumerate(sample_diag[0]) if sample_diag[1][i] == 'N' or sample_diag[1][i] == 'V' or sample_diag[1][i] == 'S']
        ref_beats = sample_diag
        resASaco.append(gr.globalResult(ref_beats, lead))
        resZone.append(stZone.compareZones(ref_beats,lead))
    
    res_lead[examen] = resASaco
    res_zone[examen] = resZone
    
    
    import leadJoin.sample_lead_joiner as lj
    import leadJoin.decisor.join_is_beat_decisor as decisor
    reload(decisor)
    reload(lj)
    res_complete,res_by_zone = lj.sample_lead_joiner(
            [aa,bb,cc], 6, decisor.weighted_decisor, weight_per_deep = [0,1,2,3,3,3,3,3,3,3,3,3,3,3,3])
    res_complete2,res_by_zone2 = lj.sample_lead_joiner(
            [aa,bb,cc], 3, decisor.just_one_decisor, weight_per_deep = [0,1,2,3,3,3,3,3,3,3,3,3,3,3,3])    
    
    res_joint[examen]= [st.compareLists(ref_beats, res_complete),st.compareLists(ref_beats, res_complete2)] 
    
    cardios_result[examen] =  st.compareLists(ref_beats,cadios_diag)
