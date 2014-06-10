




def characterizeAFul(signal, beat_position):
    FF1 = caracterize_beats_MIT(signal, beat_position)
    FF2 = caracterize_beats_Jose(signal, beat_position)
    FF3 = caracterize_beats_MIT2(signal, beat_position)
    FF4 = caracterize_beats_Jose2(signal, beat_position)    
    #FF5 = caracterize_beats_Altura(signal, beat_position)
    #FF6 = caracterize_beats_Altura2(signal, beat_position)
    #FF7 = caracterize_beats_Wavelet1(signal, beat_position)
    #FF8 = caracterize_beats_Wavelet4(signal, beat_position)
    #FF9 = caracterize_amplitude_mean(signal, beat_position)
    FF10 = caracterize_RR(signal, beat_position)
    #return [[FF1[i],FF2[i],FF3[i],FF4[i],FF5[i],FF6[i],FF7[i],FF8[i],FF9[i],FF10[i]] for i in range(len(FF1))]





def caracterize_beats_MIT(signal, beat_position,fs = 100):
    def _noNan(aux):
        from numpy import isnan,isinf
        if isnan(aux): return 0
        elif isinf(aux): return 10
        else: return aux
        
    from numpy import zeros,convolve,median,diff,std

    L1 = round(0.160*fs)
    L2 = round(0.240*fs)
        
    interesantes = beat_position

    FF1 = zeros(len(interesantes));
    for i,x in enumerate(interesantes):

        segment = signal[interesantes[i]-L1:interesantes[i]+L2]
        
        s1 = convolve(segment,[1, 0, -1])
        s2 = convolve(s1,[1, 0, -1]);
        s1 = s1[5:-6]
        s2 = s2[5:-6]
        form_factor = std(s2)*std(segment)/std(s1)/std(s1);
        FF1[i] = _noNan(form_factor)
    return FF1

def caracterize_beats_Jose(signal, beat_position):
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
    
    interesantes = beat_position
       
    FF2 = zeros(len(interesantes));    
    for i,x in enumerate(interesantes):

        segment = signal[interesantes[i]-L1:interesantes[i]+L2]
        
        s1 = convolve(segment,[1,1, 0, -1,-1])
        s2 = convolve(s1,[-1-1, 0, 1,1]);
        s1 = s1[5:35]
        s2 = s2[5:35]
        form_factor = std(s2)*std(segment)/std(s1)/std(s1);
        FF2[i] = _noNan(form_factor)
    return FF2

def caracterize_beats_MIT2(signal, beat_position):
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
    
    interesantes = beat_position
    
    
    FF3 = zeros(len(interesantes));
    for i,x in enumerate(interesantes):
        segment = signal[interesantes[i]-L11:interesantes[i]+L22]
        
        s1 = convolve(segment,[1, 0, -1])
        s2 = convolve(s1,[1, 0, -1]);
        s1 = s1[5:35]
        s2 = s2[5:35]
        form_factor = std(s2)*std(segment)/std(s1)/std(s1);
        FF3[i] = _noNan(form_factor)
    return FF3
    

def caracterize_beats_Jose2(signal, beat_position):    
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
    
    interesantes = beat_position
    
    FF4 = zeros(len(interesantes));
    for i,x in enumerate(interesantes):

        segment = signal[interesantes[i]-L11:interesantes[i]+L22]
        
        s1 = convolve(segment,[1,1, 0, -1,-1])
        s2 = convolve(s1,[-1-1, 0, 1,1]);
        s1 = s1[5:35]
        s2 = s2[5:35]
        form_factor = std(s2)*std(segment)/std(s1)/std(s1);

        FF4[i] = _noNan(form_factor)
    return FF4




def caracterize_beats_Altura(signal, beat_position):
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
    
    interesantes = beat_position
    

    FF5 = zeros(len(interesantes));        
    for i,x in enumerate(interesantes):
        segment = signal[interesantes[i]-L1:interesantes[i]+L2]
        from numpy import max, min,mean
        form_factor = mean(segment)-median(segment)
        FF5[i] = _noNan(form_factor)
    return FF5

def caracterize_beats_Altura2(signal, beat_position):
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
    
    interesantes = beat_position
    

    FF6 = zeros(len(interesantes));        
    for i,x in enumerate(interesantes):
        segment = signal[interesantes[i]-L11:interesantes[i]+L22]
        from numpy import max, min
        form_factor = std(segment)
        FF6[i] = _noNan(form_factor)
    return FF6

def caracterize_beats_Wavelet1(signal, beat_position):
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
    
    interesantes = beat_position
    

    FF7 = zeros(len(interesantes));        
    for i,x in enumerate(interesantes):
        segment = signal[interesantes[i]-L1:interesantes[i]+L22]
        s1 = convolve(segment,[1, -1])[5:35]
        from numpy import max, min
        form_factor = std(s1)
        FF7[i] = _noNan(form_factor)
    return FF7
    
def caracterize_beats_Wavelet4(signal, beat_position):
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
    
    interesantes = beat_position
    

    FF8 = zeros(len(interesantes));        
    for i,x in enumerate(interesantes):
        segment = signal[interesantes[i]-L1:interesantes[i]+L2]
        s1 = convolve(segment,[1,1,1,1,-1,-1-1,-1])[9:-9]
        from numpy import max, min
        form_factor = std(s1)
        FF8[i] = _noNan(form_factor)
    return FF8

def caracterize_amplitude_mean(signal, beat_position):
    def _noNan(aux):
        from numpy import isnan,isinf
        if isnan(aux): return 0
        elif isinf(aux): return 10
        else: return aux
        
    from numpy import zeros,convolve,median,diff,std,max,abs,mean
    fs = 100
    L1 = round(0.160*fs)
    L2 = (0.240*fs)
    
    L11 = 6
    L22 = 15
    
    interesantes = beat_position
    

    FF8 = zeros(len(interesantes));        
    for i,x in enumerate(interesantes):
        max_value = max(abs(signal[interesantes[i]-L1:interesantes[i]+L2]))
        mean_value = mean(signal[interesantes[i]-L1:interesantes[i]+L2])
        FF8[i] = _noNan(max_value-mean)
    return FF8

def caracterize_RR(signal, beat_position):
    def _noNan(aux):
        from numpy import isnan,isinf
        if isnan(aux): return 0
        elif isinf(aux): return 10
        else: return aux
        
    from numpy import zeros,convolve,median,diff,std,max,abs,mean,array
    fs = 100
    L1 = round(0.160*fs)
    L2 = (0.240*fs)
    
    L11 = 10
    L22 = 10
    
    interesantes = beat_position
    

    FF8 = zeros(len(interesantes));        
    for i,x in enumerate(interesantes):
        #rr_estimat = median(diff(array(beat_position[interesantes[i-L11]:interesantes[i+L22]])))
        if i > 0 and i < len(interesantes)-1:
            FF8[i] = float((beat_position[i+1]-beat_position[i]))/(beat_position[i]-beat_position[i-1])
        elif i == 0:
            FF8[i] = float((beat_position[i+1]-beat_position[i]))/(beat_position[i+2]-beat_position[i+1])
        else:
            FF8[i] = float((beat_position[i]-beat_position[i-1]))/(beat_position[i-1]-beat_position[i-2])
    return FF8

  
  