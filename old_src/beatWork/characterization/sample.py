
'''

These modules have to characterize beats using any kind of techniques.

INPUT:

   - signal:         signal where beats has been detected
   - beat_position:  positions in signal where beats have been detected

OUTPUT:

   - beats characterization:  array like structure with as many rows as beats_positions input parameter
   
'''
from beatWork.characterization.waveform_characterizers import autocorrelation

def normalizeCharacterizator(signal, beat_position,beat_information):
    beats, pos = sampleCharacterizator(signal, beat_position,beat_information, normalize = False)
    from numpy import sqrt, dot
    return map(lambda x: x/sqrt(dot(x,x)),beats),pos

def sampleCharacterizator(signal, beat_position,beat_information, normalize = True):
    '''
    from numpy import sqrt, dot,argmax,argmin
    centroids = [argmax(signal[x-5:x+5])+x-5 for x in beat_position if x -5 > 0 and x + 5 <= len(signal)]
    #print str(centroids)
    batecs_waveform = [signal[centroids[i]-10:centroids[i]+20] for i,x in enumerate(beat_position) if centroids[i]-10 >= 0 and centroids[i]+20 <= len(signal)]
    new_beat_position = [centroids[i] for i,x in enumerate(beat_position) if centroids[i]-10 >= 0 and centroids[i]+20 <= len(signal)]
    if normalize:
        batecs_waveform = [x/sqrt(dot(x,x)) for x in batecs_waveform]

    return batecs_waveform, new_beat_position
    '''

    from numpy import sqrt, dot,argmax,argmin
    centroids_max = [argmax(signal[x-5:x+5])+x-5 for x in beat_position if x -5 > 0 and x + 5 <= len(signal)]
    centroids_min = [argmin(signal[x-5:x+5])+x-5 for x in beat_position if x -5 > 0 and x + 5 <= len(signal)]
    beat_position = [x for x in beat_position if x -5 > 0 and x + 5 <= len(signal)]
    #print str(centroids)
    
    b_wave = []
    b_pos = []
    
    
    for ttt in [centroids_max,centroids_min]:
        centroids = ttt
        batecs_waveform = [signal[centroids[i]-10:centroids[i]+20] for i,x in enumerate(beat_position) if centroids[i]-10 >= 0 and centroids[i]+20 <= len(signal)]
        new_beat_position = [x for i,x in enumerate(beat_position) if centroids[i]-10 >= 0 and centroids[i]+20 <= len(signal)]
        if normalize:
            batecs_waveform = [x/sqrt(dot(x,x)) for x in batecs_waveform]
        b_wave = b_wave + batecs_waveform
        b_pos = b_pos + new_beat_position

    return b_wave, b_pos




def sampleCharacterizator_JustCenter(signal, beat_position,beat_information, normalize = True):

    from numpy import sqrt, dot,argmax,argmin
    beat_position = [x for x in beat_position if x -5 > 0 and x + 5 <= len(signal)]
    #print str(centroids)
    
    b_wave = []
    b_pos = []
    
    
    centroids = beat_position
    batecs_waveform = [signal[centroids[i]-10:centroids[i]+20] for i,x in enumerate(beat_position) if centroids[i]-10 >= 0 and centroids[i]+20 <= len(signal)]
    new_beat_position = [x for i,x in enumerate(beat_position) if centroids[i]-10 >= 0 and centroids[i]+20 <= len(signal)]
    if normalize:
        batecs_waveform = [x/sqrt(dot(x,x)) for x in batecs_waveform]
    b_wave = b_wave + batecs_waveform
    b_pos = b_pos + new_beat_position

    return b_wave, b_pos




def characterizeAFulCorr(signal, beat_position):
    from estocastic_characterizers import *
    FF1 = caracterize_beats_MIT(signal, beat_position)
    FF2 = caracterize_beats_Jose(signal, beat_position)
    FF3 = caracterize_beats_MIT2(signal, beat_position)
    FF4 = caracterize_beats_Jose2(signal, beat_position)    
    FF5 = caracterize_beats_Altura(signal, beat_position)
    FF6 = caracterize_beats_Altura2(signal, beat_position)
    FF7 = caracterize_beats_Wavelet1(signal, beat_position)
    FF8 = caracterize_beats_Wavelet4(signal, beat_position)
    
    from waveform_characterizers import *
    FF9 = autocorrelation(signal, beat_position)
    
    from numpy import array
    return [[FF1[i],FF2[i],FF3[i],FF4[i],FF5[i],FF6[i],FF7[i],FF8[i]]+FF9[i] for i in range(len(FF1))]
