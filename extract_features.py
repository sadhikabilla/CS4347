import librosa
import numpy as np
import scipy
import scipy.io.wavfile
import scipy.signal
import scipy.stats

# Define our own MFCC feature extractor (as per Assignment 3)
def myMFCC(floatdata, samplerate, nfft):
    nfilt = 26
    low_freq = 0
    high_freq = (1127 * np.log(1 + (samplerate/2)/700))
    
    # Space out equally the Mel points
    mel_points = np.linspace(low_freq, high_freq, nfilt + 2)
    # Convert Mel points back to frequency (Hz)
    hz_points = (700 * (np.exp(mel_points/1127) - 1))
    bin = nfft * hz_points/samplerate
    
    fbank = np.zeros((nfilt, int(np.floor(nfft/2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(np.floor(bin[m-1]))   # left
        f_m = int(np.round(bin[m]))           # center
        f_m_plus = int(np.ceil(bin[m+1]))     # right
    
        for k in range(f_m_minus, f_m):
            fbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    
    bufferstart = []
    for i in range(len(floatdata)):
        if i*512+1024 >= len(floatdata):
            break;
        bufferstart.append(i*512);
    
    buffermfcc = np.zeros((nfilt, len(bufferstart)))
    
    for i in range(len(bufferstart)):
        bufferdata = floatdata[bufferstart[i]:bufferstart[i]+1024]
        emp_bufferdata = np.append(bufferdata[0], bufferdata[1:]-0.95*np.asarray(bufferdata[:-1]))
        window = scipy.signal.hamming(len(emp_bufferdata))
        emp_bufferdata = emp_bufferdata * window
        
        bufferfft = scipy.fftpack.fft(emp_bufferdata)
        bufferfft = bufferfft[0:(int)(len(bufferfft)/2)+1]        
        bufferfft = np.abs(bufferfft)
        
        if all([x == 0 for x in bufferfft]):
            bufferfft_nonzero = [0.000001]*len(bufferfft)
        else:
            bufferfft_nonzero = bufferfft
        
        mel_vector = []
        for j in range(len(fbank)):
            mel_vector.append(np.log10(np.dot(bufferfft_nonzero, fbank[j])))         
        mfcc = scipy.fftpack.dct(mel_vector)
        
        for j in range(len(fbank)):
            buffermfcc[j,i] = mfcc[j]
        
    return buffermfcc

in_file = 'testing_files.txt'
out_file = 'features_test.csv'
TESTING = True # False if running on training dataset

fin = open(in_file, "r")
fout = open(out_file, "w")

frame_size = 1024
hop_size = 512

file_cnt = 1
line = fin.readline()
input = []
while line:
    
    print(file_cnt)
    features = []
    
    input = line.split()
    filename = input[0]
    signal, sample_rate = librosa.load(filename)
    duration = librosa.get_duration(signal, sample_rate)    

    mfcc = myMFCC(signal, sample_rate, frame_size)
    for i in range(len(mfcc)):
        features.append(np.mean(mfcc[i]))
    for i in range(len(mfcc)):
        features.append(np.std(mfcc[i]))
    for i in range(len(mfcc)):
        features.append(scipy.stats.skew(mfcc[i]))
    for i in range(len(mfcc)):
        features.append(scipy.stats.kurtosis(mfcc[i]))
    
    mfcc_delta = librosa.feature.delta(mfcc)
    for i in range(len(mfcc_delta)):
        features.append(np.mean(mfcc_delta[i]))
    for i in range(len(mfcc_delta)):
        features.append(np.std(mfcc_delta[i]))
    for i in range(len(mfcc_delta)):
        features.append(scipy.stats.skew(mfcc_delta[i]))
    for i in range(len(mfcc_delta)):
        features.append(scipy.stats.kurtosis(mfcc_delta[i]))
    
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    for i in range(len(mfcc_delta2)):
        features.append(np.mean(mfcc_delta2[i]))
    for i in range(len(mfcc_delta2)):
        features.append(np.std(mfcc_delta2[i]))
    for i in range(len(mfcc_delta2)):
        features.append(scipy.stats.skew(mfcc_delta2[i]))
    for i in range(len(mfcc_delta2)):
        features.append(scipy.stats.kurtosis(mfcc_delta2[i]))
    
    tempo, beats = librosa.beat.beat_track(y=signal, sr=sample_rate)
    features.append(tempo)
    note_onset = librosa.onset.onset_detect(y=signal, sr=sample_rate)
    note_density = len(note_onset)/duration
    features.append(note_density)
    
    zcr = librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_size, hop_length=hop_size) 
    rms = librosa.feature.rmse(y=signal, frame_length=frame_size, hop_length=hop_size)
    
    features.append(np.mean(zcr))
    features.append(np.std(zcr))
    features.append(scipy.stats.skew(zcr.T))   
    features.append(scipy.stats.kurtosis(zcr.T)) 
    
    features.append(np.mean(rms))
    features.append(np.std(rms))    
    features.append(scipy.stats.skew(rms.T))
    features.append(scipy.stats.kurtosis(rms.T))
    
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
    spectral_flatness = librosa.feature.spectral_flatness(y=signal, n_fft=frame_size, hop_length=hop_size)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)

    features.append(np.mean(spectral_centroid))
    features.append(np.std(spectral_centroid))   
    features.append(scipy.stats.skew(spectral_centroid.T))
    features.append(scipy.stats.kurtosis(spectral_centroid.T))
    
    features.append(np.mean(spectral_bandwidth))
    features.append(np.std(spectral_bandwidth))
    features.append(scipy.stats.skew(spectral_bandwidth.T))
    features.append(scipy.stats.kurtosis(spectral_bandwidth.T))
    
    features.append(np.mean(spectral_flatness))
    features.append(np.std(spectral_flatness))
    features.append(scipy.stats.skew(spectral_flatness.T))
    features.append(scipy.stats.kurtosis(spectral_flatness.T))
    
    features.append(np.mean(spectral_rolloff))        
    features.append(np.std(spectral_rolloff)) 
    features.append(scipy.stats.skew(spectral_rolloff.T))
    features.append(scipy.stats.kurtosis(spectral_rolloff.T))

    
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
    for i in range(len(spectral_contrast)):
        features.append(np.mean(spectral_contrast[i]))
    for i in range(len(spectral_contrast)):
        features.append(np.std(spectral_contrast[i]))
    for i in range(len(spectral_contrast)):
        features.append(scipy.stats.skew(spectral_contrast[i]))
    for i in range(len(spectral_contrast)):
        features.append(scipy.stats.kurtosis(spectral_contrast[i]))
        
    contrast_delta = librosa.feature.delta(spectral_contrast)  
    for i in range(len(contrast_delta)):
        features.append(np.mean(contrast_delta[i]))
    for i in range(len(contrast_delta)):
        features.append(np.std(contrast_delta[i]))
    for i in range(len(contrast_delta)):
        features.append(scipy.stats.skew(contrast_delta[i]))
    for i in range(len(contrast_delta)):
        features.append(scipy.stats.kurtosis(contrast_delta[i]))
        
    contrast_delta2 = librosa.feature.delta(spectral_contrast, order=2)
    for i in range(len(contrast_delta2)):
        features.append(np.mean(contrast_delta2[i]))
    for i in range(len(contrast_delta2)):
        features.append(np.std(contrast_delta2[i]))
    for i in range(len(contrast_delta2)):
        features.append(scipy.stats.skew(contrast_delta2[i]))
    for i in range(len(contrast_delta2)):
        features.append(scipy.stats.kurtosis(contrast_delta2[i]))
    
    S = np.abs(librosa.stft(y=signal, n_fft=frame_size, hop_length=hop_size))
    chroma = librosa.feature.chroma_stft(S=S, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
    for i in range(len(chroma)):
        features.append(np.mean(chroma[i]))
    for i in range(len(chroma)):
        features.append(np.std(chroma[i]))
    for i in range(len(chroma)):
        features.append(scipy.stats.skew(chroma[i]))
    for i in range(len(chroma)):
        features.append(scipy.stats.skew(chroma[i]))
        
    ##  FEATURES
    #   26 MFCC mean, 26 MFCC std, 26 MFCC skew, 26 MFCC kurtosis
    #   26 D-MFCC mean, 26 D-MFCC std, 26 D-MFCC skew, 26 D-MFCC kurtosis
    #   26 DD-MFCC mean, 26 DD-MFCC std, 26 DD-MFCC skew, 26 DD-MFCC kurtosis
    #   tempo, note onset
    #   ZCR mean, ZCR std, ZCR skew, ZCR kurt, RMS mean, RMS std, RMS skew, RMS kurt  
    #   centroid mean, centroid std, centroid skew, centroid kurt, bandwidth mean, bandwidth std, bandwidth skew, bandwidth kurt
    #   flatness mean, flatness std, flatness skew, flatness kurt, rolloff mean, rolloff std, rolloff skew, rolloff kurt
    #   7 OSC mean, 7 OSC std, 7 OSC skew, 7 OSC kurt
    #   7 D-OSC mean, 7 D-OSC std, 7 D-OSC skew, 7 D-OSC kurt
    #   7 DD-OSC mean, 7 DD-OSC std, 7 DD-OSC skew, 7 DD-OSC kurt
    #   12 chroma mean, 12 chroma std, 12 chroma skew, 12 chroma kurt 
    
    if TESTING:
        for j in range(len(features)-1):
            print("%.6f" % features[j], end=",", file=fout)
        print("%.6f" % features[-1], file=fout)    
    else:
        for j in range(len(features)):
            print("%.6f" % features[j], end=",", file=fout)
        print(input[1], file=fout)
       
    line = fin.readline()
    file_cnt += 1

fout.close()
fin.close()