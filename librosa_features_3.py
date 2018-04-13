import librosa
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io.wavfile
import scipy.signal
import scipy.stats

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


print("Getting librosa_features")
fin = open("training_files.txt", "r")
fout = open("mfcc_librosa_updated_csv.csv", "w")

frame_size = 1024
sample_rate = 22050
hop_size = 512

file_cnt = 1
line = fin.readline()
input = []
while line:
    
    print(file_cnt)
    features = []
    
    input = line.split()
    filename = input[0]
    signal, rate = librosa.load(filename)
        
    mfcc = myMFCC(signal, sample_rate, frame_size)
    #mfcc2 = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc_filt, n_fft=frame_size, hop_length=hop_size)
    #print(len(mfcc[0]))
    
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
    features.append(len(note_onset))
    
    zcr = librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_size, hop_length=hop_size) 
    rms = librosa.feature.rmse(y=signal, frame_length=frame_size, hop_length=hop_size)
    features.append(np.mean(zcr))
    features.append(np.mean(rms))
    features.append(np.std(zcr))
    features.append(np.std(rms))    
    features.append(scipy.stats.skew(zcr.T))
    features.append(scipy.stats.skew(rms.T))
    features.append(scipy.stats.kurtosis(zcr.T))
    features.append(scipy.stats.kurtosis(rms.T))
    
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
    spectral_flatness = librosa.feature.spectral_flatness(y=signal, n_fft=frame_size, hop_length=hop_size)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)

    features.append(np.mean(spectral_centroid))
    features.append(np.mean(spectral_bandwidth))
    features.append(np.mean(spectral_flatness))
    features.append(np.mean(spectral_rolloff))        
    features.append(np.std(spectral_centroid))
    features.append(np.std(spectral_bandwidth))
    features.append(np.std(spectral_flatness))
    features.append(np.std(spectral_rolloff)) 
 #   features.append(scipy.stats.skew(spectral_contrast.T))
    features.append(scipy.stats.skew(spectral_flatness.T))
 #   features.append(scipy.stats.kurtosis(spectral_contrast.T))
    features.append(scipy.stats.kurtosis(spectral_flatness.T))
    
    for i in range(len(spectral_contrast)):
        features.append(np.mean(spectral_contrast[i]))
    for i in range(len(spectral_contrast)):
        features.append(np.std(spectral_contrast[i]))
    for i in range(len(spectral_contrast)):
        features.append(scipy.stats.skew(spectral_contrast[i]))
    for i in range(len(spectral_contrast)):
        features.append(scipy.stats.kurtosis(spectral_contrast[i]))
    
    S = np.abs(librosa.stft(y=signal, n_fft=frame_size, hop_length=hop_size))
    chroma = librosa.feature.chroma_stft(S=S, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
    for i in range(len(chroma)):
        features.append(np.mean(chroma[i]))
    for i in range(len(chroma)):
        features.append(np.std(chroma[i]))
    #print(features)
    
    ##  FEATURES
    #   26 MFCC mean, 26 MFCC std, 26 MFCC skew, 26 MFCC kurtosis
    #   26 DMFCC mean, 26 DMFCC std, 26 DMFCC skew, 26 DMFCC kurtosis
    #   26 DDMFCC mean, 26 DDMFCC std, 26 DDMFCC skew, 26 DDMFCC kurtosis
    #   tempo, note onset, ZCR mean, RMS mean, ZCR std, RMS std 
    #   centroid mean, contrast mean, bandwidth mean, flatness mean, rolloff mean
    #   centroid std, contrast std, bandwidth std, flatness std, rollofd std
    #   12 chroma mean, 12 chroma std 
    
    for j in range(len(features)):
        print("%.6f" % features[j], end=",", file=fout)
    print(input[1], file=fout)
    
    #print("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s" % (numpy.mean(bufferrms), numpy.mean(bufferpar), numpy.mean(bufferzcr), numpy.mean(buffermad), numpy.mean(buffermean_ad), numpy.std(bufferrms), numpy.std(bufferpar), numpy.std(bufferzcr), numpy.std(buffermad), numpy.std(buffermean_ad), numpy.mean(buffersc), numpy.mean(buffersro), numpy.mean(buffersfm), numpy.mean(bufferparfft), numpy.mean(bufferflux), numpy.std(buffersc), numpy.std(buffersro), numpy.std(buffersfm), numpy.std(bufferparfft), numpy.std(bufferflux), numpy.mean(buffermfcc[0]), numpy.mean(buffermfcc[1]), numpy.mean(buffermfcc[2]), numpy.mean(buffermfcc[3]), numpy.mean(buffermfcc[4]), numpy.mean(buffermfcc[5]), numpy.mean(buffermfcc[6]), numpy.mean(buffermfcc[7]), numpy.mean(buffermfcc[8]), numpy.mean(buffermfcc[9]), numpy.mean(buffermfcc[10]), numpy.mean(buffermfcc[11]), numpy.mean(buffermfcc[12]), numpy.mean(buffermfcc[13]), numpy.mean(buffermfcc[14]), numpy.mean(buffermfcc[15]), numpy.mean(buffermfcc[16]), numpy.mean(buffermfcc[17]), numpy.mean(buffermfcc[18]), numpy.mean(buffermfcc[19]), numpy.mean(buffermfcc[20]), numpy.mean(buffermfcc[21]), numpy.mean(buffermfcc[22]), numpy.mean(buffermfcc[23]), numpy.mean(buffermfcc[24]), numpy.mean(buffermfcc[25]), numpy.std(buffermfcc[0]), numpy.std(buffermfcc[1]), numpy.std(buffermfcc[2]), numpy.std(buffermfcc[3]), numpy.std(buffermfcc[4]), numpy.std(buffermfcc[5]), numpy.std(buffermfcc[6]), numpy.std(buffermfcc[7]), numpy.std(buffermfcc[8]), numpy.std(buffermfcc[9]), numpy.std(buffermfcc[10]), numpy.std(buffermfcc[11]), numpy.std(buffermfcc[12]), numpy.std(buffermfcc[13]), numpy.std(buffermfcc[14]), numpy.std(buffermfcc[15]), numpy.std(buffermfcc[16]), numpy.std(buffermfcc[17]), numpy.std(buffermfcc[18]), numpy.std(buffermfcc[19]), numpy.std(buffermfcc[20]), numpy.std(buffermfcc[21]), numpy.std(buffermfcc[22]), numpy.std(buffermfcc[23]), numpy.std(buffermfcc[24]), numpy.std(buffermfcc[25]), input[1]), file=fout) 
    
    line = fin.readline()
    file_cnt += 1

fout.close()
fin.close()
print("Completed librosa_features")