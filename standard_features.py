import librosa
import math
import matplotlib.pyplot as plt
import numpy
import scipy
import scipy.io.wavfile
import scipy.signal

print("Getting standard_features")
fin = open("training_files.txt", "r")
fout = open("training_arff.arff", "w")

print("@RELATION music_speech@ATTRIBUTE RMS_MEAN_TIME NUMERIC@ATTRIBUTE PAR_MEAN_TIME NUMERIC@ATTRIBUTE ZCR_MEAN_TIME NUMERIC@ATTRIBUTE MAD_MEAN_TIME NUMERIC@ATTRIBUTE MEAN_AD_MEAN_TIME NUMERIC@ATTRIBUTE RMS_STD_TIME NUMERIC@ATTRIBUTE PAR_STD_TIME NUMERIC@ATTRIBUTE ZCR_STD_TIME NUMERIC@ATTRIBUTE MAD_STD_TIME NUMERIC@ATTRIBUTE MEAN_AD_STD_TIME NUMERIC@ATTRIBUTE SC_MEAN_SPECTRAL NUMERIC@ATTRIBUTE SRO_MEAN_SPECTRAL NUMERIC@ATTRIBUTE SFM_MEAN_SPECTRAL NUMERIC@ATTRIBUTE PARFFT_MEAN_SPECTRAL NUMERIC@ATTRIBUTE FLUX_MEAN_SPECTRAL NUMERIC@ATTRIBUTE SC_STD_SPECTRAL NUMERIC@ATTRIBUTE SRO_STD_SPECTRAL NUMERIC@ATTRIBUTE SFM_STD_SPECTRAL NUMERIC@ATTRIBUTE PARFFT_STD_SPECTRAL NUMERIC@ATTRIBUTE FLUX_STD_SPECTRAL NUMERIC@ATTRIBUTE MFCC_0 NUMERIC@ATTRIBUTE MFCC_1 NUMERIC@ATTRIBUTE MFCC_2 NUMERIC@ATTRIBUTE MFCC_3 NUMERIC@ATTRIBUTE MFCC_4 NUMERIC@ATTRIBUTE MFCC_5 NUMERIC@ATTRIBUTE MFCC_6 NUMERIC@ATTRIBUTE MFCC_7 NUMERIC@ATTRIBUTE MFCC_8 NUMERIC@ATTRIBUTE MFCC_9 NUMERIC@ATTRIBUTE MFCC_10 NUMERIC@ATTRIBUTE MFCC_11 NUMERIC@ATTRIBUTE MFCC_12 NUMERIC@ATTRIBUTE MFCC_13 NUMERIC@ATTRIBUTE MFCC_14 NUMERIC@ATTRIBUTE MFCC_15 NUMERIC@ATTRIBUTE MFCC_16 NUMERIC@ATTRIBUTE MFCC_17 NUMERIC@ATTRIBUTE MFCC_18 NUMERIC@ATTRIBUTE MFCC_19 NUMERIC@ATTRIBUTE MFCC_20 NUMERIC@ATTRIBUTE MFCC_21 NUMERIC@ATTRIBUTE MFCC_22 NUMERIC@ATTRIBUTE MFCC_23 NUMERIC@ATTRIBUTE MFCC_24 NUMERIC@ATTRIBUTE MFCC_25 NUMERIC@ATTRIBUTE MFCC_26 NUMERIC@ATTRIBUTE MFCC_27 NUMERIC@ATTRIBUTE MFCC_28 NUMERIC@ATTRIBUTE MFCC_29 NUMERIC@ATTRIBUTE MFCC_30 NUMERIC@ATTRIBUTE MFCC_31 NUMERIC@ATTRIBUTE MFCC_32 NUMERIC@ATTRIBUTE MFCC_33 NUMERIC@ATTRIBUTE MFCC_34 NUMERIC@ATTRIBUTE MFCC_35 NUMERIC@ATTRIBUTE MFCC_36 NUMERIC@ATTRIBUTE MFCC_37 NUMERIC@ATTRIBUTE MFCC_38 NUMERIC@ATTRIBUTE MFCC_39 NUMERIC@ATTRIBUTE MFCC_40 NUMERIC@ATTRIBUTE MFCC_41 NUMERIC@ATTRIBUTE MFCC_42 NUMERIC@ATTRIBUTE MFCC_43 NUMERIC@ATTRIBUTE MFCC_44 NUMERIC@ATTRIBUTE MFCC_45 NUMERIC@ATTRIBUTE MFCC_46 NUMERIC@ATTRIBUTE MFCC_47 NUMERIC@ATTRIBUTE MFCC_48 NUMERIC@ATTRIBUTE MFCC_49 NUMERIC@ATTRIBUTE MFCC_50 NUMERIC@ATTRIBUTE MFCC_51 NUMERIC@ATTRIBUTE class {blues,classical,country,disco,hiphop,jazz,metal,pop,reggae,rock}@DATA", file=fout)

# Building Mel filters
nfilt = 26
nfft = 1024
samplerate = 22050

low_freq = 0
high_freq = (1127 * numpy.log(1 + (samplerate/2)/700))

# Space out equally the Mel points
mel_points = numpy.linspace(low_freq, high_freq, nfilt + 2)
# Convert Mel points back to frequency (Hz)
hz_points = (700 * (numpy.exp(mel_points/1127) - 1))
bin = nfft * hz_points/samplerate

fbank = numpy.zeros((nfilt, int(numpy.floor(nfft/2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(numpy.floor(bin[m-1]))   # left
    f_m = int(numpy.round(bin[m]))           # center
    f_m_plus = int(numpy.ceil(bin[m+1]))     # right

    for k in range(f_m_minus, f_m):
        fbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus)
    for k in range(f_m, f_m_plus):
        fbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m)

# Plotting Mel filters
#xl = [i*(samplerate/nfft) for i in range(len(fbank[0]))]
#plt.figure()
#for i in range(len(fbank)):
#    plt.plot(xl, fbank[i])
#plt.axis([0, 12000, 0.0, 1.0])
#plt.title("26 Triangular MFCC filters, 22050Hz signal, window size 1024")
#plt.xlabel("Frequency (Hz)")
#plt.ylabel("Amplitude")
#plt.show()
#
#plt.figure()
#for i in range(len(fbank)):
#    plt.plot(xl, fbank[i], marker='o', linestyle='-')
#plt.axis([0, 12000, 0.0, 1.0])
#plt.title("26 Triangular MFCC filters, 22050Hz signal, window size 1024")
#plt.xlabel("Frequency (Hz)")
#plt.ylabel("Amplitude")
#plt.xlim((0,300))
#plt.show()

file_cnt = 1
line = fin.readline()
input = []
while line:
    print(file_cnt)
    input = line.split()

    filename = input[0]
    
    floatdata, rate = librosa.load(filename)
    
#    rate, data = scipy.io.wavfile.read(filename)
#    floatdata = []
#    floatdata[:] = [x / 32768.0 for x in data]
    
    # Get list of buffer start indices
    bufferstart = []
    for i in range(len(floatdata)):
        if i*512+1024 >= len(floatdata):
            break;
        bufferstart.append(i*512);
    
    # Initialising buffer feature vector
    bufferrms = []
    bufferpar = []
    bufferzcr = []
    buffermad = []
    buffermean_ad = []
    buffersc = []
    buffersro = []
    buffersfm = []
    bufferparfft = []
    bufferflux = []
    buffermfcc = numpy.zeros((nfilt, len(bufferstart)))
    
    for i in range(len(bufferstart)):
        bufferdata = floatdata[bufferstart[i]:bufferstart[i]+1024]
        
#        if all([x == 0 for x in bufferdata]):
#            print(i, " buffer all zeros")
        
        ## TEMPORAL
        # Root-mean-squared
        sq = 0.0
        for j in range(len(bufferdata)):
            sq += bufferdata[j]**2
        buf_rms = math.sqrt(sq/len(bufferdata))
        bufferrms.append(buf_rms)
        
        # Peak-to-average ratio (temporal)
        if buf_rms != 0:
            bufferpar.append(max(numpy.fabs(bufferdata))/buf_rms)
        else:
            bufferpar.append(max(numpy.fabs(bufferdata)))
            
        # Zero crossing
        zccnt = 0
        for j in range(1,len(bufferdata)):
            if (bufferdata[j]*bufferdata[j-1]) < 0:
                zccnt += 1
        bufferzcr.append(zccnt/(len(bufferdata)-1))
        
        # Median absolute deviation    
        median = numpy.median(bufferdata)
        mediandata = []
        mediandata[:] = [numpy.abs(x - median) for x in bufferdata]
        buffermad.append(numpy.median(mediandata))    
            
        # Mean absolute deviation    
        mean = numpy.mean(bufferdata)
        meandata = []
        meandata[:] = [numpy.abs(x - mean) for x in bufferdata]
        buffermean_ad.append(numpy.mean(meandata))
        
        ## SPECTRAL
        window = scipy.signal.hamming(len(bufferdata))
        bufferwin = bufferdata * window
        
        bufferfft = scipy.fftpack.fft(bufferwin)
        bufferfft = bufferfft[0:(int)(len(bufferfft)/2)+1]        
        bufferfft = numpy.abs(bufferfft)
        
        if all([x == 0 for x in bufferfft]):
            bufferfft_nonzero = [0.000001]*len(bufferfft)
        else:
            bufferfft_nonzero = bufferfft
        
        # Spectral centroid
        sc_num = 0
        for j in range(len(bufferfft)):
            sc_num += j * bufferfft[j]
        if numpy.sum(bufferfft) != 0:
            buffersc.append(sc_num/numpy.sum(bufferfft))
        else:
            buffersc.append(sc_num)
        
        # Spectral roll-off
        L_threshold = 0.85 * sum(bufferfft)
        for R in range(len(bufferfft)):
            if sum(bufferfft[0:R+1]) >= L_threshold:
                break;
        buffersro.append(R)        
        
        # Spectral flatness measure
        log = numpy.log(bufferfft_nonzero)
        #log = numpy.log(bufferfft)
        geo_mean = numpy.exp(numpy.mean(log))
        if numpy.mean(bufferfft) != 0:
            buffersfm.append(geo_mean/numpy.mean(bufferfft)) 
        else:
            buffersfm.append(geo_mean)
            
        # Peak-to-average ratio (spectral)
        sq = 0.0
        for j in range(len(bufferfft)):
            sq += bufferfft[j]**2
        buf_rms = math.sqrt(sq/len(bufferfft))  
        if buf_rms != 0:
            bufferparfft.append(max(numpy.fabs(bufferfft))/buf_rms)
        else:
            bufferparfft.append(max(numpy.fabs(bufferfft)))
        
        # Spectral flux
        flux = 0;
        if i == 0:
            # Flux 0 is just sum of elements of first buffer
            bufferflux.append(sum(bufferfft))
        else:
            for j in range(len(bufferfft)):
                if bufferfft[j] - prevbuffer[j] > 0:
                    flux += bufferfft[j] - prevbuffer[j]
            bufferflux.append(flux)
        # Save as previous buffer for use in next iteration
        prevbuffer = bufferfft
  
        ## PERCEPTUAL
        # MFCC
        emp_bufferdata = numpy.append(bufferdata[0], bufferdata[1:]-0.95*numpy.asarray(bufferdata[:-1]))
        window = scipy.signal.hamming(len(emp_bufferdata))
        emp_bufferdata = emp_bufferdata * window
        
        bufferfft = scipy.fftpack.fft(emp_bufferdata)
        bufferfft = bufferfft[0:(int)(len(bufferfft)/2)+1]        
        bufferfft = numpy.abs(bufferfft)
        
        if all([x == 0 for x in bufferfft]):
            bufferfft_nonzero = [0.000001]*len(bufferfft)
        else:
            bufferfft_nonzero = bufferfft
        
        mel_vector = []
        for j in range(len(fbank)):
            mel_vector.append(numpy.log10(numpy.dot(bufferfft_nonzero, fbank[j])))         
        mfcc = scipy.fftpack.dct(mel_vector)
        
        for j in range(len(fbank)):
            buffermfcc[j,i] = mfcc[j]
            
    print("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s" % (numpy.mean(bufferrms), numpy.mean(bufferpar), numpy.mean(bufferzcr), numpy.mean(buffermad), numpy.mean(buffermean_ad), numpy.std(bufferrms), numpy.std(bufferpar), numpy.std(bufferzcr), numpy.std(buffermad), numpy.std(buffermean_ad), numpy.mean(buffersc), numpy.mean(buffersro), numpy.mean(buffersfm), numpy.mean(bufferparfft), numpy.mean(bufferflux), numpy.std(buffersc), numpy.std(buffersro), numpy.std(buffersfm), numpy.std(bufferparfft), numpy.std(bufferflux), numpy.mean(buffermfcc[0]), numpy.mean(buffermfcc[1]), numpy.mean(buffermfcc[2]), numpy.mean(buffermfcc[3]), numpy.mean(buffermfcc[4]), numpy.mean(buffermfcc[5]), numpy.mean(buffermfcc[6]), numpy.mean(buffermfcc[7]), numpy.mean(buffermfcc[8]), numpy.mean(buffermfcc[9]), numpy.mean(buffermfcc[10]), numpy.mean(buffermfcc[11]), numpy.mean(buffermfcc[12]), numpy.mean(buffermfcc[13]), numpy.mean(buffermfcc[14]), numpy.mean(buffermfcc[15]), numpy.mean(buffermfcc[16]), numpy.mean(buffermfcc[17]), numpy.mean(buffermfcc[18]), numpy.mean(buffermfcc[19]), numpy.mean(buffermfcc[20]), numpy.mean(buffermfcc[21]), numpy.mean(buffermfcc[22]), numpy.mean(buffermfcc[23]), numpy.mean(buffermfcc[24]), numpy.mean(buffermfcc[25]), numpy.std(buffermfcc[0]), numpy.std(buffermfcc[1]), numpy.std(buffermfcc[2]), numpy.std(buffermfcc[3]), numpy.std(buffermfcc[4]), numpy.std(buffermfcc[5]), numpy.std(buffermfcc[6]), numpy.std(buffermfcc[7]), numpy.std(buffermfcc[8]), numpy.std(buffermfcc[9]), numpy.std(buffermfcc[10]), numpy.std(buffermfcc[11]), numpy.std(buffermfcc[12]), numpy.std(buffermfcc[13]), numpy.std(buffermfcc[14]), numpy.std(buffermfcc[15]), numpy.std(buffermfcc[16]), numpy.std(buffermfcc[17]), numpy.std(buffermfcc[18]), numpy.std(buffermfcc[19]), numpy.std(buffermfcc[20]), numpy.std(buffermfcc[21]), numpy.std(buffermfcc[22]), numpy.std(buffermfcc[23]), numpy.std(buffermfcc[24]), numpy.std(buffermfcc[25]), input[1]), file=fout) 
    
    line = fin.readline()
    file_cnt += 1

fout.close()
fin.close()
print("Completed standard_features")