#1---------------------------------------------------------------------

from flask import Flask, request, render_template
import numpy as np
import scipy
import pickle
import librosa
from scipy.fftpack import dct

app = Flask(__name__)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

#---------------------------------------------------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp3'}

def initialize(mp):
    signal,sr = librosa.load(mp,duration=8,sr=16000)
    signal = signal[0:int(3.5 * sr)]
    return sr,signal

def lowPassFilter(signal, pre_emphasis=0.97):
	return np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

def preEmphasis(mp):
	sr,signal = initialize(mp)										
	pre_emphasis=0.97														
	emphasizedSignal = lowPassFilter(signal)				
	return emphasizedSignal

def framing(mp):	
    windowSize = 0.75			
    windowStep = 0.01				
    sr, signal = initialize(mp)
    frame_length, frame_step = windowSize * sr, windowStep * sr
    signal_length = len(preEmphasis(mp))
    overlap = int(round(frame_length)) 
    frameSize = int(round(frame_step)) 
    numberOfframes = int(np.ceil(float(np.abs(signal_length - frameSize)) / overlap ))
    pad_signal_length = numberOfframes * frameSize + overlap
    if pad_signal_length >= signal_length:
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(preEmphasis(mp), z)
    else:
        pad_signal = preEmphasis(mp)

    indices = np.tile(np.arange(0, overlap), (numberOfframes, 1)) + np.tile(np.arange(0, 
                numberOfframes * frameSize, frameSize), (overlap, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)
    return frames

def fouriertransform(mp):				
	NFFT = 512
	frames = framing(mp)
	mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  
	pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  
	return pow_frames

def filterbanks(mp):
    nfilt = 40
    low_freq_mel = 0
    NFFT = 512

    sr, signal = initialize(mp)
    high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))  
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  
    hz_points = (700 * (10**(mel_points / 2595) - 1))  
    bin = np.floor((NFFT + 1) * hz_points / sr)

    pow_frames = fouriertransform(mp)
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   
        f_m = int(bin[m])             
        f_m_plus = int(bin[m + 1])    

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  
    filter_banks = 20 * np.log10(filter_banks) 
  
    return filter_banks

def	mfcc(mp):
	num_ceps = 12
	cep_lifter = 22
	filter_banks = filterbanks(mp)
	mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
	(nframes, ncoeff) = mfcc.shape
	n = np.arange(ncoeff)
	lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
	mfcc *= lift
	mfcc = np.mean(mfcc, axis=0)
	return mfcc

#2---------------------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file.save(file.filename)
    mp = file.filename
    mfcc_features = mfcc(mp).reshape(1, -1)
    prediction = model.predict(mfcc_features)
    if prediction[0] == 0:
        gender = "cewek"
    else:
        gender = "cowok"
    return render_template('index.html', prediction=gender)

if __name__ == '__main__':
    app.run(debug=True)

#---------------------------------------------------------------------