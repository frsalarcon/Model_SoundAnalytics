import tensorflow as tf
import numpy as np
import tensorflow_io as tfio
from tensorflow import keras
from keras.models import load_model

#tensorflow==2.8.0
#numpy== 1.20.3
#tensorflow_io==0.23.1
#keras==2.8.0


model = load_model('model.h5')

ruta='t1.ogg'

def predict(ruta):
    data=[]
    audio_binary = tf.io.read_file(ruta)  
    waveform = tfio.audio.decode_vorbis(audio_binary)  
    waveform = tf.squeeze(waveform, axis=[-1])  
    split_wave=np.array_split(waveform,len(waveform)//44100)
    for j in range(len(waveform)//44100):
        signals = tf.reshape(split_wave[j], [1, -1])
        spec=tfio.audio.spectrogram(signals, nfft=2048, window=512, stride=256)
        mel_spectrogram = tfio.audio.melscale(spec, rate=44100, mels=128, fmin=0, fmax=16000)
        data.append(mel_spectrogram)
    predict=[]
    for i in range(len(data)):
        predict.append(model.predict(data[i])[0])
    return predict

predict(ruta)