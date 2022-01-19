import tensorflow as tf
import numpy as np
import tensorflow_io as tfio
from tensorflow import keras
from IPython.display import Image, display
import matplotlib.pyplot as plt
from keras.models import load_model

'''
Clasificaciones del modelo:
['E-aves_fondo',
 'E-grillo',
 'I-grillo_auto_camion',
 'E-grillo-motorauto',
 'E-grillo-viento',
 'I-aves_fondo_Indus',
 'E-queltehues',
 'I-troncos',
 'E-fondo']
'''

# Cargar modelo.
model_relu = load_model('0.045v.h5')

# Cargar sonido
ruta='ejemplo2.ogg'

def predecir(ruta):
  audio_10=ruta
  data=[]
  audio_binary = tf.io.read_file(audio_10)   
  waveform = tfio.audio.decode_vorbis(audio_binary)    
  split_wave=np.array_split(waveform,len(waveform)/44100)
  for j in range(len(waveform)//44100):
      signals = tf.reshape(split_wave[j], [1, -1])
      spec=tfio.audio.spectrogram(signals, nfft=512, window=512, stride=256)
      data.append(spec)
  predict=[]
  for i in range(len(data)):
      predict.append(model_relu.predict(data[i])[0])
  predict=np.transpose(predict)
  return predict  

predecir(ruta)
