import tensorflow as tf
import numpy as np
import tensorflow_io as tfio
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image, display
import matplotlib.pyplot as plt
from keras.models import load_model

# Cargar modelo.
model_relu = load_model('0.01v_2.h5')

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
  # Esctrucuta de predict:= [Fondo, Queltehue, Tronco]

  '''
  #Grafica:
  plt.figure(figsize=(5,3))
  yticks = range(0,3,1)
  xticks=np.arange(0,len(data),10)
  plt.imshow(predict, aspect='auto', interpolation='nearest', cmap='Purples')
  plt.yticks(yticks, ['Tronco', 'Queltehue','Fondo'])
  plt.xticks(xticks, rotation=45)
  plt.show()
  '''
  return predict  

predecir(ruta)
