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
 'I-cadenas',
 'E-grillo',
 'I-grillo_auto_camion',
 'I-grillo-camion_retro',
 'I-camion_aves',
 'I-fondo_industria',
 'E-grillo_gaviotas',
 'I-grillo-golpeteo',
 'E-grillo-motorauto',
 'E-grillo-pasos_pasto',
 'E-grillo-queltehue',
 'E-grillo-viento',
 'E-grillo_voz_mujer',
 'E-grillo-voz_niña',
 'E-grillo_voz_niño',
 'I-aves_fondo_Indus',
 'I-pajaro_tronco',
 'E-perro_pajaros',
 'I-queltehue_pajaro_fondo_indus',
 'E-queltehues',
 'I-troncos']

'''


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

  '''
  #Grafica:
  plt.figure(figsize=(5,3))
  yticks = range(0,3,1)
  xticks=np.arange(0,len(data),10)
  plt.imshow(predict, aspect='auto', interpolation='nearest', cmap='Purples')
  plt.yticks(yticks, ['E-aves_fondo',
 'I-cadenas',
 'E-grillo',
 'I-grillo_auto_camion',
 'I-grillo-camion_retro',
 'I-camion_aves',
 'I-fondo_industria',
 'E-grillo_gaviotas',
 'I-grillo-golpeteo',
 'E-grillo-motorauto',
 'E-grillo-pasos_pasto',
 'E-grillo-queltehue',
 'E-grillo-viento',
 'E-grillo_voz_mujer',
 'E-grillo-voz_niña',
 'E-grillo_voz_niño',
 'I-aves_fondo_Indus',
 'I-pajaro_tronco',
 'E-perro_pajaros',
 'I-queltehue_pajaro_fondo_indus',
 'E-queltehues',
 'I-troncos'])
  plt.xticks(xticks, rotation=45)
  plt.show()
  '''
  return predict  

predecir(ruta)
