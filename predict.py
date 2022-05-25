#!pip install tflite-runtime==2.7.0
#!pip install numpy==1.21.5
#!pip install SoundFile==0.10.3.post1
#!pip install librosa==0.8.1


import numpy as np
import soundfile as sf
import librosa
from tflite_runtime.interpreter import Interpreter

# Ruta modelo .tflite
interpreter = Interpreter(model_path="modelo.tflite")

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ruta de audio
audio = 'test.ogg'

data, samplerate = sf.read(audio)
data_split=np.array_split(data,len(data)//44100)

# predic almacenadas
predicciones=[]

for i in range(len(data_split)):
    S = librosa.feature.melspectrogram(y=data_split[i], sr=44100, n_mels=128,fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = S_dB[::-1]
    S_dB = np.expand_dims(S_dB, axis=0)
    S_dB = np.expand_dims(S_dB, axis=3)
    input_shape = input_details[0]['shape']
    input_data = np.float32(S_dB)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predicciones.append(interpreter.get_tensor(output_details[0]['index'])[0])

