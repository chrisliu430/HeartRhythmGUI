import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.models import load_model
import sys
import getpass

audioFile = sys.argv[1]
imgName = sys.argv[2].strip(".wav") + ".png"
#change_your_data
H030H, fs = librosa.load(audioFile, sr=22050) #load_audio
H030H_dur = librosa.get_duration(y=H030H, sr=22050) #audio_duration

librosa.display.waveplot(H030H, fs, x_axis='time')
plt.title("Heart Sound Chart")
plt.savefig("./images/HeartSound/" + imgName)
# plt.show()

mfccs = np.mean(librosa.feature.mfcc(y=H030H, sr=22050, n_mfcc=20).T,axis=0) #extract_20_mfccs
A = np.reshape(mfccs,(1,20,-1))

model = load_model('./models/' + sys.argv[3] + '.hdf5')
result = model.predict(A)
max = np.argmax(result)

sys.stdout.write(str(max))