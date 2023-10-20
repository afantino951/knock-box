import numpy as np
from sklearn import svm

import librosa
from librosa import feature
from sklearn.model_selection import train_test_split

fn_list_i = [
 feature.chroma_stft,
 feature.spectral_centroid,
 feature.spectral_bandwidth,
 feature.spectral_rolloff
]

fn_list_ii = [
 feature.rms,
 feature.zero_crossing_rate
]

def get_feature_vector(y, sr):
   feat_vect_i = [ np.mean(funct(y=y, sr=sr)) for funct in fn_list_i]
   feat_vect_ii = [ np.mean(funct(y=y)) for funct in fn_list_ii]
   feature_vector = feat_vect_i + feat_vect_ii
   return feature_vector

'''
Import audio files
Q = Quiet
K = Knock
E = Elbow
'''
Q_audio_files = ["Q15.wav", "Q14.wav", "Q13.wav", "Q12.wav", "Q11.wav", "Q10.wav", "Q9.wav", "Q8.wav", "Q7.wav", "Q6.wav", "Q5.wav", "Q4.wav", "Q3.wav", "Q2.wav", "Q1.wav"]
K_audio_files = ["K15.wav", "K14.wav", "K13.wav", "K12.wav", "K11.wav", "K10.wav", "K9.wav", "K8.wav", "K7.wav", "K6.wav", "K5.wav", "K4.wav", "K3.wav", "K2.wav", "K1.wav"]
E_audio_files = ["E15.wav", "E14.wav", "E13.wav", "E12.wav", "E11.wav", "E10.wav", "E9.wav", "E8.wav", "E7.wav", "E6.wav", "E5.wav", "E4.wav", "E3.wav", "E2.wav", "E1.wav"]

Q_audios_feat = []
K_audios_feat = []
E_audios_feat = []

for file in Q_audio_files:
   X , sr = librosa.load('gdrive/My Drive/sounds/' + file, sr = None) #File directory
   feature_vector = get_feature_vector(X, sr)
   Q_audios_feat.append(feature_vector)

for file in K_audio_files:
   X , sr = librosa.load('gdrive/My Drive/sounds/' + file, sr = None) #File directory
   feature_vector = get_feature_vector(X, sr)
   K_audios_feat.append(feature_vector)

for file in E_audio_files:
   X , sr = librosa.load('gdrive/My Drive/sounds/' + file, sr = None) #File directory
   feature_vector = get_feature_vector(X, sr)
   E_audios_feat.append(feature_vector)

X = Q_audios_feat + K_audios_feat + E_audios_feat
y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]

iter = 100
count = 0
correct = 0

for i in range(iter):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    prediction = clf.predict(X_test)
    for j in range(len(prediction)):
        if prediction[j] == y_test[j]:
            correct = correct + 1
        count = count + 1

print("Acc =", correct/count)
