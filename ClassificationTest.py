import numpy as np
from sklearn import svm
import os

import librosa
from librosa import feature
from sklearn.model_selection import train_test_split

cwd = os.getcwd()
classes = ["quiet", "palm", "knuckle", "elbow"]

def get_path(cls):
    return os.path.join(cwd, "training_data", cls) ## Directory

audios_feat = []
y = []

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

## Feature Extraction ##

def get_feature_vector(y, sr): 
    feat_vect_i = [ np.mean(funct(y=y, sr=sr)) for funct in fn_list_i]
    feat_vect_ii = [ np.mean(funct(y=y)) for funct in fn_list_ii] 
    feature_vector = feat_vect_i + feat_vect_ii 
    return feature_vector

for cls in classes:
    path = get_path(cls)
    sample_ind = 0
    sample_files = os.listdir(path)

    while sample_ind < len(sample_files):
        X , sr = librosa.load(f"{path}/sample_{sample_ind}.wav", sr = 44100) #File directory
        feature_vector = get_feature_vector(X, sr)
        audios_feat.append(feature_vector)
        y.append(classes.index(cls))
        sample_ind = sample_ind + 1

### Normalizing ###

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(audios_feat)
Scaled_x = scaler.transform(audios_feat)

### Training ###

iter = 1000
count = 0
correct = 0

for i in range(iter):

    X_train, X_test, y_train, y_test = train_test_split(Scaled_x, y, test_size=0.2)

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    prediction = clf.predict(X_test)
    for j in range(len(prediction)):
        if prediction[j] == y_test[j]:
            correct = correct + 1
        count = count + 1

print("Acc =", correct/count)