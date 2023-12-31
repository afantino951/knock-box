import os
import sys
import argparse
import numpy as np
import sounddevice as sd
import librosa
from librosa import feature
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from joblib import dump, load


classes = ["quiet", "palm", "knuckle", "elbow"]

#fn_list_i = [
    #feature.chroma_stft,
    #feature.spectral_centroid,
    #feature.spectral_bandwidth,
    #feature.spectral_rolloff,
    #feature.spectral_contrast,
    #feature.mfcc,
    #feature.melspectrogram
#]
 
#fn_list_ii = [
    #feature.rms,
    #feature.zero_crossing_rate,
    #feature.spectral_flatness,
#]

def get_training_data(cls, dir):
    """
    Return list of filenames for the given class label from the training dir
    """
    if not os.path.isdir(dir):
        print("ERR: dir is not a directory")
        sys.exit(-1)

    cls_data_dir = os.path.join(dir, cls)

    if not os.path.isdir(cls_data_dir):
        print(f"ERR: {cls} directory does not exist in {dir}")
        return []

    return os.listdir(cls_data_dir), cls_data_dir

def get_feature_vector(y, sr): 
    #feat_vect_i = [np.mean(funct(y=y, sr=sr), axis=1) for funct in fn_list_i]
    #feat_vect_ii = [np.mean(funct(y=y), axis=1) for funct in fn_list_ii]
    #feature_vector = feat_vect_i + feat_vect_ii
    feature_vector = np.mean(feature.mfcc(y=y, sr=sr), axis=1)
    return feature_vector

def preprocess_data(class_files_data, class_files_dir, cls):
    """
    Open the .wav files in list and preprocess the data 
    returns tuple([training data], [y])
    """
    audios_feat = []
    y = []
    
    for files in (x for x in class_files_data if x.endswith('.wav')):
        X, sr = librosa.load(f"{class_files_dir}/{files}", sr = 44100) #File directory
        feature_vector = get_feature_vector(X, sr)
        audios_feat.append(feature_vector)
        y.append(classes.index(cls))

    return audios_feat, y

def train_svm(training_data, labels, clf):
    """
    Train the svm with the data provided. 
    returns the model that with the best accuracy
    """
    scaler = StandardScaler()

    if clf == []:
        scaler.fit(training_data)
        Scaled_x_train = scaler.transform(training_data)

        ## GridSearchCV
    
        param_grid = {'penalty': ['l1', 'l2', 'elasticnet', None],
                      'alpha': [0.1, 0.01, 0.001, 0.0001]}

        grid = GridSearchCV(SGDClassifier(), param_grid, refit=True, verbose=3) 
        grid.fit(Scaled_x_train, labels)

        clf_t = grid.best_estimator_

    else:
        clf_t = clf

    ## Train clf_t and return best trained model

    iteration = 1000

    clfs = []
    accs = []

    for i in range(iteration):
        clf_temp = clf_t
        
        count = 0
        correct = 0

        X_train, X_test, y_train, y_test = train_test_split(training_data, labels, test_size=0.2)

        scaler = StandardScaler()
        scaler.fit(X_train)
        Scaled_x_train = scaler.transform(X_train)

        clf_temp.partial_fit(Scaled_x_train, y_train)

        Scaled_x_test = scaler.transform(X_test)

        prediction = clf_t.predict(Scaled_x_test)
        for j in range(len(prediction)):
            if prediction[j] == y_test[j]:
                correct = correct + 1
            count = count + 1

        clfs.append(clf_temp)
        accs.append(correct/count)
    
    return clfs[accs.index(max(accs))], scaler

def load_model(filename):
    if not os.path.exists(filename):
        print(f"ERR: file does not exist")
        sys.exit(-1)

    return load(filename)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="script to train a SVM"
            )
    parser.add_argument("data_dir", metavar="[PATH_DIR]",
                        help="The directory that contains your training data")
    parser.add_argument("-c", "--classes", 
                        required=False, 
                        help="NOT IMPLEMENTED YET"
                        )
    parser.add_argument("-s", "--save", metavar="[FILE_PATH]",
                        dest="save_file", 
                        required=False, 
                        help="Save resulting model to file"
                        )
    parser.add_argument("-ss", "--savescaler", metavar="[FILE_PATH]",
                        dest="save_scaler", 
                        required=False, 
                        help="Save scaler to file"
                        )
    parser.add_argument("-a", "--append", metavar="[FILE_PATH]", 
                        dest="append_model", 
                        required=False,
                        help="Use model stored in a .joblib as a starting point for training")
    args = parser.parse_args()


    print(args)

    #TODO @Allen: add SVM logic to functions and put best trained model in `clf`

    training_data = []
    labels = []
    
    for cls in classes:
        data, data_dir = get_training_data(cls, "training_data")
        preprocessed_data, label = preprocess_data(data, data_dir, cls)
        training_data = training_data + preprocessed_data
        labels = labels + label

    clf = []
    if args.append_model is not None:
        clf = load_model(args.append_model)

    clf, scaler = train_svm(training_data, labels, clf)


    if args.save_file is not None:
        dump(clf, args.save_file)
    if args.save_scaler is not None:
        dump(scaler, args.save_scaler)
    
