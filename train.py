import os
import sys
import argparse
import numpy as np
import sounddevice as sd
import librosa
from librosa import feature

from joblib import dump, load


classes = ["quiet", "palm", "knuckle", "elbow"]

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
    feat_vect_i = [ np.mean(funct(y=y, sr=sr)) for funct in fn_list_i]
    feat_vect_ii = [ np.mean(funct(y=y)) for funct in fn_list_ii] 
    feature_vector = feat_vect_i + feat_vect_ii 
    return feature_vector

def preprocess_data(class_files_data, class_files_dir, cls):
    """
    Open the .wav files in list and preprocess the data 
    returns tuple([training data], [y])
    """
    audios_feat = []
    y = []
    
    for files in (x for x in class_files_data if x.endswith('.wav')):
        X , sr = librosa.load(f"{class_files_dir}/{class_files_data}", sr = 44100) #File directory
        feature_vector = get_feature_vector(X, sr)
        audios_feat.append(feature_vector)
        y.append(classes.index(cls))
        
    return audios_feat, y

def train_svm():
    """
    Train the svm with the data provided. 
    returns the model that with the best accuracy
    """
    pass

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
    parser.add_argument("-a", "--append", metavar="[FILE_PATH]", 
                        dest="append_model", 
                        required=False,
                        help="Use model stored in a .joblib as a starting point for training")
    args = parser.parse_args()


    print(args)

    #TODO @Allen: add SVM logic to functions and put best trained model in `clf`
    quiet_data , quiet_data_dir = get_training_data("quiet", "training_data")
    palm_data , palm_data_dir = get_training_data("palm", "training_data")
    knuckle_data , knuckle_data_dir = get_training_data("knuckle", "training_data")
    elbow_data , elbow_data_dir = get_training_data("elbow", "training_data")

    quiet_preprocessed_data, quiet_label = preprocess_data(quiet_data, quiet_data_dir, "quiet")
    palm_preprocessed_data, palm_label = preprocess_data(palm_data, palm_data_dir, "palm")
    knuckle_preprocessed_data, knuckle_label = preprocess_data(knuckle_data, knuckle_data_dir, "knuckle")
    elbow_preprocessed_data, elbow_label = preprocess_data(elbow_data, elbow_data_dir, "elbow")

    preprocessed_data = quiet_preprocessed_data + palm_preprocessed_data + knuckle_preprocessed_data + elbow_preprocessed_data
    labels = quiet_label + palm_label + knuckle_label + elbow_label

    clf = []
    if args.append_model is not None:
        clf = load_model(args.append_model)

    train_svm()


    if args.save_file is not None:
        dump(clf, args.save_file)
    
