import os
import sys
import argparse
import numpy as np
import sounddevice as sd

from joblib import dump, load


classes = ["quiet", "palm", "knuckle", "elbow"]

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

    return os.listdir(cls_data_dir)

def preprocess_data(class_files_data):
    """
    Open the .wav files in list and preprocess the data 
    returns tuple([training data], [test data])
    """
    pass

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
    quiet_data = get_training_data("quiet", "training_data")

    preprocess_data(quiet_data)

    clf = []
    if args.append_model is not None:
        clf = load_model(args.append_model)

    train_svm()


    if args.save_file is not None:
        dump(clf, args.save_file)
    
