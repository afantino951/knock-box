import sounddevice as sd
import argparse
import os
from pynput import keyboard
from scipy.io.wavfile import write


cwd = os.getcwd()
classes = ["quiet", "palm", "knuckle", "elbow"]
fs = 44100  # Sample rate
seconds = 2  # Duration of recording

def get_path(cls_ind):
    return os.path.join(cwd, "training_data", classes[cls_ind])

def get_sample_ind(path):
    sample_files = os.listdir(path)
    # print(len(sample_files))
    if len(sample_files) == 0: return 0
    
    return len(sample_files)
    


if __name__ == "__main__":
    cls_str = ""
    for i, cls in enumerate(classes):
        f_str = f"{(i+1)} {cls} \n"
        cls_str += f_str
    
    cls_ind = int(input(f"Enter the class number that you would like to train\n{cls_str}"))
    cls_ind -= 1

    
    path = get_path(cls_ind)
    if not os.path.exists(path):
        print(f"training_data/{classes[cls_ind]} does not exist. Making it now")
        os.makedirs(path)


    while True:
        # Set the filepath and sample number for the data
        path = get_path(cls_ind)
        sample_ind = get_sample_ind(path)
        filename = f"{path}/sample_{sample_ind}.wav"
        # print(filename)

        # Record the sample and save to file
        print("RECORDING STARTING")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        write(filename, fs, myrecording)  # Save as WAV file 
        print("RECORDING ENDED")
        print(f"Saving in {filename}")

        cont_str = input(f"Press enter to record another {classes[cls_ind]} sample. Press any other key and enter to quit: ")
    
        if cont_str != "":
            break




    
