import sounddevice as sd
import numpy as np
from joblib import dump, load
import librosa
from librosa import feature
from sklearn.preprocessing import StandardScaler
import requests
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from train import load_model
from train import get_feature_vector
from train import preprocess_data

# servo functions

esp32_ip = "192.168.137.66"

def close_door():
    servo_angle_close = 180
    url_close = f"http://{esp32_ip}/?value={servo_angle_close}"
    try:
        response = requests.get(url_close)
        if response.status_code == 200:
            print(f"Successfully set servo angle to {servo_angle_close} degrees.")
        else:
            print(f"Failed to set servo angle. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")

def open_door():
   servo_angle_open = 90
   url_open = f"http://{esp32_ip}/?value={servo_angle_open}"
   try:
       response = requests.get(url_open)
       if response.status_code == 200:
           print(f"Successfully set servo angle to {servo_angle_open} degrees.")
       else:
           print(f"Failed to set servo angle. Status code: {response.status_code}")
   except requests.exceptions.RequestException as e:
       print(f"Request error: {e}")

class DoorStateMachine:
    def __init__(self, allowable_quiet):
        self.state = "start"
        #self.correct_sequence = ["knuckle", "palm", "knuckle", "knuckle", "palm"]
        self.correct_sequence = ["knuckle", "elbow", "knuckle"]
        self.sequence_index = 0
        self.quiet_count = 0
        self.quiet_limit = allowable_quiet
        self.ignore = False
        self.ignore_count = 0
        close_door()

    def process_interaction(self, interaction):
        if self.state == "start":
            if interaction == self.correct_sequence[self.sequence_index] and not self.ignore:
                self.sequence_index += 1
                print(interaction)
                print("Correct Count:", self.sequence_index, '/', len(self.correct_sequence))
                self.quiet_count = 0
                self.ignore = True
            elif interaction == "quiet" and not self.ignore:
                self.quiet_count += 1
                if self.quiet_count > self.quiet_limit:
                    self.sequence_index = 0
                    self.quiet_count = 0
                    print("Reset, Quiet too Long")
                self.ignore_count = 0
                self.ignore = False
            elif self.ignore:
                self.ignore_count += 1
                if self.ignore_count > 3:
                    self.ignore_count = 0
                    self.ignore = False
            else:
                self.ignore = True
                self.sequence_index = 0
                self.quiet_count = 0
                print("Reset, Wrong Answer: ", interaction)

            if self.sequence_index == len(self.correct_sequence):
                self.state = "success"
        elif self.state == "success":
            print("Door Opened!")
            open_door()
            self.state = "start"
            self.sequence_index = 0
            self.quiet_count = 0
            time.sleep(10)
            close_door()
            print("Door Closed!")

classes = ["quiet", "palm", "knuckle", "elbow"]

#fn_list_i = [
#    feature.chroma_stft,
#    feature.spectral_centroid,
#    feature.spectral_bandwidth,
#    feature.spectral_rolloff,
#    feature.spectral_contrast,
#    feature.mfcc,
#    feature.melspectrogram
#]

#fn_list_ii = [
#    feature.rms,
#    feature.zero_crossing_rate,
#    feature.spectral_flatness,
#]

clf = load_model("model.joblib") # Model name
scaler = load_model("scaler.joblib")

door_state_machine = DoorStateMachine(15)
door_state_machine.state = "start"
door_state_machine.sequence_index = 0
door_state_machine.quiet_count = 0

def audio_callback(indata, frames, time, status):
    # Process the audio data here

    #print("Processing audio chunk:", indata.shape)

    audios_feat = []
    feature_vector = get_feature_vector(indata[:,0], 44100)
    audios_feat.append(feature_vector)
    Scaled_test = scaler.transform(audios_feat)

    #print(clf.predict(Scaled_test))
    door_state_machine.process_interaction(classes[clf.predict(Scaled_test)[0]])
    
    

# Set the audio parameters
sample_rate = 44100  # Sample rate in Hz
chunk_duration = 0.2  # 1-second audio chunks
block_size = int(sample_rate * chunk_duration)

# Start streaming audio input with the callback function
with sd.InputStream(callback=audio_callback, blocksize=block_size):
    print(f"Recording audio in {chunk_duration} second chunks. Press Ctrl+C to stop.")
    try:
        sd.sleep(10000000000)  # Run the audio stream indefinitely
    except KeyboardInterrupt:
        print("\nRecording stopped.")

# Clean up the audio stream
sd.stop()
