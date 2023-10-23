import sounddevice as sd
import numpy as np
from joblib import dump, load
import librosa
from librosa import feature
from sklearn.preprocessing import StandardScaler

from train import load_model
from train import get_feature_vector
from train import preprocess_data


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

clf = load_model("model.joblib") # Model name

def audio_callback(indata, frames, time, status):
    # Process the audio data here

    print("Processing audio chunk:", indata.shape)
    
    audios_feat = []
    
    feature_vector = get_feature_vector(indata, 44100)
    audios_feat.append(feature_vector)


    ### THIS PART SHOULD BE MODIFIED, LOAD SCALER
    scaler = StandardScaler()
    scaler.fit(indata)
    Scaled_test = scaler.transform(indata)
    ###

    print(classes[clf.predict(Scaled_test)])
    
    

# Set the audio parameters
sample_rate = 44100  # Sample rate in Hz
chunk_duration = 1  # 1-second audio chunks
block_size = int(sample_rate * chunk_duration)

# Start streaming audio input with the callback function
with sd.InputStream(callback=audio_callback, blocksize=block_size):
    print(f"Recording audio in {chunk_duration} second chunks. Press Ctrl+C to stop.")
    try:
        sd.sleep(-1)  # Run the audio stream indefinitely
    except KeyboardInterrupt:
        print("\nRecording stopped.")

# Clean up the audio stream
sd.stop()
