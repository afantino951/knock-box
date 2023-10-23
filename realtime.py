import sounddevice as sd
import numpy as np

def audio_callback(indata, frames, time, status):
    # Process the audio data here
    print("Processing audio chunk:", indata.shape)

# Set the audio parameters
sample_rate = 44100  # Sample rate in Hz
chunk_duration = 2  # 2-second audio chunks
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
sd.terminate()
