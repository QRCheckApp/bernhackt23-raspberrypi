import pyaudio
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
from collections import deque
import time
import requests


def initialize_pyaudio():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    return p, numdevices


def capture_audio(p, rate, frames_per_buffer, duration=3):
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=frames_per_buffer)
        frames = [stream.read(frames_per_buffer) for _ in range(int(rate / frames_per_buffer * duration))]
        stream.stop_stream()
        stream.close()
        audio_data = b''.join(frames)
        return AudioSegment(data=audio_data, sample_width=p.get_sample_size(pyaudio.paInt16), frame_rate=rate,
                            channels=1)
    except Exception as e:
        print(f"Error during audio capture: {e}")
        return None


def should_trigger_alarm(last_fifteen_seconds):
    return sum(last_fifteen_seconds) >= 3


def main():
    # Initialize PyAudio
    try:
        p, numdevices = initialize_pyaudio()
    except Exception as e:
        print(f"Error initializing PyAudio: {e}")
        return

    # Load the saved Keras model
    model_path = "./output/"  # Change the path accordingly
    model = tf.keras.models.load_model(model_path)

    # Check the expected input shape
    input_shape = model.layers[0].input_shape
    print("Expected input shape:", input_shape)

    # Sampling rate and other settings
    RATE = 44100
    FRAMES_PER_BUFFER = 1024

    last_fifteen_seconds = deque(maxlen=5)  # Each entry represents 3 seconds, so 5 entries make up 15 seconds

    for i in range(numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0 and "USB PnP Sound Device" in device_info.get('name'):
            print(f"USB audio device found: {device_info.get('name')}. Press Ctrl+C to exit.")
            while True:
                try:
                    audio = capture_audio(p, rate=RATE, frames_per_buffer=FRAMES_PER_BUFFER)
                    if audio:
                        detected = detect_intervals(audio, model)
                        last_fifteen_seconds.append(1 if detected else 0)

                        if should_trigger_alarm(last_fifteen_seconds):
                            last_fifteen_seconds.clear()
                            print("!!!!!!!!!ALARM TRIGGERED!!!!!!!!!")
                            requests.get('https://bernhackt23-backend.web01.dalcloud.net/api/rasp/alarm/z76tuhgb6z7tuhg76zu8th')

                except KeyboardInterrupt:
                    print("\nExiting.")
                    break
                except Exception as e:
                    print(f"\nAn error occurred: {e}")


def detect_intervals(audio, model, target_class_index=0):
    samples = np.array(audio.get_array_of_samples())

    # Trim or pad the audio data to match the input shape
    required_length = 22050  # as expected by the model
    if len(samples) > required_length:
        samples = samples[:required_length]
    elif len(samples) < required_length:
        samples = np.pad(samples, (0, required_length - len(samples)), 'constant')

    samples = np.expand_dims(samples, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(samples)

    # Check if the target class is detected
    if np.argmax(prediction) == target_class_index:
        print("Target class detected.")
        return True
    else:
        print("Target class not detected.")
        return False


if __name__ == "__main__":
    main()
