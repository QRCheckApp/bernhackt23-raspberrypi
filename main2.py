import pyaudio
import numpy as np
from pydub import AudioSegment
import tensorflow as tf


def initialize_pyaudio():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    return p, numdevices


def capture_audio(p, rate, frames_per_buffer, duration=1):
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


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    input_shape = model.input_shape
    print(f"Model input shape: {input_shape}")
    return model


def detect_intervals(audio, model):
    # Konvertiert das AudioSegment in ein Numpy-Array
    samples = np.array(audio.get_array_of_samples())
    samples = samples.astype(np.float32) / 32768  # Normalisierung

    # Zuschneiden oder Auffüllen des Arrays, wie zuvor erläutert
    required_size = 43 * 232
    if samples.shape[0] > required_size:
        samples = samples[:required_size]
    elif samples.shape[0] < required_size:
        padding = np.zeros(required_size - samples.shape[0])
        samples = np.concatenate([samples, padding])

    # Formen Sie das Array so um, dass es dem Eingabeformat Ihres Modells entspricht
    input_data = samples.reshape(1, 43, 232, 1)

    # Führt die Vorhersage mit dem Modell aus
    prediction = model.predict(input_data)

    # Debug-Ausgabe
    print(f"Raw prediction output: {prediction}")

    # Interpretiert die Ausgabe
    predicted_class = np.argmax(prediction)

    # Überprüfen, ob die Vorhersage über einem bestimmten Schwellenwert liegt
    confidence = prediction[0][predicted_class]
    if confidence > 0.5:  # Ändern Sie diesen Wert entsprechend
        print(f"Predicted class: {predicted_class} with confidence {confidence}")
    else:
        print("No class recognized.")


def main():
    RATE = 44100
    FRAMES_PER_BUFFER = 1024
    try:
        p, numdevices = initialize_pyaudio()
        model = load_model("./output")  # Ersetzen Sie dies durch den tatsächlichen Pfad zu Ihrem Modell
    except Exception as e:
        print(f"Error initializing PyAudio: {e}")
        return

    for i in range(numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0 and "USB" in device_info.get('name'):
            print(f"USB audio device found: {device_info.get('name')}. Press Ctrl+C to exit.")
            while True:
                try:
                    audio = capture_audio(p, rate=RATE, frames_per_buffer=FRAMES_PER_BUFFER)
                    if audio:
                        detect_intervals(audio, model)
                except KeyboardInterrupt:
                    print("\nExiting.")
                    break
                except Exception as e:
                    print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
