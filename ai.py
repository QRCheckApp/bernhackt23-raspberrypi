import sounddevice as sd
import numpy as np
import tensorflow as tf

# Lade das gespeicherte Keras-Modell
model_path = "./output/"  # Ändern Sie den Pfad entsprechend
model = tf.keras.models.load_model(model_path)

# Überprüfe die erwartete Eingabeform
input_shape = model.layers[0].input_shape
print("Erwartete Eingabeform:", input_shape)

# Sampling-Rate und andere Einstellungen
SAMPLE_RATE = 44100
DURATION = 10  # Dauer des Audiostreams in Sekunden
TARGET_CLASS_INDEX = 0  # Index der Klasse, die erkannt werden soll
FRAME_LENGTH = int(SAMPLE_RATE * 3)  # 0.5 Sekunden Fenster, wie im ursprünglichen Modell

def callback(indata, frames, time, status):
    audio_data = np.squeeze(indata)  # Entfernen der Kanaldimension

    # Schneiden Sie die Daten auf die erwartete Länge (falls erforderlich)
    if len(audio_data) > FRAME_LENGTH:
        audio_data = audio_data[:FRAME_LENGTH]

    # Erweitern der Dimensionen, um sie dem Modell anzupassen
    audio_data = np.expand_dims(audio_data, axis=0)

    # Modellvorhersage
    prediction = model.predict(audio_data)

    # Ausgabe der rohen Modellvorhersage für Debugging-Zwecke
    print("Rohvorhersage:", prediction)

    if np.argmax(prediction) == TARGET_CLASS_INDEX:
        print("!!!!!!!!!Zielklasse erkannt, führe Aktion aus.")

# Starte den Audiostream
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=FRAME_LENGTH):
    print(f"Lausche für {DURATION} Sekunden...")
    sd.sleep(DURATION * 1000)

print("\nBeendet.")
