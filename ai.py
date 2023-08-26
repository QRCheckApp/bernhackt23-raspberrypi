import sounddevice as sd
import numpy as np
import tensorflow as tf

# Lade das konvertierte Keras-Modell
model = tf.keras.models.load_model('./output/')

# Überprüfe die erwartete Eingabeform
input_shape = model.layers[0].input_shape
print("Erwartete Eingabeform:", input_shape)

# Sampling-Rate und andere Einstellungen
SAMPLE_RATE = 44100
DURATION = 10  # Dauer des Audiostreams in Sekunden
TARGET_CLASS_INDEX = 1  # Index der Klasse, die erkannt werden soll


def callback(indata, frames, time, status):
    audio_data = np.array(indata).T  # Transponieren, um die Form (Zeitfenster, Frequenzbänder) zu erhalten

    # Ausgabe der Form der Audiodaten für Debugging-Zwecke
    print("Form der Audiodaten vor dem Resampling:", audio_data.shape)

    # Resampling der Audiodaten auf die erwartete Form
    resampled_audio_data = np.resize(audio_data, (43, 232))

    # Ausgabe der Form der Audiodaten nach dem Resampling für Debugging-Zwecke
    print("Form der Audiodaten nach dem Resampling:", resampled_audio_data.shape)

    # Erweitern der Dimensionen, um sie dem Modell anzupassen
    resampled_audio_data = np.expand_dims(resampled_audio_data, axis=[0, -1])  # Fügt Batch- und Kanaldimensionen hinzu

    # Modellvorhersage
    prediction = model.predict(resampled_audio_data)

    # Ausgabe der rohen Modellvorhersage für Debugging-Zwecke
    print("Rohvorhersage:", prediction)

    if prediction.argmax() == TARGET_CLASS_INDEX:
        print("Zielklasse erkannt, führe Aktion aus.")


# Starte den Audiostream
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
    print(f"Lausche für {DURATION} Sekunden...")
    sd.sleep(DURATION * 1000)

print("\nBeendet.")
