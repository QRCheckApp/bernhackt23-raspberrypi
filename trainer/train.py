# Generieren des vollständigen Codes für den frame-basierten Ansatz zur Sounderkennung
# mit TensorFlow und Python.

import os
import librosa
import numpy as np
import tensorflow as tf


# Funktion zum Laden der Daten mit einem frame-basierten Ansatz
def load_data(data_directory, sample_rate=44100, frame_length=3, hop_length=0.25):
    X = []
    y = []
    frame_length_samples = int(frame_length * sample_rate)
    hop_length_samples = int(hop_length * sample_rate)

    valid_extensions = ['.wav', '.flac', '.mp3']  # Add or remove extensions as needed

    for label in os.listdir(data_directory):
        class_directory = os.path.join(data_directory, label)
        if os.path.isdir(class_directory):
            for filename in os.listdir(class_directory):
                if not any(filename.endswith(ext) for ext in valid_extensions):
                    continue  # Skip file if it has an invalid extension

                filepath = os.path.join(class_directory, filename)
                try:
                    audio, _ = librosa.load(filepath, sr=sample_rate)
                except Exception as e:
                    print(f"Could not load {filepath}: {e}")
                    continue

                # Laden der Audiodatei
                audio, _ = librosa.load(filepath, sr=sample_rate)

                # Sliding Window-Ansatz
                for i in range(0, len(audio) - frame_length_samples + 1, hop_length_samples):
                    frame = audio[i:i + frame_length_samples]
                    X.append(frame)
                    y.append(label)

                # Falls das letzte Fenster unvollständig ist, schneiden wir es ab
                if len(audio) >= frame_length_samples:
                    last_frame = audio[-frame_length_samples:]
                    X.append(last_frame)
                    y.append(label)

    return np.array(X), np.array(y)


# Modell erstellen
frame_length = 0.5
sample_rate = 44100
input_shape = int(frame_length * sample_rate)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(input_shape,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2)  # Anzahl der Klassen
])

# Modell kompilieren
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Daten laden
data_directory = './data/'  # Ändern Sie dies entsprechend dem Pfad Ihrer Audiodateien
X, y = load_data(data_directory, frame_length=frame_length)

# Konvertieren der Labels in numerische Werte
unique_labels = np.unique(y)
label_to_index = {label: index for index, label in enumerate(unique_labels)}
y_numeric = np.array([label_to_index[label] for label in y])

# Modell trainieren
model.fit(X, y_numeric, epochs=30)

model.save("../output")

