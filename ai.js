const tf = require('@tensorflow/tfjs-node');
const mic = require('mic');

const FFT = require('fft-js').fft;
const FFTUtil = require('fft-js').util;

function prepareAudioData(audioBuffer) {
  // Umwandlung der Audiodaten in ein Array von 16-Bit-Integers
  const int16Array = new Int16Array(audioBuffer.buffer);

  // Normalisierung der Daten in den Bereich [-1, 1]
  const float32Array = Float32Array.from(int16Array).map(n => n / 32768);

  // Erstellen eines Tensors für das Modell
  const tensor = tf.tensor([float32Array]);

  // Umformen des Tensors in die erwartete Form
  const reshapedTensor = tensor.reshape([null, 43, 232, 1]); // Ersetzen Sie die Dimensionen durch die tatsächlichen Werte

  return reshapedTensor;
}


async function loadModel() {
  // Lädt das Modell; ersetzen Sie den Pfad durch den tatsächlichen Pfad zu Ihrem Modell
  const model = await tf.loadLayersModel('file://input/model.json');
  return model;
}

async function main() {
  const model = await loadModel();

  const micInstance = mic({
    rate: '44100', // Abtastrate des Audios
    channels: '2', // Anzahl der Kanäle
    debug: false, // Debug-Informationen anzeigen oder nicht
    exitOnSilence: 6 // Beendet die Aufnahme bei 6 Sekunden Stille
  });

  const micInputStream = micInstance.getAudioStream();

  micInputStream.on('data', async (data) => {
  const preparedAudioData = prepareAudioData(data);

  // Vorhersage machen
  const prediction = model.predict(preparedAudioData);
  const predictedClass = prediction.argMax();
  const classId = (await predictedClass.data())[0];

  // Ergebnis ausgeben
  console.log(`Predicted class: ${classId}`);
});

  micInputStream.on('error', (err) => {
    console.error('Error in Mic stream: ', err);
  });

  micInstance.start();
}

main().catch(console.error);
