import pyaudio
import numpy as np
from pydub import AudioSegment


def initialize_pyaudio():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    return p, numdevices


def capture_audio(p, rate, frames_per_buffer, duration=5):
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


def detect_intervals(audio, min_freq=2800, max_freq=3100, threshold_db=75):
    window_size = 50  # ms
    sub_window_size = 50  # ms
    tone_duration = 0
    no_tone_duration = 0
    state = "Tone1"  # Initial state
    pattern_found = False

    for start_time in range(0, len(audio) - window_size, sub_window_size):
        end_time = start_time + window_size
        audio_window = audio[start_time:end_time]
        samples = np.array(audio_window.get_array_of_samples())
        frequencies = np.fft.rfft(samples)
        magnitude = np.abs(frequencies)
        freqs = np.fft.rfftfreq(len(samples), 1 / audio_window.frame_rate)
        condition = (freqs > min_freq) & (freqs < max_freq)
        filtered_magnitude = magnitude[condition]
        max_magnitude = np.max(filtered_magnitude) if np.any(condition) else 0
        loudness_db = 20 * np.log10(max_magnitude + 1e-9)

        if loudness_db >= threshold_db:
            tone_duration += window_size
            no_tone_duration = 0  # Reset the no_tone_duration
        else:
            no_tone_duration += window_size
            tone_duration = 0  # Reset the tone_duration

        print(f"Window {start_time}-{end_time} ms: {'Tone' if loudness_db >= threshold_db else 'No Tone'}, Max magnitude: {max_magnitude}")

        # State machine logic
        if state == "Tone1":
            if tone_duration >= 500:
                state = "Pause1"
                tone_duration = 0  # Reset for the next state
        elif state == "Pause1":
            if no_tone_duration >= 500:
                state = "Tone2"
                no_tone_duration = 0  # Reset for the next state
        elif state == "Tone2":
            if tone_duration >= 500:
                state = "Pause2"
                tone_duration = 0  # Reset for the next state
        elif state == "Pause2":
            if no_tone_duration >= 500:
                pattern_found = True
                break

    print("\nPattern detected in the audio!" if pattern_found else "\nNo pattern detected in the audio.")
    return pattern_found


def main():
    RATE = 44100
    FRAMES_PER_BUFFER = 1024
    try:
        p, numdevices = initialize_pyaudio()
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
                        detect_intervals(audio)
                except KeyboardInterrupt:
                    print("\nExiting.")
                    break
                except Exception as e:
                    print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
