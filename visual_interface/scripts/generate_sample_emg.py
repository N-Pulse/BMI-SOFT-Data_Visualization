"""
Generate a sample EMG data file for testing the simulation mode.
Creates a simple rest-versus-compression pattern you can upload to the app.
"""
import json
import numpy as np


def generate_sample_emg_session(duration_seconds=10, sampling_rate=250, num_channels=1):
    """
    Create a synthetic EMG session with rest and wrist compression bursts.
    """
    num_samples = duration_seconds * sampling_rate
    t = np.linspace(0, duration_seconds, num_samples)
    data = []

    for ch in range(num_channels):
        signal = np.random.randn(num_samples) * 3  # low noise floor

        # Define burst centers as fractions of the total duration
        burst_centers = [0.2, 0.45, 0.7, 0.9]
        burst_duration = int(0.2 * sampling_rate)

        for center in burst_centers:
            start = max(0, int(center * sampling_rate) - burst_duration // 2)
            end = min(num_samples, start + burst_duration)
            window = np.hanning(end - start)
            # Envelope + higher-frequency oscillation to mimic activation
            burst = 60 * window + 25 * np.sin(2 * np.pi * 80 * t[start:end])
            signal[start:end] += burst

        data.append(signal.tolist())

    metadata = {
        "sampling_rate": sampling_rate,
        "num_channels": num_channels,
        "duration_seconds": duration_seconds,
        "num_samples": num_samples,
        "channel_names": [f"CH{i+1}" for i in range(num_channels)],
        "description": "Synthetic EMG (rest + wrist compression bursts) for simulation mode",
    }

    return {"metadata": metadata, "data": data}


if __name__ == "__main__":
    print("Generating sample EMG data...")
    emg_session = generate_sample_emg_session()
    output_file = "sample_emg_data.json"
    with open(output_file, "w") as f:
        json.dump(emg_session, f, indent=2)
    print(f"Sample EMG data saved to {output_file}")
    print(
        f"Duration: {emg_session['metadata']['duration_seconds']} s, "
        f"Channels: {emg_session['metadata']['num_channels']}, "
        f"Samples: {emg_session['metadata']['num_samples']}"
    )
