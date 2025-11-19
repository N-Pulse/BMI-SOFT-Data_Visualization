"""
Generate a sample EEG data file for testing the simulation mode
This creates realistic synthetic EEG data that can be uploaded to the app
"""
import numpy as np
import json
import os

def generate_realistic_eeg_session(duration_seconds=30, sampling_rate=250, num_channels=8):
    """
    Generate realistic EEG data for a full session
    
    Args:
        duration_seconds: Length of the recording in seconds
        sampling_rate: Sampling frequency (Hz)
        num_channels: Number of EEG channels
    
    Returns:
        Dictionary containing EEG data and metadata
    """
    num_samples = duration_seconds * sampling_rate
    t = np.linspace(0, duration_seconds, num_samples)
    data = []
    
    for ch in range(num_channels):
        # Create channel-specific characteristics
        channel_signal = np.zeros(num_samples)
        
        # Alpha rhythm (8-13 Hz) - relaxed, eyes closed
        alpha_freq = 10 + np.random.randn() * 0.5
        alpha_amplitude = 15 + ch * 2  # Varies by channel
        channel_signal += alpha_amplitude * np.sin(2 * np.pi * alpha_freq * t)
        
        # Beta rhythm (13-30 Hz) - active thinking
        beta_freq = 20 + np.random.randn() * 2
        beta_amplitude = 8 + ch
        channel_signal += beta_amplitude * np.sin(2 * np.pi * beta_freq * t)
        
        # Theta rhythm (4-8 Hz) - drowsiness
        theta_freq = 6 + np.random.randn() * 0.5
        theta_amplitude = 10
        channel_signal += theta_amplitude * np.sin(2 * np.pi * theta_freq * t)
        
        # Delta rhythm (0.5-4 Hz) - deep sleep
        delta_freq = 2 + np.random.randn() * 0.3
        delta_amplitude = 20
        channel_signal += delta_amplitude * np.sin(2 * np.pi * delta_freq * t)
        
        # 1/f pink noise
        pink_noise = np.cumsum(np.random.randn(num_samples)) * 0.3
        channel_signal += pink_noise
        
        # Baseline offset
        baseline = 50 + ch * 15
        channel_signal += baseline
        
        # Add occasional artifacts
        # Eye blink artifacts
        num_blinks = np.random.randint(5, 15)
        for _ in range(num_blinks):
            blink_time = np.random.randint(0, num_samples - 100)
            blink_amplitude = 150 + np.random.randn() * 30
            blink_duration = 50
            blink_shape = np.exp(-((np.arange(num_samples) - blink_time) ** 2) / blink_duration)
            channel_signal += blink_amplitude * blink_shape
        
        # Muscle artifacts
        num_movements = np.random.randint(2, 8)
        for _ in range(num_movements):
            movement_start = np.random.randint(0, num_samples - 500)
            movement_duration = np.random.randint(200, 500)
            movement_noise = np.random.randn(movement_duration) * 30
            channel_signal[movement_start:movement_start+movement_duration] += movement_noise
        
        data.append(channel_signal.tolist())
    
    # Create metadata
    metadata = {
        "sampling_rate": sampling_rate,
        "num_channels": num_channels,
        "duration_seconds": duration_seconds,
        "num_samples": num_samples,
        "channel_names": [f"CH{i+1}" for i in range(num_channels)],
        "description": "Synthetic EEG data for simulation mode testing",
        "generation_date": "2025-11-16"
    }
    
    return {
        "metadata": metadata,
        "data": data
    }

# Generate and save the sample data
if __name__ == "__main__":
    print("Generating sample EEG data...")
    
    # Generate 30 seconds of EEG data
    eeg_session = generate_realistic_eeg_session(duration_seconds=30, sampling_rate=250, num_channels=8)
    
    # Save to JSON file
    output_file = "sample_eeg_data.json"
    with open(output_file, 'w') as f:
        json.dump(eeg_session, f, indent=2)
    
    print(f"Sample EEG data generated successfully!")
    print(f"File saved as: {output_file}")
    print(f"Duration: {eeg_session['metadata']['duration_seconds']} seconds")
    print(f"Channels: {eeg_session['metadata']['num_channels']}")
    print(f"Samples: {eeg_session['metadata']['num_samples']}")
    print(f"Sampling rate: {eeg_session['metadata']['sampling_rate']} Hz")
    print("\nYou can now upload this file in the simulation mode of the EEG app!")
