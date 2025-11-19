import logging
import json
from flask import Flask, render_template, request, jsonify, Response
import threading
import time
from scipy.signal import butter, filtfilt, iirnotch
import numpy as np
from flask_socketio import SocketIO, emit
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
import mne
from datetime import datetime
import io
import zipfile
import tempfile
import os

# Configure logging for detailed information during execution
logging.basicConfig(level=logging.INFO)

# Initialize Flask app and SocketIO for WebSocket support
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# BrainFlow specific settings
params = BrainFlowInputParams() 
params.serial_port = '/dev/spidev0.0'

# Initialize the variables
enabled_channels = 8  # Default to 8 channels enabled
ref_enabled = True  # Default to REF enabled
biasout_enabled = True  # Default to BIASOUT enabled
fs = 250  # Sampling frequency
bandpass_enabled = False
baseline_correction_enabled = False
smoothing_enabled = False

lowcut = 3.0
highcut = 45.0
order = 2

# Set up 8 ch for read data
collected_data = [[] for _ in range(8)]

calibration_values = [0] * 8

spi = None
chip = None
line = None
running = False

simulation_mode = False
simulation_data_file = None
simulation_data_index = 0
simulation_loaded_data = None
current_signal_type = 'eeg'
#####

def cleanup_spi_gpio():
    """Clean up SPI and GPIO resources"""
    global spi, chip, line
    try:
        logging.info("Cleaning up SPI and GPIO...")
        if spi:
            spi.close()
            spi = None
            logging.info("SPI closed.")
        if line:
            line.release()
            line = None
            logging.info("GPIO line released.")
        if chip:
            chip.close()
            chip = None
            logging.info("GPIO chip closed.")
    except Exception as e:
        logging.error(f"SPI and GPIO cleanup error: {e}")

def check_gpio_conflicts():
    """Check if GPIO is already in use"""
    try:
        import gpiod
        # Attempt to open the GPIO line to see if it's already in use
        test_chip = gpiod.Chip('/dev/gpiochip0')
        test_line = test_chip.get_line(26)
        test_line.request(consumer="test", type=gpiod.LINE_REQ_EV_FALLING_EDGE)
        test_line.release()
        test_chip.close()
        return False  # No conflicts
    except Exception:
        return True  # Conflicts detected

def generate_synthetic_eeg_data(num_samples=250, num_channels=8):
    """
    Generate realistic synthetic EEG data for testing without hardware
    
    Args:
        num_samples: Number of samples to generate (default: 250 = 1 second at 250Hz)
        num_channels: Number of EEG channels (default: 8)
    
    Returns:
        numpy array of shape (num_channels, num_samples) with synthetic EEG data
    """
    t = np.linspace(0, num_samples / fs, num_samples)
    data = np.zeros((num_channels, num_samples))
    
    for ch in range(num_channels):
        # Base alpha rhythm (8-13 Hz) - dominant in relaxed state
        alpha_freq = 10 + np.random.randn() * 1
        alpha_component = 20 * np.sin(2 * np.pi * alpha_freq * t)
        
        # Beta rhythm (13-30 Hz) - active thinking
        beta_freq = 20 + np.random.randn() * 3
        beta_component = 10 * np.sin(2 * np.pi * beta_freq * t)
        
        # Theta rhythm (4-8 Hz) - drowsiness
        theta_freq = 6 + np.random.randn() * 1
        theta_component = 15 * np.sin(2 * np.pi * theta_freq * t)
        
        # Add some 1/f noise (pink noise)
        pink_noise = np.cumsum(np.random.randn(num_samples)) * 0.5
        
        # Random baseline drift
        baseline = 50 + ch * 10 + np.random.randn() * 5
        
        # Combine all components
        channel_signal = baseline + alpha_component + beta_component + theta_component + pink_noise
        
        # Occasionally add artifacts (blinks, movements)
        if np.random.rand() > 0.8 and num_samples > 10:
            blink_duration = max(5, min(50, num_samples // 2))
            if num_samples > blink_duration:
                max_start = max(1, num_samples - blink_duration + 1)
                blink_time = np.random.randint(0, max_start)
                blink_window = np.arange(blink_duration) - blink_duration / 2
                blink_shape = np.exp(-(blink_window ** 2) / max(25, blink_duration))
                blink = 100 * blink_shape
                end_idx = blink_time + blink_duration
                channel_signal[blink_time:end_idx] += blink[:end_idx - blink_time]
        
        if np.random.rand() > 0.8 and num_samples > 20:
            movement_duration = max(10, min(num_samples // 2, np.random.randint(200, 500)))
            if num_samples > movement_duration:
                max_start = max(1, num_samples - movement_duration + 1)
                movement_start = np.random.randint(0, max_start)
                movement_noise = np.random.randn(movement_duration) * 30
                end_idx = movement_start + movement_duration
                channel_signal[movement_start:end_idx] += movement_noise[:end_idx - movement_start]
        
        data[ch] = channel_signal
    
    return data

def generate_synthetic_emg_data(num_samples=250, num_channels=8):
    """
    Generate synthetic EMG data, which typically has higher frequency components
    """
    t = np.linspace(0, num_samples / fs, num_samples)
    data = np.zeros((num_channels, num_samples))
    
    for ch in range(num_channels):
        # Simulate bursts of muscle activity (20-150 Hz)
        burst_freq = 80 + np.random.randn() * 20
        burst = 30 * np.sin(2 * np.pi * burst_freq * t)
        
        # Higher frequency noise
        high_freq_noise = np.random.randn(num_samples) * 15
        
        # Low frequency drift
        drift = np.cumsum(np.random.randn(num_samples)) * 0.1
        
        channel_signal = burst + high_freq_noise + drift
        
        # Random activation bursts
        if np.random.rand() > 0.6 and num_samples > 20:
            burst_duration = max(20, min(100, num_samples // 2))
            max_start = max(1, num_samples - burst_duration + 1)
            burst_start = np.random.randint(0, max_start)
            burst_envelope = np.hanning(burst_duration)
            burst_wave = 60 * burst_envelope * np.sin(2 * np.pi * 120 * np.linspace(0, 1, burst_duration))
            end_idx = burst_start + burst_duration
            channel_signal[burst_start:end_idx] += burst_wave[:end_idx - burst_start]
        
        data[ch] = channel_signal
    
    return data

#####

def read_eeg_data_file_simulation():
    """Read EEG data from uploaded file and replay it in real-time"""
    global collected_data, running, simulation_loaded_data, simulation_data_index
    
    if not simulation_loaded_data:
        logging.error("No simulation data loaded")
        socketio.emit('error', {'message': "Please upload an EEG data file first"})
        return
    
    metadata = simulation_loaded_data.get('metadata', {})
    file_sampling_rate = metadata.get('sampling_rate', fs) or fs
    
    try:
        data_arrays = simulation_loaded_data['data']
        num_channels = len(data_arrays)
        
        if num_channels == 0:
            raise ValueError("Uploaded file does not contain channel data")
        
        channel_lengths = [len(ch) for ch in data_arrays if isinstance(ch, list)]
        if not channel_lengths or min(channel_lengths) == 0:
            raise ValueError("Uploaded EEG file contains empty channels")
        
        # Ensure all channels have the same number of samples by truncating to the shortest
        total_samples = min(channel_lengths)
        if total_samples != max(channel_lengths):
            logging.warning("Channel lengths differ. Truncating to %s samples for playback.", total_samples)
        data_arrays = [channel[:total_samples] for channel in data_arrays]
        
        # Determine how much data to send per update (approx. 100 ms of data)
        samples_per_chunk = max(1, int(file_sampling_rate * 0.1))
        samples_per_chunk = min(samples_per_chunk, total_samples)
        
        simulation_data_index = 0  # Restart from the beginning each time analysis starts
        collected_data = [[] for _ in range(num_channels)]
        max_buffer_samples = total_samples
        
        logging.info(
            "Starting FILE SIMULATION mode - replaying uploaded EEG data (%s channels, %s samples/channel)",
            num_channels,
            total_samples
        )
        
        # Precompute filter coefficients if we need them
        if bandpass_enabled:
            b, a = butter(order, [lowcut, highcut], btype='band', fs=file_sampling_rate)
        else:
            b = a = None
        
        while running:
            start_idx = simulation_data_index
            end_idx = start_idx + samples_per_chunk
            chunk_data = []
            
            for channel_data in data_arrays:
                if end_idx <= total_samples:
                    chunk = channel_data[start_idx:end_idx]
                else:
                    wrap = end_idx - total_samples
                    chunk = channel_data[start_idx:] + channel_data[:wrap]
                chunk_data.append(chunk)
            
            simulation_data_index = (simulation_data_index + samples_per_chunk) % total_samples
            
            data_transposed = np.array(chunk_data)
            
            # Apply filters if enabled (same as real data)
            if bandpass_enabled and b is not None and a is not None:
                for channel_idx in range(data_transposed.shape[0]):
                    try:
                        data_transposed[channel_idx] = filtfilt(b, a, data_transposed[channel_idx])
                    except Exception as e:
                        logging.error(f"Error applying filters to channel {channel_idx}: {e}")
            
            # Apply baseline correction if enabled
            if baseline_correction_enabled:
                for idx in range(data_transposed.shape[0]):
                    if idx < len(calibration_values):
                        data_transposed[idx] = data_transposed[idx] - calibration_values[idx]
            
            data_for_frontend = data_transposed.tolist()
            
            # Keep a rolling buffer of the streamed data for exports/analytics
            for idx, channel_data in enumerate(data_for_frontend):
                collected_data[idx].extend(channel_data)
                if len(collected_data[idx]) > max_buffer_samples:
                    collected_data[idx] = collected_data[idx][-max_buffer_samples:]
            
            socketio.emit('eeg_data', {'channels': data_for_frontend})
            
            # Sleep roughly the amount of real time that chunk represents
            time.sleep(samples_per_chunk / file_sampling_rate)
            
    except Exception as e:
        logging.error(f"File simulation error: {e}")
        socketio.emit('error', {'message': f"File simulation error: {str(e)}"})

def read_eeg_data_simulation():
    """Read synthetic EEG data for testing without hardware"""
    global collected_data, running
    
    logging.info("Starting SIMULATION mode - generating synthetic EEG data")
    
    try:
        while running:
            data_transposed = generate_synthetic_eeg_data(num_samples=25, num_channels=enabled_channels)
            
            # Apply filters if enabled (same as real data)
            if bandpass_enabled:
                for channel_idx in range(data_transposed.shape[0]):
                    try:
                        b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
                        data_transposed[channel_idx] = filtfilt(b, a, data_transposed[channel_idx])
                    except Exception as e:
                        logging.error(f"Error applying filters to channel {channel_idx}: {e}")
            
            # Apply baseline correction if enabled
            if baseline_correction_enabled:
                for idx in range(data_transposed.shape[0]):
                    if idx < len(calibration_values):
                        data_transposed[idx] = data_transposed[idx] - calibration_values[idx]
            
            data_for_frontend = data_transposed.tolist()
            
            # Reset collected_data for each new read
            collected_data = [[] for _ in range(enabled_channels)]
            
            for idx, channel_data in enumerate(data_for_frontend):
                collected_data[idx].extend(channel_data)
            
            logging.info(f"Processed {len(data_for_frontend)} channels (SIMULATION)")
            
            socketio.emit('eeg_data', {
                'channels': data_for_frontend
            })
            
            time.sleep(0.1)  # 100ms update rate
            
    except Exception as e:
        logging.error(f"Simulation error: {e}")
        socketio.emit('error', {'message': f"Simulation error: {str(e)}"})

def read_emg_data_simulation():
    """Generate synthetic EMG data for testing"""
    global collected_data, running
    
    logging.info("Starting SIMULATION mode - generating synthetic EMG data")
    
    try:
        while running:
            data_transposed = generate_synthetic_emg_data(num_samples=25, num_channels=enabled_channels)
            
            if bandpass_enabled:
                for channel_idx in range(data_transposed.shape[0]):
                    try:
                        b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
                        data_transposed[channel_idx] = filtfilt(b, a, data_transposed[channel_idx])
                    except Exception as e:
                        logging.error(f"Error applying filters to EMG channel {channel_idx}: {e}")
            
            if baseline_correction_enabled:
                for idx in range(data_transposed.shape[0]):
                    if idx < len(calibration_values):
                        data_transposed[idx] = data_transposed[idx] - calibration_values[idx]
            
            data_for_frontend = data_transposed.tolist()
            collected_data = [[] for _ in range(enabled_channels)]
            for idx, channel_data in enumerate(data_for_frontend):
                collected_data[idx].extend(channel_data)
            
            socketio.emit('eeg_data', {'channels': data_for_frontend})
            time.sleep(0.1)
    except Exception as e:
        logging.error(f"EMG simulation error: {e}")
        socketio.emit('error', {'message': f"EMG simulation error: {str(e)}"})


def read_eeg_data_brainflow():
    """Read EEG data from BrainFlow and emit to frontend"""
    global collected_data, running
    try:
        board = BoardShim(BoardIds.PIEEG_BOARD.value, params)
        board.prepare_session()
        board.start_stream(45000, '')

        while running:
            data = board.get_current_board_data(fs)
            eeg_channels = BoardShim.get_eeg_channels(BoardIds.PIEEG_BOARD.value)
            data_transposed = data[eeg_channels, :]

            logging.info(f"Raw BrainFlow data shape: {data_transposed.shape}")

            if data_transposed.size == 0:
                logging.error("No data retrieved from BrainFlow")
                continue

            if bandpass_enabled:
                for channel_idx in range(data_transposed.shape[0]):
                    try:
                        DataFilter.detrend(data_transposed[channel_idx], DetrendOperations.CONSTANT.value)
                        DataFilter.perform_bandpass(
                            data_transposed[channel_idx], 
                            BoardShim.get_sampling_rate(BoardIds.PIEEG_BOARD.value), 
                            3.0, 45.0, 2, 
                            FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
                        )
                        DataFilter.perform_bandstop(
                            data_transposed[channel_idx], 
                            BoardShim.get_sampling_rate(BoardIds.PIEEG_BOARD.value), 
                            48.0, 52.0, 2, 
                            FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
                        )
                        DataFilter.perform_bandstop(
                            data_transposed[channel_idx], 
                            BoardShim.get_sampling_rate(BoardIds.PIEEG_BOARD.value), 
                            58.0, 62.0, 2, 
                            FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
                        )
                    except Exception as e:
                        logging.error(f"Error applying filters to channel {channel_idx}: {e}")

            if baseline_correction_enabled:
                for idx in range(data_transposed.shape[0]):
                    if idx < len(calibration_values):
                        data_transposed[idx] = data_transposed[idx] - calibration_values[idx]

            if data_transposed.shape[0] > 0:
                ref_channel_index = 0
                ref_values = data_transposed[ref_channel_index]
                ref_mean = np.mean(ref_values)
                ref_std = np.std(ref_values)

                logging.info(f"REF Channel - Mean: {ref_mean}, Std Dev: {ref_std}")

                # Normalize if the mean is significantly higher than expected
                if ref_mean > 1000:
                    data_transposed[ref_channel_index] = (ref_values - ref_mean) / ref_std
                    logging.info(f"Normalized REF Channel")

            # Convert to list for easier processing
            data_list = data_transposed.tolist()
            
            # Reset collected_data for each new read
            collected_data = [[] for _ in range(len(eeg_channels))]
            
            for idx, channel_data in enumerate(data_list):
                collected_data[idx].extend(channel_data)
                
            logging.info(f"Processed {len(data_list)} channels")
            
            socketio.emit('update_data', {
                'raw': [channel[0] if len(channel) > 0 else 0 for channel in data_list]
            })
            time.sleep(0.1)

        board.stop_stream()
        board.release_session()
    except BrainFlowError as e:
        logging.error(f"BrainFlow error: {str(e)}")
        socketio.emit('error', {'message': f"BrainFlow error: {str(e)}"})
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        socketio.emit('error', {'message': f"Unexpected error: {e}"})

def calibrate():
    """Calibrate the EEG channels"""
    global calibration_values
    try:
        logging.info("Starting calibration process")

        if simulation_mode:
            # Simulate calibration with synthetic data
            logging.info("Running SIMULATION calibration")
            calibration_duration = 5  # seconds
            calibration_data = [[] for _ in range(enabled_channels)]
            
            start_time = time.time()
            while time.time() - start_time < calibration_duration:
                data_transposed = generate_synthetic_eeg_data(num_samples=250, num_channels=enabled_channels)
                
                for idx in range(enabled_channels):
                    calibration_data[idx].extend(data_transposed[idx].tolist())
                
                time.sleep(0.1)
            
            calibration_values = [np.mean(ch_data) if len(ch_data) > 0 else 0 for ch_data in calibration_data]
            logging.info(f"Simulated calibration values: {calibration_values}")
            
            return calibration_values
        else:
            board = BoardShim(BoardIds.PIEEG_BOARD.value, params)
            board.prepare_session()
            board.start_stream(45000, '')

            calibration_duration = 5  # seconds
            calibration_data = [[] for _ in range(enabled_channels)]

            start_time = time.time()
            while time.time() - start_time < calibration_duration:
                data = board.get_current_board_data(250)
                eeg_channels = BoardShim.get_eeg_channels(BoardIds.PIEEG_BOARD.value)
                data_transposed = data[eeg_channels, :]

                if data_transposed.size == 0:
                    logging.error("No data retrieved from BrainFlow")
                    continue

                for idx in range(min(data_transposed.shape[0], enabled_channels)):
                    calibration_data[idx].extend(data_transposed[idx].tolist())

            calibration_values = [np.mean(ch_data) if len(ch_data) > 0 else 0 for ch_data in calibration_data]
            logging.info(f"BrainFlow calibration values: {calibration_values}")

            board.stop_stream()
            board.release_session()
        
            return calibration_values

    except BrainFlowError as e:
        logging.error(f"BrainFlow calibration error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected calibration error: {e}")
        raise

def create_csv(data):
    """Create CSV data from EEG data"""
    import csv
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Channel' + str(i+1) for i in range(len(data))])
    for row in zip(*data):
        writer.writerow(row)
    output.seek(0)
    return output.getvalue()

def create_bids_edf(data, subject_id='01', task='resting', run='01'):
    """
    Create BIDS-compliant EDF file with metadata using MNE-Python
    
    Args:
        data: List of channel data arrays
        subject_id: Subject identifier
        task: Task name (e.g., 'resting', 'motorexecution')
        run: Run number
    
    Returns:
        Tuple of (edf_bytes, json_metadata, channels_tsv)
    """
    global fs, enabled_channels
    
    # Prepare channel information
    channel_labels = [f'CH{i+1}' for i in range(len(data))]
    n_channels = len(data)
    
    # Get the minimum length across all channels
    min_length = min(len(ch) for ch in data if len(ch) > 0) if data else 0
    
    if min_length == 0:
        raise ValueError("No data available for export")
    
    # Truncate all channels to minimum length and convert to numpy array
    data_truncated = np.array([ch[:min_length] for ch in data])
    
    info = mne.create_info(
        ch_names=channel_labels,
        sfreq=fs,
        ch_types=['eeg'] * n_channels
    )
    
    raw = mne.io.RawArray(data_truncated * 1e-6, info)  # Convert uV to V for MNE
    
    from datetime import timezone
    raw.set_meas_date(datetime.now(timezone.utc))
    
    temp_dir = tempfile.gettempdir()
    edf_filename = os.path.join(temp_dir, f'sub-{subject_id}_task-{task}_run-{run}_eeg.edf')
    
    try:
        # Export to EDF format
        mne.export.export_raw(edf_filename, raw, fmt='edf', overwrite=True)
        
        # Read the EDF file as bytes
        with open(edf_filename, 'rb') as edf_file:
            edf_bytes = edf_file.read()
        
        # Clean up temporary file
        os.remove(edf_filename)
        
    except Exception as e:
        logging.error(f"Error creating EDF file with MNE: {e}")
        if os.path.exists(edf_filename):
            os.remove(edf_filename)
        raise
    
    # Create BIDS-compliant JSON sidecar metadata
    json_metadata = {
        "TaskName": task,
        "TaskDescription": f"EEG recording during {task} state",
        "Instructions": "Please remain still and relaxed during the recording",
        "InstitutionName": "PiEEG Laboratory",
        "Manufacturer": "PiEEG",
        "ManufacturersModelName": "PiEEG Board",
        "SamplingFrequency": fs,
        "PowerLineFrequency": 50,
        "EEGChannelCount": enabled_channels,
        "EOGChannelCount": 0,
        "ECGChannelCount": 0,
        "EMGChannelCount": 0,
        "MiscChannelCount": 0,
        "TriggerChannelCount": 0,
        "RecordingDuration": len(data_truncated[0]) / fs if len(data_truncated) > 0 else 0,
        "RecordingType": "continuous",
        "EEGReference": "Common average reference" if ref_enabled else "n/a",
        "EEGGround": "BIASOUT" if biasout_enabled else "n/a",
        "EEGPlacementScheme": "Custom",
        "SoftwareFilters": {
            "Bandpass filter": {
                "lower cutoff (Hz)": lowcut,
                "upper cutoff (Hz)": highcut,
                "order": order
            }
        } if bandpass_enabled else "n/a",
        "HardwareFilters": "n/a",
        "SoftwareVersions": "BrainFlow 5.10.1, MNE-Python",
        "SubjectArtefactDescription": "n/a"
    }
    
    # Create channels.tsv content
    channels_tsv = "name\ttype\tunits\tdescription\tstatus\n"
    for i in range(n_channels):
        channels_tsv += f"{channel_labels[i]}\tEEG\tuV\tEEG channel {i+1}\tgood\n"
    
    return edf_bytes, json_metadata, channels_tsv



# Define the main route to serve the web interface
@app.route('/')
def index():
    global current_signal_type
    return render_template('index.html', signal_type=current_signal_type)

@app.route('/toggle-simulation', methods=['POST'])
def toggle_simulation():
    """Toggle between real hardware and simulation mode"""
    global simulation_mode
    data = request.json
    simulation_mode = data.get('simulation', False)
    mode_str = "SIMULATION" if simulation_mode else "HARDWARE"
    logging.info(f"Switched to {mode_str} mode")
    return jsonify({
        "status": f"{mode_str} mode enabled", 
        "mode": "simulation" if simulation_mode else "hardware",
        "simulation_mode": simulation_mode
    })

@app.route('/set-signal-type', methods=['POST'])
def set_signal_type():
    """Switch between EEG and EMG interfaces"""
    global current_signal_type
    data = request.json or {}
    requested_type = data.get('signal_type', 'eeg').lower()
    
    if requested_type not in ('eeg', 'emg'):
        return jsonify({"status": "error", "message": "Invalid signal type"}), 400
    
    current_signal_type = requested_type
    logging.info(f"Switched signal type to {current_signal_type.upper()}")
    
    return jsonify({
        "status": "Signal type updated",
        "signal_type": current_signal_type
    })


@app.route('/start-analysis', methods=['POST'])
def start_analysis():
    """Start EEG/EMG analysis"""
    global running, simulation_mode, simulation_loaded_data, current_signal_type
    running = True

    if simulation_mode:
        if simulation_loaded_data:
            logging.info(f"Starting analysis in FILE SIMULATION mode ({current_signal_type.upper()})")
            threading.Thread(target=read_eeg_data_file_simulation, daemon=True).start()
        else:
            logging.info(f"Starting analysis in SYNTHETIC SIMULATION mode ({current_signal_type.upper()})")
            target = read_emg_data_simulation if current_signal_type == 'emg' else read_eeg_data_simulation
            threading.Thread(target=target, daemon=True).start()
    else:
        if current_signal_type == 'emg':
            logging.warning("EMG hardware mode not supported yet")
            return jsonify({"status": "EMG hardware mode is not supported. Enable simulation mode for EMG."}), 400
        
        logging.info("Starting analysis in HARDWARE mode")
        cleanup_spi_gpio()  # Ensure no conflicts before starting BrainFlow

        if check_gpio_conflicts():
            return jsonify({"status": "GPIO conflict detected. Please resolve before starting BrainFlow."}), 409
        
        threading.Thread(target=read_eeg_data_brainflow, daemon=True).start()
    
    return jsonify({"status": "Analysis started"})
    
###

@app.route('/stop-analysis', methods=['POST'])
def stop_analysis():
    """Stop EEG analysis"""
    global running
    running = False
    time.sleep(1)  # Ensure threads have time to exit
    cleanup_spi_gpio()
    socketio.emit('analysis_stopped')  # Notify frontend to update the settings
    return jsonify({"status": "Analysis stopped"})

###

@app.route('/update-settings', methods=['POST'])
def update_settings():
    """Update analysis settings"""
    global lowcut, highcut, order, baseline_correction_enabled, enabled_channels
    global ref_enabled, biasout_enabled, bandpass_enabled, smoothing_enabled
    
    data = request.json
    lowcut = float(data.get('lowcut', lowcut))
    highcut = float(data.get('highcut', highcut))
    order = int(data.get('order', order))
    baseline_correction_enabled = data.get('baseline_correction_enabled', baseline_correction_enabled)
    enabled_channels = int(data.get('enabled_channels', enabled_channels))
    ref_enabled = data.get('ref_enabled', ref_enabled)
    biasout_enabled = data.get('biasout_enabled', biasout_enabled)
    bandpass_enabled = data.get('bandpass_filter_enabled', bandpass_enabled)
    smoothing_enabled = data.get('smoothing_enabled', smoothing_enabled)
    
    logging.info(f"Updated settings: lowcut={lowcut}, highcut={highcut}, order={order}, "
                 f"baseline_correction_enabled={baseline_correction_enabled}, "
                 f"enabled_channels={enabled_channels}, ref_enabled={ref_enabled}, "
                 f"biasout_enabled={biasout_enabled}, bandpass_enabled={bandpass_enabled}, "
                 f"smoothing_enabled={smoothing_enabled}")
    return jsonify({"status": "Settings updated"})

###

@app.route('/calibrate', methods=['POST'])
def calibrate_route():
    """Calibrate the EEG channels"""
    try:
        values = calibrate()
        return jsonify({"status": "Calibration completed", "values": values})
    except Exception as e:
        logging.error(f"Calibration failed: {e}")
        return jsonify({"status": "Calibration failed", "error": str(e)}), 500

@app.route('/export-data')
def export_data():
    """Export EEG data as BIDS-compliant EDF format with metadata"""
    try:
        # Get parameters from request
        num_rows = int(request.args.get('num_rows', 5000))
        export_format = request.args.get('format', 'edf')  # 'edf' or 'csv'
        subject_id = request.args.get('subject', '01')
        task = request.args.get('task', 'resting')
        run = request.args.get('run', '01')
        
        if not collected_data or len(collected_data[0]) == 0:
            return Response("No data available for export", status=400)
        
        # Limit number of rows
        if num_rows > len(collected_data[0]):
            num_rows = len(collected_data[0])
        
        data_to_export = [ch[:num_rows] for ch in collected_data]
        
        if export_format == 'csv':
            # Legacy CSV export
            csv_data = create_csv(data_to_export)
            return Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': 'attachment;filename=eeg_data.csv'}
            )
        else:
            # BIDS-compliant EDF export (default)
            edf_bytes, json_metadata, channels_tsv = create_bids_edf(
                data_to_export, 
                subject_id=subject_id, 
                task=task, 
                run=run
            )
            
            # Create a ZIP file containing EDF, JSON, and channels.tsv
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add EDF file
                edf_filename = f'sub-{subject_id}_task-{task}_run-{run}_eeg.edf'
                zip_file.writestr(edf_filename, edf_bytes)
                
                # Add JSON sidecar
                json_filename = f'sub-{subject_id}_task-{task}_run-{run}_eeg.json'
                zip_file.writestr(json_filename, json.dumps(json_metadata, indent=2))
                
                # Add channels.tsv
                channels_filename = f'sub-{subject_id}_task-{task}_run-{run}_channels.tsv'
                zip_file.writestr(channels_filename, channels_tsv)
                
                # Add README
                readme_content = f"""# BIDS EEG Dataset

This dataset follows the Brain Imaging Data Structure (BIDS) specification for EEG data.

## Files:
- {edf_filename}: EEG data in European Data Format (EDF+)
- {json_filename}: Metadata describing the recording parameters
- {channels_filename}: Channel information in TSV format

## Specifications:
- Subject: {subject_id}
- Task: {task}
- Run: {run}
- Sampling Frequency: {fs} Hz
- Number of Channels: {len(data_to_export)}
- Recording Duration: {len(data_to_export[0]) / fs:.2f} seconds

For more information about BIDS, visit:
https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html
"""
                zip_file.writestr('README.txt', readme_content)
            
            zip_buffer.seek(0)
            
            return Response(
                zip_buffer.getvalue(),
                mimetype='application/zip',
                headers={
                    'Content-Disposition': f'attachment;filename=sub-{subject_id}_task-{task}_run-{run}_eeg_BIDS.zip'
                }
            )
            
    except Exception as e:
        logging.error(f"Error exporting data: {e}")
        return Response(
            f"Error exporting data: {str(e)}",
            status=500
        )
@app.route('/upload-simulation-data', methods=['POST'])
def upload_simulation_data():
    """Upload EEG data file for simulation mode"""
    global simulation_loaded_data, simulation_data_index
    
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400
        
        # Read and parse JSON data
        file_content = file.read().decode('utf-8')
        data = json.loads(file_content)
        
        # Validate data structure
        if 'data' not in data or 'metadata' not in data:
            return jsonify({"status": "error", "message": "Invalid file format. Expected JSON with 'data' and 'metadata' fields"}), 400
        
        simulation_loaded_data = data
        simulation_data_index = 0
        
        metadata = data['metadata']
        
        logging.info(f"Loaded simulation data: {metadata['num_channels']} channels, {metadata['num_samples']} samples")
        
        return jsonify({
            "status": "success",
            "message": "Simulation data loaded successfully",
            "metadata": metadata
        })
        
    except json.JSONDecodeError:
        return jsonify({"status": "error", "message": "Invalid JSON format"}), 400
    except Exception as e:
        logging.error(f"Error uploading simulation data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

###

# Main entry point of the application
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True, debug=True)
