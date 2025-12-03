# Import the required libraries
import logging
import json
from flask import Flask, render_template, request, jsonify, Response, send_file
import threading
import time
from scipy.signal import butter, filtfilt, iirnotch
import numpy as np
from flask_socketio import SocketIO, emit
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
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
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_interval=20,
    ping_timeout=60
)

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
downsample_factor = 1
current_stream_sampling_rate = fs
features_enabled = False

lowcut = 3.0
highcut = 45.0
order = 2

# Set up 8 ch for read data
collected_data = [[] for _ in range(8)]
collected_data_lock = threading.Lock()

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
recording_active = False
recording_paused = False
recording_data = []
recording_sampling_rate = fs
recording_start_time = None
recording_metadata = {}
event_markers = []
recording_lock = threading.Lock()
event_lock = threading.Lock()
metadata_cache = {}

filter_state = {
    "bandpass": None,
    "notch_50": None,
    "notch_60": None,
    "sampling_rate": fs
}
filter_lock = threading.Lock()

qc_stats = {
    "total_chunks": 0,
    "bad_chunks": 0,
    "bad_channels": 0
}
qc_lock = threading.Lock()
chunk_counter = 0
chunk_counter_lock = threading.Lock()

max_buffer_seconds = 300  # rolling buffer for exports (5 minutes default)
max_record_seconds = 1800  # cap recording buffer to 30 minutes to prevent runaway memory

# BIDS output root; files are written under sub-XX/ses-YY/[eeg|emg|motion]
BIDS_ROOT = os.path.join(os.getcwd(), "bids_output")
BIDS_MODALITY_FOLDERS = {
    "eeg": "eeg",
    "emg": "emg",
    "motion": "motion"
}
BIDS_MODALITY_SUFFIX = {
    "eeg": "eeg",
    "emg": "emg",
    "motion": "motion"
}

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

def compute_chunk_size(sampling_rate):
    """Return samples per ~100 ms chunk"""
    return max(1, int(sampling_rate * 0.1))

def update_filter_coefficients(sampling_rate):
    """Pre-compute filter coefficients based on current settings and sampling rate"""
    global filter_state
    with filter_lock:
        filter_state["sampling_rate"] = sampling_rate
        filter_state["bandpass"] = None
        filter_state["notch_50"] = None
        filter_state["notch_60"] = None
        if bandpass_enabled:
            try:
                filter_state["bandpass"] = butter(order, [lowcut, highcut], btype='band', fs=sampling_rate)
            except Exception as e:
                logging.error(f"Failed to compute bandpass filter: {e}")
        try:
            filter_state["notch_50"] = iirnotch(50, 30, sampling_rate)
            filter_state["notch_60"] = iirnotch(60, 30, sampling_rate)
        except Exception as e:
            logging.error(f"Failed to compute notch filters: {e}")

def apply_filters(data_transposed, sampling_rate):
    """Apply bandpass and notch filters when enabled"""
    with filter_lock:
        bp_coeffs = filter_state.get("bandpass")
        notch_50 = filter_state.get("notch_50")
        notch_60 = filter_state.get("notch_60")
    if bandpass_enabled and bp_coeffs:
        b, a = bp_coeffs
        try:
            min_len = 3 * max(len(a), len(b))
            if data_transposed.shape[1] >= min_len:
                data_transposed = filtfilt(b, a, data_transposed, axis=1)
            else:
                logging.debug("Skipping bandpass: chunk too short for filtfilt")
        except Exception as e:
            logging.error(f"Bandpass filtering failed: {e}")
    if current_signal_type == 'eeg':
        for notch in (notch_50, notch_60):
            if notch:
                b, a = notch
                try:
                    min_len = 3 * max(len(a), len(b))
                    if data_transposed.shape[1] >= min_len:
                        data_transposed = filtfilt(b, a, data_transposed, axis=1)
                except Exception as e:
                    logging.error(f"Notch filtering failed: {e}")
    return data_transposed

def apply_smoothing(data_transposed, window=3):
    """Optional moving average smoothing"""
    if not smoothing_enabled or window <= 1:
        return data_transposed
    try:
        kernel = np.ones(window) / window
        return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=data_transposed)
    except Exception as e:
        logging.error(f"Smoothing failed: {e}")
        return data_transposed

def compute_qc_metrics(chunk, sampling_rate, acquisition_ts, server_ts):
    """Compute quick QC stats per channel"""
    metrics = []
    bad_channels = 0
    if chunk.size == 0:
        return metrics, bad_channels
    means = np.mean(chunk, axis=1)
    stds = np.std(chunk, axis=1)
    ptps = np.ptp(chunk, axis=1)
    for idx in range(chunk.shape[0]):
        mean_val = means[idx]
        std_val = stds[idx]
        ptp_val = ptps[idx]
        bad = bool(abs(mean_val) > 500 or ptp_val > 2000 or ptp_val < 0.5)
        if bad:
            bad_channels += 1
        metrics.append({
            "channel": f"CH{idx + 1}",
            "mean": float(mean_val),
            "std": float(std_val),
            "ptp": float(ptp_val),
            "bad": bad
        })
    payload = {
        "timestamp": server_ts,
        "sampling_rate": sampling_rate,
        "signal_type": current_signal_type,
        "acquisition_timestamp": acquisition_ts,
        "latency_ms": (server_ts - acquisition_ts) * 1000 if acquisition_ts else None,
        "metrics": metrics
    }
    socketio.emit('qc_metrics', payload)
    with qc_lock:
        qc_stats["total_chunks"] += 1
        qc_stats["bad_chunks"] += 1 if bad_channels > 0 else 0
        qc_stats["bad_channels"] += bad_channels
    return metrics, bad_channels

def compute_bandpower(chunk, sfreq, bands):
    """Compute bandpower using a simple Welch estimate"""
    if chunk.size == 0:
        return {name: [0.0] * chunk.shape[0] for name in bands}
    band_results = {name: [] for name in bands}
    try:
        from scipy.signal import welch
        for ch in chunk:
            freqs, psd = welch(ch, sfreq, nperseg=min(len(ch), 256))
            for band_name, (low, high) in bands.items():
                mask = (freqs >= low) & (freqs <= high)
                band_results[band_name].append(float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0)
    except Exception as e:
        logging.error(f"Bandpower computation failed: {e}")
        band_results = {name: [0.0] * chunk.shape[0] for name in bands}
    return band_results

def compute_emg_features(chunk):
    """Compute EMG RMS and MAV per channel"""
    if chunk.size == 0:
        return [0.0] * chunk.shape[0], [0.0] * chunk.shape[0]
    rms = []
    mav = []
    for ch in chunk:
        if len(ch) == 0:
            rms.append(0.0)
            mav.append(0.0)
            continue
        rms.append(float(np.sqrt(np.mean(np.square(ch)))))
        mav.append(float(np.mean(np.abs(ch))))
    return rms, mav

def emit_features(chunk, sampling_rate, acquisition_ts):
    """Emit EEG/EMG features when enabled"""
    if not features_enabled:
        return
    server_ts = time.time()
    payload = {
        "timestamp": server_ts,
        "sampling_rate": sampling_rate,
        "signal_type": current_signal_type
    }
    if current_signal_type == 'motion':
        # Motion features not implemented; skip to reduce noise
        payload["features"] = {}
    elif current_signal_type == 'eeg':
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30)
        }
        bandpower = compute_bandpower(chunk, sampling_rate, bands)
        payload["features"] = {"bands": bandpower}
    else:
        rms, mav = compute_emg_features(chunk)
        payload["features"] = {"rms": rms, "mav": mav}
    payload["acquisition_timestamp"] = acquisition_ts
    payload["latency_ms"] = (server_ts - acquisition_ts) * 1000 if acquisition_ts else None
    socketio.emit('features', payload)

def apply_baseline_correction(data_transposed):
    """Subtract calibration baseline when enabled"""
    if not baseline_correction_enabled:
        return data_transposed
    try:
        for idx in range(min(len(calibration_values), data_transposed.shape[0])):
            data_transposed[idx] = data_transposed[idx] - calibration_values[idx]
    except Exception as e:
        logging.error(f"Baseline correction failed: {e}")
    return data_transposed

def process_and_emit_chunk(data_transposed, sampling_rate, acquisition_ts=None):
    """Common pipeline for all acquisition paths"""
    if data_transposed is None or data_transposed.size == 0:
        return
    acquisition_ts = acquisition_ts or time.time()
    try:
        processed = apply_filters(data_transposed, sampling_rate)
        processed = apply_baseline_correction(processed)
        processed = apply_smoothing(processed)
        processed, effective_rate = apply_downsampling(processed, sampling_rate)

        data_for_frontend = processed.tolist()
        update_collected_buffer(data_for_frontend, effective_rate)
        buffer_recording_data(data_for_frontend, effective_rate)

        server_ts = time.time()
        socketio.emit('eeg_data', {
            'channels': data_for_frontend,
            'sampling_rate': effective_rate,
            'signal_type': current_signal_type,
            'timestamp': server_ts,
            'acquisition_timestamp': acquisition_ts,
            'latency_ms': (server_ts - acquisition_ts) * 1000 if acquisition_ts else None,
            'downsample_factor': downsample_factor
        })
        compute_qc_metrics(processed, effective_rate, acquisition_ts, server_ts)
        emit_features(processed, effective_rate, acquisition_ts)
        with chunk_counter_lock:
            global chunk_counter
            chunk_counter += 1
            if chunk_counter % 20 == 0:
                logging.info(f"Streamed {chunk_counter} chunks @ {effective_rate} Hz (signal={current_signal_type})")
    except Exception as e:
        logging.error(f"Chunk processing error: {e}")
        raise

def apply_downsampling(data_transposed, sampling_rate):
    """Downsample data and return the effective sampling rate"""
    global downsample_factor, current_stream_sampling_rate
    factor = max(1, int(downsample_factor))
    if factor > 1 and data_transposed.shape[1] > 0:
        data_transposed = data_transposed[:, ::factor]
    effective_rate = sampling_rate / factor if factor > 0 else sampling_rate
    current_stream_sampling_rate = effective_rate
    return data_transposed, effective_rate

def buffer_recording_data(chunk, sampling_rate):
    """Append streamed samples to the recording buffer when enabled"""
    global recording_data, recording_sampling_rate
    if not recording_active or recording_paused:
        return
    if not chunk:
        return
    with recording_lock:
        if not recording_data or len(recording_data) != len(chunk):
            recording_data = [[] for _ in range(len(chunk))]
        for idx, channel_data in enumerate(chunk):
            recording_data[idx].extend(channel_data)
            max_samples = int(sampling_rate * max_record_seconds)
            if len(recording_data[idx]) > max_samples:
                recording_data[idx] = recording_data[idx][-max_samples:]
        recording_sampling_rate = sampling_rate

def update_collected_buffer(data_for_frontend, sampling_rate):
    """Keep rolling buffer for exports"""
    global collected_data
    if not data_for_frontend:
        return
    with collected_data_lock:
        if not collected_data or len(collected_data) != len(data_for_frontend):
            collected_data = [[] for _ in range(len(data_for_frontend))]
        max_samples = int(sampling_rate * max_buffer_seconds)
        for idx, channel_data in enumerate(data_for_frontend):
            collected_data[idx].extend(channel_data)
            if len(collected_data[idx]) > max_samples:
                collected_data[idx] = collected_data[idx][-max_samples:]

def reset_recording_state():
    """Reset recording-related state"""
    global recording_active, recording_paused, recording_data, recording_start_time
    global event_markers, recording_metadata, recording_sampling_rate
    with recording_lock, event_lock:
        recording_active = False
        recording_paused = False
        recording_data = []
        recording_start_time = None
        event_markers = []
        recording_metadata = {}
        recording_sampling_rate = current_stream_sampling_rate

def create_events_tsv(events):
    """Create BIDS-friendly events.tsv content from recorded markers"""
    lines = ["onset\tduration\ttrial_type\tdescription\n"]
    for evt in events:
        onset = evt.get('offset', 0)
        label = evt.get('label', 'marker')
        description = evt.get('description', '')
        lines.append(f"{onset:.3f}\t0\t{label}\t{description}\n")
    return "".join(lines)

def reset_qc_stats():
    """Reset QC counters"""
    with qc_lock:
        qc_stats["total_chunks"] = 0
        qc_stats["bad_chunks"] = 0
        qc_stats["bad_channels"] = 0


def bids_label(value, fallback):
    """Return a non-empty BIDS label"""
    label = str(value).strip() if value is not None else ""
    return label or fallback


def bids_cache_key(subject_id, session_id, task, run, modality):
    """Stable key for metadata cache"""
    return f"sub-{subject_id}_ses-{session_id}_task-{task}_run-{run}_{modality}"


def ensure_bids_dirs(subject_id, session_id, modality):
    """Ensure BIDS directory structure exists"""
    modality_folder = BIDS_MODALITY_FOLDERS.get(modality, modality)
    modality_dir = os.path.join(
        BIDS_ROOT,
        f"sub-{subject_id}",
        f"ses-{session_id}",
        modality_folder
    )
    os.makedirs(modality_dir, exist_ok=True)
    return modality_dir


def bids_basename(subject_id, session_id, task, run, modality):
    """BIDS-compliant base filename (without extension)"""
    suffix = BIDS_MODALITY_SUFFIX.get(modality, modality)
    return f"sub-{subject_id}_ses-{session_id}_task-{task}_run-{run}_{suffix}"


def bids_core_prefix(subject_id, session_id, task, run):
    """Filename prefix without modality suffix (used for channels/events)"""
    return f"sub-{subject_id}_ses-{session_id}_task-{task}_run-{run}"


def coerce_metadata_value(value):
    """Try to coerce stringified numbers/booleans into native types"""
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def get_default_metadata_fields(modality, sampling_rate, n_channels, task_label=None, subject=None, session=None, run=None):
    """Default metadata rows for the metadata editor"""
    base_fields = [
        # Recording identifiers
        {"key": "Subject", "value": subject or "01", "include": True},
        {"key": "Session", "value": session or "01", "include": True},
        {"key": "TaskName", "value": task_label or "rest", "include": True},
        {"key": "Run", "value": run or "01", "include": True},
        {"key": "TaskDescription", "value": f"Signal recording during {task_label or 'rest'}", "include": True},
        {"key": "Instructions", "value": "Please remain still and relaxed during the recording", "include": True},
        {"key": "InstitutionName", "value": "PiEEG Laboratory", "include": True},
        # Subject info
        {"key": "SubjectAge", "value": "", "include": True},
        {"key": "SubjectSex", "value": "", "include": True},
        {"key": "SubjectHandedness", "value": "", "include": True},
        {"key": "SubjectPosition", "value": "sitting", "include": True},
        # Session info
        {"key": "SessionDate", "value": "", "include": True},
        {"key": "SessionDuration", "value": "", "include": True},
        # Device info
        {"key": "Manufacturer", "value": "PiEEG", "include": True},
        {"key": "ManufacturersModelName", "value": "PiEEG Board", "include": True},
        {"key": "DeviceFirmware", "value": "", "include": True},
        {"key": "CommunicationProtocol", "value": "SPI", "include": True},
        {"key": "SamplingFrequency", "value": sampling_rate, "include": True},
        {"key": "PowerLineFrequency", "value": 50, "include": True},
        {"key": "SoftwareVersions", "value": "BrainFlow, MNE-Python", "include": True},
        # Signal and channels
        {"key": "RecordingType", "value": "continuous", "include": True},
        {"key": "SignalType", "value": modality, "include": True},
        {"key": "NumberOfChannels", "value": n_channels, "include": True},
        # Environment
        {"key": "EnvironmentLocation", "value": "Campus Biotech", "include": True},
        {"key": "EnvironmentNoise", "value": "quiet", "include": True},
        {"key": "EnvironmentTemperature", "value": "", "include": True},
        # Filters
        {"key": "FilterHighPass", "value": lowcut, "include": True},
        {"key": "FilterLowPass", "value": highcut, "include": True},
        {"key": "FilterOrder", "value": order, "include": True}
    ]
    if modality == "eeg":
        base_fields.extend([
            {"key": "EEGChannelCount", "value": n_channels, "include": True},
            {"key": "EEGReference", "value": "Common average reference" if ref_enabled else "n/a", "include": True},
            {"key": "EEGGround", "value": "BIASOUT" if biasout_enabled else "n/a", "include": True},
            {"key": "EEGPlacementScheme", "value": "Custom", "include": True}
        ])
    elif modality == "emg":
        base_fields.extend([
            {"key": "EMGChannelCount", "value": n_channels, "include": True},
            {"key": "RecordingDevice", "value": "Surface EMG", "include": True}
        ])
    else:
        base_fields.extend([
            {"key": "MotionChannelCount", "value": n_channels, "include": True},
            {"key": "RecordingDevice", "value": "IMU / synthetic motion", "include": True}
        ])
    return base_fields


def load_metadata_fields(subject_id, session_id, task, run, modality, sampling_rate, n_channels):
    """Load metadata rows from cache, UI file, or defaults"""
    key = bids_cache_key(subject_id, session_id, task, run, modality)
    if key in metadata_cache:
        return metadata_cache[key]

    modality_dir = ensure_bids_dirs(subject_id, session_id, modality)
    base = bids_basename(subject_id, session_id, task, run, modality)
    ui_path = os.path.join(modality_dir, f"{base}_metadata_ui.json")
    sidecar_path = os.path.join(modality_dir, f"{base}.json")
    fields = []

    try:
        if os.path.exists(ui_path):
            with open(ui_path, "r") as f:
                payload = json.load(f)
            fields = payload.get("fields", [])
        elif os.path.exists(sidecar_path):
            with open(sidecar_path, "r") as f:
                payload = json.load(f)
            fields = [{"key": k, "value": v, "include": True} for k, v in payload.items()]
    except Exception as exc:
        logging.error(f"Failed to load metadata for {base}: {exc}")

    defaults = get_default_metadata_fields(
        modality,
        sampling_rate,
        n_channels,
        task_label=task,
        subject=subject_id,
        session=session_id,
        run=run
    )
    fields = merge_metadata_with_defaults(fields, defaults, subject_id, session_id, task, run, sampling_rate, modality, n_channels)

    metadata_cache[key] = fields
    return fields


def merge_metadata_with_defaults(existing, defaults, subject_id, session_id, task, run, sampling_rate, modality, n_channels):
    """Merge persisted metadata with default rows, updating key session fields from current context"""
    # Normalize existing
    normalized = []
    seen = set()
    for row in existing or []:
        key = row.get("key")
        if not key:
            continue
        seen.add(key)
        normalized.append({
            "key": key,
            "value": row.get("value"),
            "include": row.get("include", True)
        })

    # Add missing defaults
    for row in defaults or []:
        key = row.get("key")
        if not key or key in seen:
            continue
        normalized.append(row)

    # Update context-bound keys to reflect current session
    context_updates = {
        "Subject": subject_id,
        "Session": session_id,
        "TaskName": task,
        "Run": run,
        "SamplingFrequency": sampling_rate,
        "SignalType": modality,
        "NumberOfChannels": n_channels
    }
    for row in normalized:
        key = row.get("key")
        if key in context_updates:
            row["value"] = context_updates[key]
    return normalized


def persist_metadata_fields(fields, subject_id, session_id, task, run, modality):
    """Persist metadata UI selections and sidecar (only included fields)"""
    modality_dir = ensure_bids_dirs(subject_id, session_id, modality)
    base = bids_basename(subject_id, session_id, task, run, modality)
    ui_path = os.path.join(modality_dir, f"{base}_metadata_ui.json")
    sidecar_path = os.path.join(modality_dir, f"{base}.json")

    filtered = {}
    for row in fields:
        key = row.get("key")
        include = row.get("include", True)
        if not key or not include:
            continue
        filtered[key] = coerce_metadata_value(row.get("value"))

    try:
        with open(ui_path, "w") as f:
            json.dump({"fields": fields}, f, indent=2)
        with open(sidecar_path, "w") as f:
            json.dump(filtered, f, indent=2)
    except Exception as exc:
        logging.error(f"Failed to persist metadata for {base}: {exc}")

    key = bids_cache_key(subject_id, session_id, task, run, modality)
    metadata_cache[key] = fields
    return filtered, sidecar_path

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


def generate_synthetic_motion_data(num_samples=250, num_channels=3):
    """
    Generate simple motion/IMU-style data (x, y, z) for testing
    """
    t = np.linspace(0, num_samples / fs, num_samples)
    data = np.zeros((num_channels, num_samples))

    for ch in range(num_channels):
        drift = np.cumsum(np.random.randn(num_samples)) * 0.05
        periodic = 0.5 * np.sin(2 * np.pi * (0.5 + ch * 0.2) * t)
        spikes = np.zeros(num_samples)
        for _ in range(np.random.randint(1, 4)):
            center = np.random.randint(0, num_samples)
            width = np.random.randint(10, 40)
            start = max(0, center - width // 2)
            end = min(num_samples, center + width // 2)
            spikes[start:end] += np.hanning(end - start) * np.random.uniform(1, 3)
        data[ch] = drift + periodic + spikes + np.random.randn(num_samples) * 0.1

    return data

def generate_sample_emg(sampling_rate=250, duration_seconds=5):
    """Generate a simple EMG sample with rest and wrist compression bursts"""
    num_channels = 1
    total_samples = int(sampling_rate * duration_seconds)
    t = np.linspace(0, duration_seconds, total_samples)
    data = np.zeros((num_channels, total_samples))
    
    # Rest: small noise floor
    noise = np.random.randn(total_samples) * 5
    
    # Wrist compression bursts at selected times
    burst_centers = [duration_seconds * 0.2, duration_seconds * 0.45, duration_seconds * 0.7, duration_seconds * 0.9]
    burst_duration = int(0.15 * sampling_rate)
    for center in burst_centers:
        start = max(0, int(center * sampling_rate) - burst_duration // 2)
        end = min(total_samples, start + burst_duration)
        burst = np.hanning(end - start) * 80
        noise[start:end] += burst
        # add higher frequency components
        noise[start:end] += 20 * np.sin(2 * np.pi * 80 * t[start:end])
    
    data[0] = noise
    metadata = {
        "num_channels": num_channels,
        "num_samples": total_samples,
        "sampling_rate": sampling_rate,
        "duration_seconds": duration_seconds,
        "signal_type": "emg",
        "description": "Sample EMG with rest and wrist compression bursts"
    }
    return {"metadata": metadata, "data": data.tolist()}

def read_eeg_data_file_simulation():
    """Read EEG data from uploaded file and replay it in real-time"""
    global collected_data, running, simulation_loaded_data, simulation_data_index, current_stream_sampling_rate
    
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
        samples_per_chunk = min(total_samples, compute_chunk_size(file_sampling_rate))
        
        simulation_data_index = 0  # Restart from the beginning each time analysis starts
        with collected_data_lock:
            collected_data = [[] for _ in range(num_channels)]
        
        logging.info(
            "Starting FILE SIMULATION mode - replaying uploaded EEG data (%s channels, %s samples/channel)",
            num_channels,
            total_samples
        )
        
        update_filter_coefficients(file_sampling_rate)
        current_stream_sampling_rate = file_sampling_rate / downsample_factor if downsample_factor > 0 else file_sampling_rate
        
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
            acquisition_ts = time.time()
            process_and_emit_chunk(data_transposed, file_sampling_rate, acquisition_ts=acquisition_ts)
            
            # Sleep roughly the amount of real time that chunk represents
            time.sleep(samples_per_chunk / file_sampling_rate)
            
    except Exception as e:
        logging.error(f"File simulation error: {e}")
        running = False
        socketio.emit('error', {'message': f"File simulation error: {str(e)}"})

def read_eeg_data_simulation():
    """Read synthetic EEG data for testing without hardware"""
    global collected_data, running, current_stream_sampling_rate
    
    logging.info("Starting SIMULATION mode - generating synthetic EEG data")
    update_filter_coefficients(fs)
    current_stream_sampling_rate = fs / downsample_factor if downsample_factor > 0 else fs
    samples_per_chunk = compute_chunk_size(fs)
    
    try:
        while running:
            data_transposed = generate_synthetic_eeg_data(num_samples=samples_per_chunk, num_channels=enabled_channels)
            acquisition_ts = time.time()
            process_and_emit_chunk(data_transposed, fs, acquisition_ts=acquisition_ts)
            time.sleep(samples_per_chunk / fs)
            
    except Exception as e:
        logging.error(f"Simulation error: {e}")
        running = False
        socketio.emit('error', {'message': f"Simulation error: {str(e)}"})

def read_emg_data_simulation():
    """Generate synthetic EMG data for testing"""
    global collected_data, running, current_stream_sampling_rate
    
    logging.info("Starting SIMULATION mode - generating synthetic EMG data")
    update_filter_coefficients(fs)
    current_stream_sampling_rate = fs / downsample_factor if downsample_factor > 0 else fs
    samples_per_chunk = compute_chunk_size(fs)
    
    try:
        while running:
            data_transposed = generate_synthetic_emg_data(num_samples=samples_per_chunk, num_channels=enabled_channels)
            acquisition_ts = time.time()
            process_and_emit_chunk(data_transposed, fs, acquisition_ts=acquisition_ts)
            time.sleep(samples_per_chunk / fs)
    except Exception as e:
        logging.error(f"EMG simulation error: {e}")
        running = False
        socketio.emit('error', {'message': f"EMG simulation error: {str(e)}"})


def read_motion_data_simulation():
    """Generate synthetic motion data for testing"""
    global collected_data, running, current_stream_sampling_rate

    logging.info("Starting SIMULATION mode - generating synthetic motion data")
    update_filter_coefficients(fs)
    current_stream_sampling_rate = fs / downsample_factor if downsample_factor > 0 else fs
    samples_per_chunk = compute_chunk_size(fs)

    try:
        while running:
            data_transposed = generate_synthetic_motion_data(num_samples=samples_per_chunk, num_channels=3)
            acquisition_ts = time.time()
            process_and_emit_chunk(data_transposed, fs, acquisition_ts=acquisition_ts)
            time.sleep(samples_per_chunk / fs)
    except Exception as e:
        logging.error(f"Motion simulation error: {e}")
        running = False
        socketio.emit('error', {'message': f"Motion simulation error: {str(e)}"})

def read_eeg_data_brainflow():
    """Read EEG data from BrainFlow and emit to frontend"""
    global collected_data, running, current_stream_sampling_rate
    board = None
    try:
        board = BoardShim(BoardIds.PIEEG_BOARD.value, params)
        board.prepare_session()
        hw_rate = BoardShim.get_sampling_rate(BoardIds.PIEEG_BOARD.value)
        update_filter_coefficients(hw_rate)
        board.start_stream(45000, '')
        current_stream_sampling_rate = hw_rate / downsample_factor if downsample_factor > 0 else hw_rate
        samples_per_chunk = compute_chunk_size(hw_rate)
        logging.info(f"BrainFlow stream started at {hw_rate} Hz, chunk {samples_per_chunk} samples")

        while running:
            acquisition_ts = time.time()
            data = board.get_current_board_data(samples_per_chunk)
            eeg_channels = BoardShim.get_eeg_channels(BoardIds.PIEEG_BOARD.value)
            data_transposed = data[eeg_channels, :]

            if data_transposed.size == 0:
                logging.warning("No data retrieved from BrainFlow")
                time.sleep(samples_per_chunk / hw_rate)
                continue

            process_and_emit_chunk(data_transposed, hw_rate, acquisition_ts=acquisition_ts)
            time.sleep(samples_per_chunk / hw_rate)

    except BrainFlowError as e:
        logging.error(f"BrainFlow error: {str(e)}")
        running = False
        socketio.emit('error', {'message': f"BrainFlow error: {str(e)}"})
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        running = False
        socketio.emit('error', {'message': f"Unexpected error: {str(e)}"})
    finally:
        try:
            if board is not None:
                board.stop_stream()
                board.release_session()
                logging.info("BrainFlow stream stopped and session released")
        except Exception as e:
            logging.error(f"Error stopping BrainFlow session: {e}")


def calibrate():
    """Calibrate the EEG channels"""
    global calibration_values, simulation_mode
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
            # Real hardware calibration
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
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Channel' + str(i+1) for i in range(len(data))])
    for row in zip(*data):
        writer.writerow(row)
    output.seek(0)
    return output.getvalue()


def create_bids_edf(data, subject_id='01', task='resting', run='01', sampling_rate=None, signal_type='eeg'):
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
    
    sampling_rate = sampling_rate or fs
    if signal_type == 'emg':
        channel_type = 'emg'
        scale_factor = 1e-6
    elif signal_type == 'motion':
        channel_type = 'misc'
        scale_factor = 1.0
    else:
        channel_type = 'eeg'
        scale_factor = 1e-6
    info = mne.create_info(ch_names=channel_labels, sfreq=sampling_rate, ch_types=[channel_type] * n_channels)
    
    raw = mne.io.RawArray(data_truncated * scale_factor, info)
    
    from datetime import timezone
    raw.set_meas_date(datetime.now(timezone.utc))
    
    temp_dir = tempfile.gettempdir()
    modality = 'emg' if signal_type == 'emg' else 'motion' if signal_type == 'motion' else 'eeg'
    edf_filename = os.path.join(temp_dir, f'sub-{subject_id}_task-{task}_run-{run}_{modality}.edf')
    
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
        "TaskDescription": f"Signal recording during {task} state",
        "Instructions": "Please remain still and relaxed during the recording",
        "InstitutionName": "PiEEG Laboratory",
        "Manufacturer": "PiEEG",
        "ManufacturersModelName": "PiEEG Board",
        "SamplingFrequency": sampling_rate,
        "PowerLineFrequency": 50,
        "EEGChannelCount": n_channels if signal_type == 'eeg' else 0,
        "EOGChannelCount": 0,
        "ECGChannelCount": 0,
        "EMGChannelCount": n_channels if signal_type == 'emg' else 0,
        "MotionChannelCount": n_channels if signal_type == 'motion' else 0,
        "MiscChannelCount": 0,
        "TriggerChannelCount": 0,
        "RecordingDuration": len(data_truncated[0]) / sampling_rate if len(data_truncated) > 0 else 0,
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
        "SubjectArtefactDescription": "n/a",
        "DownsampleFactor": downsample_factor
    }
    
    # Create channels.tsv content
    channels_tsv = "name\ttype\tunits\tdescription\tstatus\n"
    for i in range(n_channels):
        if signal_type == 'emg':
            ch_type_label = "EMG"
            units = "uV"
        elif signal_type == 'motion':
            ch_type_label = "MISC"
            units = "a.u."
        else:
            ch_type_label = "EEG"
            units = "uV"
        channels_tsv += f"{channel_labels[i]}\t{ch_type_label}\t{units}\t{ch_type_label} channel {i+1}\tgood\n"
    
    return edf_bytes, json_metadata, channels_tsv


def save_bids_recording(data, markers, subject_id, session_id, task, run, sampling_rate, modality, metadata_fields):
    """
    Write BIDS dataset (EDF/JSON/TSV) to disk for the selected modality
    """
    modality = modality or "eeg"
    n_channels = len(data)
    base_core = bids_core_prefix(subject_id, session_id, task, run)
    base_with_suffix = bids_basename(subject_id, session_id, task, run, modality)
    modality_dir = ensure_bids_dirs(subject_id, session_id, modality)

    # Prepare sidecar metadata from UI selections
    included_fields = {}
    for row in metadata_fields or []:
        key = row.get("key")
        if not key or not row.get("include", True):
            continue
        included_fields[key] = coerce_metadata_value(row.get("value"))

    edf_bytes, json_metadata, channels_tsv = create_bids_edf(
        data,
        subject_id=subject_id,
        task=task,
        run=run,
        sampling_rate=sampling_rate,
        signal_type=modality
    )

    # Enrich metadata with runtime/session info
    duration = len(data[0]) / sampling_rate if data and data[0] else 0
    excluded_keys = {row.get("key") for row in metadata_fields or [] if row.get("include") is False}
    for key in list(json_metadata.keys()):
        if key in excluded_keys:
            json_metadata.pop(key, None)
    json_metadata.update(included_fields)
    json_metadata["TaskName"] = task
    json_metadata["SamplingFrequency"] = sampling_rate
    json_metadata["Session"] = session_id
    json_metadata["RecordingDuration"] = duration
    json_metadata["RecordingStartTime"] = recording_start_time
    json_metadata["RecordingPaused"] = recording_paused
    json_metadata["MarkerCount"] = len(markers)
    json_metadata["Modality"] = modality
    with qc_lock:
        json_metadata["QualitySummary"] = {
            "TotalChunks": qc_stats["total_chunks"],
            "BadChunks": qc_stats["bad_chunks"],
            "BadChannels": qc_stats["bad_channels"]
        }
    json_metadata["FeaturesEnabled"] = features_enabled

    # Paths
    edf_path = os.path.join(modality_dir, f"{base_with_suffix}.edf")
    json_path = os.path.join(modality_dir, f"{base_with_suffix}.json")
    channels_path = os.path.join(modality_dir, f"{base_core}_channels.tsv")
    events_path = os.path.join(modality_dir, f"{base_core}_events.tsv")

    try:
        with open(edf_path, "wb") as f:
            f.write(edf_bytes)
        with open(json_path, "w") as f:
            json.dump(json_metadata, f, indent=2)
        with open(channels_path, "w") as f:
            f.write(channels_tsv)
        if markers:
            with open(events_path, "w") as f:
                f.write(create_events_tsv(markers))
        else:
            events_path = None
    except Exception as exc:
        logging.error(f"Failed to write BIDS files for {base_with_suffix}: {exc}")
        raise

    return {
        "edf": edf_path,
        "json": json_path,
        "channels": channels_path,
        "events": events_path
    }


# Define the main route to serve the web interface
@app.route('/')
def index():
    global current_signal_type
    return render_template('index.html', signal_type=current_signal_type)


@app.route('/toggle-simulation', methods=['POST'])
def toggle_simulation():
    """Toggle between real hardware and simulation mode"""
    global simulation_mode
    data = request.json or {}
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
    """Switch between EEG/EMG/motion interfaces (UI modality selector)"""
    global current_signal_type
    data = request.json or {}
    requested_type = data.get('signal_type', 'eeg').lower()
    
    if requested_type not in ('eeg', 'emg', 'motion'):
        return jsonify({"status": "error", "message": "Invalid signal type"}), 400
    
    current_signal_type = requested_type
    logging.info(f"Switched signal type to {current_signal_type.upper()}")
    update_filter_coefficients(fs)
    
    return jsonify({
        "status": "Signal type updated",
        "signal_type": current_signal_type
    })


@app.route('/start-analysis', methods=['POST'])
def start_analysis():
    """Start EEG/EMG analysis"""
    global running, simulation_mode, simulation_loaded_data, current_signal_type, current_stream_sampling_rate
    if running:
        logging.info("Analysis already running")
        return jsonify({"status": "Analysis already running"})
    logging.info(f"/start-analysis request received (simulation={simulation_mode}, signal={current_signal_type})")
    running = True
    reset_qc_stats()
    with collected_data_lock:
        channel_count = 3 if current_signal_type == 'motion' else enabled_channels
        collected_data[:] = [[] for _ in range(channel_count)]
    current_stream_sampling_rate = fs / downsample_factor if downsample_factor > 0 else fs
    mode_label = "SIMULATION" if simulation_mode else "HARDWARE"
    logging.info(f"Starting analysis ({mode_label}, signal={current_signal_type.upper()}, fs={fs}Hz)")
    
    if simulation_mode:
        if simulation_loaded_data:
            logging.info(f"Starting analysis in FILE SIMULATION mode ({current_signal_type.upper()})")
            socketio.start_background_task(read_eeg_data_file_simulation)
        else:
            logging.info(f"Starting analysis in SYNTHETIC SIMULATION mode ({current_signal_type.upper()})")
            if current_signal_type == 'emg':
                target = read_emg_data_simulation
            elif current_signal_type == 'motion':
                target = read_motion_data_simulation
            else:
                target = read_eeg_data_simulation
            socketio.start_background_task(target)
    else:
        if current_signal_type in ('emg', 'motion'):
            logging.warning(f"{current_signal_type.upper()} hardware mode not supported yet")
            return jsonify({"status": f"{current_signal_type.upper()} hardware mode is not supported. Enable simulation mode."}), 400
        
        logging.info("Starting analysis in HARDWARE mode")
        cleanup_spi_gpio()  # Ensure no conflicts before starting BrainFlow

        if check_gpio_conflicts():
            return jsonify({"status": "GPIO conflict detected. Please resolve before starting BrainFlow."}), 409
        socketio.start_background_task(read_eeg_data_brainflow)
    
    return jsonify({"status": "Analysis started"})

@app.route('/stop-analysis', methods=['POST'])
def stop_analysis():
    """Stop EEG analysis"""
    global running
    running = False
    time.sleep(0.3)  # brief pause to let threads exit
    cleanup_spi_gpio()
    socketio.emit('analysis_stopped')  # Notify frontend to update the settings
    logging.info("Analysis stopped")
    return jsonify({"status": "Analysis stopped"})

@app.route('/start-recording', methods=['POST'])
def start_recording():
    """Begin buffering data for recording"""
    global recording_active, recording_paused, recording_data, recording_start_time
    global event_markers, recording_metadata, recording_sampling_rate
    payload = request.get_json(silent=True) or {}
    subject_id = bids_label(payload.get('subject'), '01')
    task = bids_label(payload.get('task'), 'resting')
    run = bids_label(payload.get('run'), '01')
    session_id = bids_label(payload.get('session'), '01')
    modality = bids_label(payload.get('modality') or current_signal_type, current_signal_type)

    if not subject_id or not session_id:
        return jsonify({"status": "error", "message": "Subject and session are required"}), 400

    with recording_lock, event_lock:
        recording_active = True
        recording_paused = False
        recording_data = []
        event_markers = []
        recording_start_time = time.time()
        recording_metadata = {
            "subject": subject_id,
            "task": task,
            "run": run,
            "session": session_id,
            "modality": modality
        }
        recording_sampling_rate = current_stream_sampling_rate
        metadata_cache[bids_cache_key(subject_id, session_id, task, run, modality)] = load_metadata_fields(
            subject_id,
            session_id,
            task,
            run,
            modality,
            recording_sampling_rate,
            3 if modality == 'motion' else enabled_channels
        )
    logging.info(
        f"Recording started for subject={subject_id} session={session_id} task={task} run={run} modality={modality}"
    )
    return jsonify({
        "status": "recording_started",
        "recording_paused": recording_paused,
        "sampling_rate": recording_sampling_rate,
        "metadata": recording_metadata
    })

@app.route('/pause-recording', methods=['POST'])
def pause_recording():
    """Toggle or set pause state for recording"""
    global recording_paused
    if not recording_active:
        return jsonify({"status": "error", "message": "Recording is not active"}), 400
    
    payload = request.get_json(silent=True) or {}
    requested_pause = payload.get('pause')
    with recording_lock:
        if requested_pause is None:
            recording_paused = not recording_paused
        else:
            recording_paused = bool(requested_pause)
    
    state = "paused" if recording_paused else "resumed"
    return jsonify({"status": f"Recording {state}", "recording_paused": recording_paused})


@app.route('/metadata', methods=['GET', 'POST'])
def metadata_route():
    """Load or save BIDS sidecar metadata for the current session"""
    if request.method == 'GET':
        params = request.args
        subject_id = bids_label(params.get('subject'), '01')
        session_id = bids_label(params.get('session'), '01')
        task = bids_label(params.get('task'), 'resting')
        run = bids_label(params.get('run'), '01')
        modality = bids_label(params.get('modality', current_signal_type), current_signal_type)
        sampling_rate = float(params.get('sampling_rate', current_stream_sampling_rate or fs))
        n_channels = int(params.get('channels', enabled_channels))
        fields = load_metadata_fields(subject_id, session_id, task, run, modality, sampling_rate, n_channels)
        return jsonify({"fields": fields})

    payload = request.get_json(silent=True) or {}
    subject_id = bids_label(payload.get('subject'), '01')
    session_id = bids_label(payload.get('session'), '01')
    task = bids_label(payload.get('task'), 'resting')
    run = bids_label(payload.get('run'), '01')
    modality = bids_label(payload.get('modality', current_signal_type), current_signal_type)
    fields = payload.get('fields', [])
    filtered, sidecar_path = persist_metadata_fields(fields, subject_id, session_id, task, run, modality)
    return jsonify({
        "status": "saved",
        "sidecar_path": sidecar_path,
        "included_fields": filtered,
        "modality": modality
    })

@app.route('/add-marker', methods=['POST'])
def add_marker():
    """Add an event marker tied to the current recording (trigger handling)"""
    global event_markers
    if not recording_active:
        return jsonify({"status": "error", "message": "Recording is not active"}), 400
    
    payload = request.get_json(silent=True) or {}
    label = payload.get('label', 'event')
    description = payload.get('description', '')
    timestamp = time.time()
    offset = timestamp - recording_start_time if recording_start_time else 0
    
    marker = {
        "label": label,
        "description": description,
        "time": timestamp,
        "offset": offset
    }
    with event_lock:
        event_markers.append(marker)
        total = len(event_markers)
    return jsonify({"status": "marker_added", "marker": marker, "total_markers": total})

@app.route('/stop-recording', methods=['POST'])
def stop_recording():
    """Finalize recording and return a BIDS-friendly archive"""
    if not recording_active:
        return jsonify({"status": "error", "message": "Recording is not active"}), 400
    
    payload = request.get_json(silent=True) or {}
    subject_id = bids_label(payload.get('subject', recording_metadata.get('subject', '01')), '01')
    session_id = bids_label(payload.get('session', recording_metadata.get('session', '01')), '01')
    task = bids_label(payload.get('task', recording_metadata.get('task', 'resting')), 'resting')
    run = bids_label(payload.get('run', recording_metadata.get('run', '01')), '01')
    modality = bids_label(payload.get('modality', recording_metadata.get('modality', current_signal_type)), current_signal_type)
    chosen_sampling_rate = payload.get('sampling_rate', recording_sampling_rate or current_stream_sampling_rate or fs)
    effective_sampling_rate = float(chosen_sampling_rate)

    with recording_lock:
        data_copy = [list(ch) for ch in recording_data] if recording_data else []
    with event_lock:
        markers_copy = list(event_markers)
    
    try:
        if not data_copy or not any(len(ch) for ch in data_copy):
            reset_recording_state()
            return jsonify({"status": "error", "message": "No recorded data available"}), 400
        
        metadata_fields = load_metadata_fields(
            subject_id,
            session_id,
            task,
            run,
            modality,
            effective_sampling_rate,
            len(data_copy)
        )
        # Persist UI config immediately so include/exclude selections are not lost
        persist_metadata_fields(metadata_fields, subject_id, session_id, task, run, modality)

        file_paths = save_bids_recording(
            data_copy,
            markers_copy,
            subject_id,
            session_id,
            task,
            run,
            effective_sampling_rate,
            modality,
            metadata_fields
        )

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for label, path in file_paths.items():
                if path and os.path.exists(path):
                    arcname = os.path.relpath(path, BIDS_ROOT)
                    zip_file.write(path, arcname=arcname)
            zip_file.writestr('recording_metadata.json', json.dumps({
                "subject": subject_id,
                "session": session_id,
                "task": task,
                "run": run,
                "modality": modality,
                "sampling_rate": effective_sampling_rate,
                "duration_seconds": len(data_copy[0]) / effective_sampling_rate if data_copy and len(data_copy[0]) > 0 else 0,
                "markers": markers_copy,
                "quality": qc_stats
            }, indent=2))
        
        zip_buffer.seek(0)
        reset_recording_state()
        logging.info(f"Recording stopped and packaged (subject={subject_id}, task={task}, run={run})")
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"sub-{subject_id}_ses-{session_id}_task-{task}_run-{run}_{modality}_recording.zip"
        )
    except Exception as e:
        logging.error(f"Error finalizing recording: {e}")
        reset_recording_state()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/update-settings', methods=['POST'])
def update_settings():
    """Update analysis settings"""
    global lowcut, highcut, order, baseline_correction_enabled, enabled_channels
    global ref_enabled, biasout_enabled, bandpass_enabled, smoothing_enabled
    global fs, downsample_factor, current_stream_sampling_rate, features_enabled
    
    data = request.json or {}
    # Mode-aware defaults
    if 'lowcut' not in data or 'highcut' not in data:
        if current_signal_type == 'emg':
            lowcut_default, highcut_default = 20.0, 150.0
        elif current_signal_type == 'motion':
            lowcut_default, highcut_default = 0.5, 20.0
        else:
            lowcut_default, highcut_default = 3.0, 45.0
    else:
        lowcut_default, highcut_default = lowcut, highcut
    
    lowcut = float(data.get('lowcut', lowcut_default))
    highcut = float(data.get('highcut', highcut_default))
    order = int(data.get('order', order))
    baseline_correction_enabled = data.get('baseline_correction_enabled', baseline_correction_enabled)
    enabled_channels = int(data.get('enabled_channels', enabled_channels))
    ref_enabled = data.get('ref_enabled', ref_enabled)
    biasout_enabled = data.get('biasout_enabled', biasout_enabled)
    bandpass_enabled = data.get('bandpass_filter_enabled', bandpass_enabled)
    smoothing_enabled = data.get('smoothing_enabled', smoothing_enabled)
    features_enabled = data.get('features_enabled', features_enabled)
    fs = max(1, int(data.get('sampling_rate', fs)))
    downsample_factor = max(1, int(data.get('downsample_factor', downsample_factor)))
    current_stream_sampling_rate = fs / downsample_factor if downsample_factor > 0 else fs
    update_filter_coefficients(fs)
    
    logging.info(f"Updated settings: lowcut={lowcut}, highcut={highcut}, order={order}, "
                 f"baseline_correction_enabled={baseline_correction_enabled}, "
                 f"enabled_channels={enabled_channels}, ref_enabled={ref_enabled}, "
                 f"biasout_enabled={biasout_enabled}, bandpass_enabled={bandpass_enabled}, "
                 f"smoothing_enabled={smoothing_enabled}, sampling_rate={fs}, "
                 f"downsample_factor={downsample_factor}, features_enabled={features_enabled}")
    return jsonify({"status": "Settings updated", "features_enabled": features_enabled})


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
        session_id = request.args.get('session', '01')
        task = request.args.get('task', 'resting')
        run = request.args.get('run', '01')
        sampling_rate = float(request.args.get('sampling_rate', current_stream_sampling_rate or fs))
        modality = request.args.get('modality', current_signal_type)
        if modality not in ('eeg', 'emg', 'motion'):
            modality = 'eeg'
        modality_label = modality.upper()
        with collected_data_lock:
            data_copy = [list(ch) for ch in collected_data] if collected_data else []
        with event_lock:
            markers_copy = list(event_markers)
        
        if not data_copy or len(data_copy[0]) == 0:
            return Response("No data available for export", status=400)
        
        # Limit number of rows
        if num_rows > len(data_copy[0]):
            num_rows = len(data_copy[0])
        
        data_to_export = [ch[:num_rows] for ch in data_copy]
        
        if export_format == 'csv':
            # Legacy CSV export
            csv_data = create_csv(data_to_export)
            return Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': 'attachment;filename=eeg_data.csv'}
            )
        else:
            metadata_fields = load_metadata_fields(
                subject_id,
                session_id,
                task,
                run,
                modality,
                sampling_rate,
                len(data_to_export)
            )
            persist_metadata_fields(metadata_fields, subject_id, session_id, task, run, modality)
            file_paths = save_bids_recording(
                data_to_export,
                markers_copy,
                subject_id,
                session_id,
                task,
                run,
                sampling_rate,
                modality,
                metadata_fields
            )
            events_path = file_paths.get("events")
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for label, path in file_paths.items():
                    if path and os.path.exists(path):
                        arcname = os.path.relpath(path, BIDS_ROOT)
                        zip_file.write(path, arcname=arcname)
                zip_file.writestr('README.txt', f"""# BIDS {modality_label} Dataset

Files follow the Brain Imaging Data Structure (BIDS) specification.

- {os.path.relpath(file_paths.get('edf'), BIDS_ROOT) if file_paths.get('edf') else 'N/A'}: data file
- {os.path.relpath(file_paths.get('json'), BIDS_ROOT) if file_paths.get('json') else 'N/A'}: sidecar metadata
- {os.path.relpath(file_paths.get('channels'), BIDS_ROOT) if file_paths.get('channels') else 'N/A'}: channel info
{"- " + os.path.relpath(events_path, BIDS_ROOT) + ": events" if events_path else ""}
""")
            
            zip_buffer.seek(0)
            
            return Response(
                zip_buffer.getvalue(),
                mimetype='application/zip',
                headers={
                    'Content-Disposition': f'attachment;filename=sub-{subject_id}_ses-{session_id}_task-{task}_run-{run}_{modality}_BIDS.zip'
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

@app.route('/sample-emg-data')
def sample_emg_data():
    """Provide a sample EMG JSON payload for simulation upload"""
    try:
        sr = int(request.args.get('sampling_rate', fs))
        duration = float(request.args.get('duration', 5))
        sample = generate_sample_emg(sampling_rate=sr, duration_seconds=duration)
        return jsonify(sample)
    except Exception as e:
        logging.error(f"Error generating sample EMG data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Main entry point of the application
if __name__ == '__main__':
    socketio.run(
        app,
        host='0.0.0.0',
        port=5001,
        allow_unsafe_werkzeug=True,
        debug=False,
        use_reloader=False
    )
