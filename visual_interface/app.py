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
collected_data = []

calibration_values = [0] * 8

spi = None
chip = None
line = None
running = False

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

#####

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

# Define the main route to serve the web interface
@app.route('/')
def index():
    return render_template('index.html')



@app.route('/start-analysis', methods=['POST'])
def start_analysis():
    global running
    running = True
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
    """Export EEG data as CSV"""
    try:
        num_rows = int(request.args.get('num_rows', 5000))
        if collected_data and len(collected_data[0]) > 0:
            if num_rows > len(collected_data[0]):
                num_rows = len(collected_data[0])
            csv_data = create_csv([ch[:num_rows] for ch in collected_data])
        else:
            csv_data = create_csv([[]])
            
        return Response(
            csv_data,
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment;filename=eeg_data.csv'}
        )
    except Exception as e:
        logging.error(f"Error exporting data: {e}")
        return Response(
            "Internal Server Error",
            status=500
        )


###

# Main entry point of the application
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True, debug=True)
