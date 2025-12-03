# BMI-SOFT-Data_Visualization

A Flask + Socket.IO web interface for streaming, visualizing, and exporting EEG/EMG data.

- **Synthetic**: Generates realistic EEG or EMG waveforms on the fly.
- **File replay**: Upload a JSON file (generated with `scripts/generate_sample_eeg.py`) to stream saved data through the UI.

## Prerequisites

- Python 3.10+
- Virtual environment with dependencies from `requirements.txt` (`pip install -r requirements.txt`)
- Optional: BrainFlow-compatible hardware (PiEEG) for real EEG acquisition

## Running the server

Use the no-reload Flask CLI command (so log output and WebSocket threads run in a single process). Run it from the project root:

```bash
FLASK_ENV=development FLASK_APP=app.py flask run --debug --no-reload
```

Alternatively, start the Socket.IO server directly:

```bash
python - <<'PY'
from app import app, socketio
socketio.run(app, host='0.0.0.0', port=5001, debug=False)
PY
```

Keep the terminal open while testing—this is where simulation/hardware logs appear and the process must stay alive while the browser displays data.

## Using the interface

1. Open your browser to `http://127.0.0.1:5001/`.
2. Choose **EEG** or **EMG** at the top. EMG currently supports simulation only.
3. Toggle **Simulation Mode** on if you want synthetic data or file replay. Leave it off for real EEG hardware.
4. Upload a JSON EEG file created by `scripts/generate_sample_eeg.py` to replay recorded data.
5. Adjust channel/filter settings as needed.
6. Click **Start Analysis**:
   - In hardware mode, BrainFlow reads the PiEEG board.
   - In simulation mode, the selected signals stream to the chart in real time.
7. Stop analysis any time with **Stop Analysis**.
8. Use **Start Calibration** to compute baseline offsets (simulation mode generates synthetic baselines).
9. Use **Export Data** to download the latest collected samples (CSV or BIDS-compliant EDF within a ZIP). (Currentla in the developpement)

## Generating sample data

```
python scripts/generate_sample_eeg.py
```

This creates `sample_eeg_data.json` which you can upload under Simulation Mode to test file replay.

## Notes

- The frontend logs `[v0] ...` messages in the browser console to help debug socket connectivity and chart updates.
