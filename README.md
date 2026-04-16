# N-Pulse Visual Interface — User Guide

A Flask + Socket.IO web interface for streaming, visualizing, and recording EEG/EMG biosignals in real time. Supports live hardware, synthetic simulation, and file replay from recorded sessions.

---

## Prerequisites

- Python 3.10+
- Virtual environment with all dependencies installed:
  ```bash
  pip install -r requirements.txt
  ```
- *Optional:* BrainFlow-compatible hardware (PiEEG) for real EEG acquisition
- *Optional:* `pylsl` for DSI-24 streaming via LSL

---

## Starting the server

Run from the project root (keep this terminal open while using the interface):

```bash
FLASK_ENV=development FLASK_APP=app.py flask run --debug --no-reload
```

Or launch directly with Socket.IO:

```bash
python - <<'PY'
from app import app, socketio
socketio.run(app, host='0.0.0.0', port=5001, debug=False)
PY
```

Then open **http://127.0.0.1:5001** in your browser.

---

## Signal type

At the top of the interface, select what signal you want to work with:

| Button | What it does |
|--------|-------------|
| **EEG** | Streams EEG data (hardware, simulation, or file replay) |
| **EMG** | Streams EMG data (simulation only for now) |
| **Motion** | Streams motion/accelerometer data |

**Synchronized EEG + EMG** toggle (below the signal buttons): runs both EEG and EMG simultaneously, showing two independent chart panels. Green = both streams active, Orange = only one stream.

---

## Hardware vs Simulation mode

Toggle **Simulation Mode** on or off in the Source Settings section.

### Simulation OFF — live hardware

- Set **Hardware Source** to:
  - `BrainFlow / PiEEG` — reads a BrainFlow-compatible board connected via USB/SPI
  - `DSI via LSL` — reads the first matching LSL EEG stream from a DSI-24 headset

### Simulation ON — synthetic or file replay

- **No file uploaded** → synthetic waveforms are generated on the fly (useful for UI testing without hardware)
- **File uploaded** → replays the recorded file through the same pipeline as live data

#### Uploading a recording file

Supported formats: `.fif`, `.edf`, `.xdf`, `.json`

Two ways to load:

1. **Single file** — click *Choose file* and select your signal file, then optionally upload a matching `events.tsv`
2. **Session folder** — click *Upload session folder* and select an entire BIDS-like folder (the interface auto-detects the signal file, `events.tsv`, `events.json`, and `channels.tsv`)

---

## Streaming — step by step

1. Choose signal type and hardware/simulation source
2. Set the number of **channels** (up to 64 for EEG, 16 for EMG)
3. *(Optional)* Adjust filter and signal processing settings
4. Click **Start Stream**
5. Click **Stop Stream** when done

---

## Waveform display

### Display modes

Select from the **Display** dropdown in the waveform panel header:

| Mode | Description |
|------|-------------|
| **EEG stacked** *(default)* | Each channel on its own baseline — best for comparing signal shapes and event timing across channels |
| **Overlap** | All channels share one y-axis — useful for quick amplitude comparison |

### Live vs Freeze

Two buttons sit next to the Display selector:

| Button | Behaviour |
|--------|-----------|
| **Live** | Chart auto-scrolls to show the most recent 8 seconds of signal |
| **⏸ Freeze** | Chart stops updating at the current moment so you can analyse it. For file replay this also pauses the server — no data is missed. Click **Live** to resume from exactly where you stopped. |

### File replay scrubber

When replaying a `.fif` / `.edf` / `.xdf` / `.json` file, a timeline scrubber appears below the chart:

- The thumb advances automatically as the file plays
- **Drag the thumb** to jump to any position in the file — the chart clears and refills from the new position within ~100 ms
- Scrubbing automatically switches back to **Live** mode so you see the signal immediately

---

## Channel selection

Below the waveform, a compact grid shows all enabled channels. Each cell is coloured to match its trace on the chart.

- **Click a cell** to show/hide that channel
- **All / None / Invert** buttons for bulk selection

The same grid is shown for the EMG panel when dual-stream mode is active.

---

## Calibration

Click **Calibrate** to measure the DC baseline of each channel over 5 seconds.

The per-channel offsets are stored and subtracted from incoming data whenever **Baseline Correction** is enabled in the signal processing settings. This removes electrode drift and DC bias that can push traces off screen or saturate the amplifier's range.

> Run calibration at the start of each session, before the participant begins any task. In simulation mode, synthetic baselines are generated automatically.

---

## Recording to BIDS

The **Recording Controls** section lets you save data in BIDS-compliant format.

1. Fill in **Subject**, **Session**, **Task**, **Run**, and **Modality** fields
2. Click **Start Recording** — data collection starts alongside the live stream
3. *(Optional)* Click **Pause** / resume mid-session
4. Click **Stop & Save** — the interface writes:
   - `_eeg.edf` (or `_emg.edf`) — raw signal in EDF format
   - `_eeg.json` — metadata sidecar
   - `_channels.tsv` — channel list
   - `_events.tsv` — event markers

### Adding event markers

During a recording, type a label in the **Marker label** field (e.g. `grip`, `rest`, `blink`) and click **Add Marker**. Each marker is timestamped and saved to `_events.tsv`.

---

## Exporting the buffer

Click **Export Buffer** to download the current in-memory data buffer (independent of a formal recording). Exports as CSV or as a BIDS-compliant EDF inside a ZIP archive.

---

## Live metrics

The **Live Metrics** panel shows real-time signal quality indicators:

| Metric | Meaning |
|--------|---------|
| **Avg Power** | Mean signal power across all channels (µV²) |
| **Muscle Activation** | RMS amplitude — proxy for muscle activity in EMG mode |
| **SNR** | Estimated signal-to-noise ratio |
| **Channel Quality** | 0–100 % score derived from SNR; a warning appears on the chart if quality drops below 35 % |

The **band-power chart** and **feature time chart** below the metrics update every 400 ms with spectral features (delta, theta, alpha, beta, gamma bands).

---

## Synchronized EEG + EMG (dual-stream)

Toggle **Synchronized EEG + EMG** to stream both signals simultaneously:

- EEG appears in the main waveform panel
- EMG appears in a second panel below with its own channel grid and RMS/MAV metrics
- Set the number of EMG channels (1–16) in the field that appears when the toggle is on
- Both streams can be recorded and exported independently

---

## Generating test data

```bash
python scripts/generate_sample_eeg.py
```

Creates `sample_eeg_data.json` in the project root. Upload it under Simulation Mode → file replay to test the full pipeline without hardware.

---

## DSI-24 via LSL (hardware mode)

1. Start the DSI bridge application that publishes the LSL stream
2. In the UI: set Simulation Mode **OFF**, Signal Type **EEG**, Hardware Source **DSI via LSL**
3. Click **Start Stream**

Optional environment variables for stream matching:

```bash
DSI_LSL_TYPE=EEG          # stream type to search for (default: EEG)
DSI_LSL_NAME=DSI-24       # stream name filter (default: any)
DSI_LSL_TIMEOUT=10        # seconds to wait for the stream (default: 8)
```

Example:
```bash
DSI_LSL_TYPE=EEG DSI_LSL_NAME=DSI-24 DSI_LSL_TIMEOUT=10 python app.py
```

---

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Chart is blank after Start Stream | Is Simulation Mode on? Is a file loaded if you expect file replay? Check the terminal for errors. |
| Scrubber does not appear | Only shown during file replay — upload a `.fif`/`.edf`/`.xdf`/`.json` file and start the stream |
| Signal freezes immediately | Check if **⏸ Freeze** is active — click **Live** to resume |
| Hardware not detected | Check USB/SPI connection, BrainFlow board ID setting, or LSL bridge status |
| High noise / poor quality | Run **Calibrate**, enable **Baseline Correction** and **Notch filter** in signal processing settings |
| `pylsl` import error | Requires Python < 3.14; use a Python 3.12 virtual environment for DSI mode |
