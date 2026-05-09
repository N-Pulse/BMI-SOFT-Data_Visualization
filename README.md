# N-Pulse BMI Visual Interface

Real-time web interface for visualizing, streaming, replaying, and recording EEG/EMG biosignals.

The app is built with **Flask + Socket.IO** and supports:

- synthetic EEG/EMG simulation,
- file replay from recorded sessions,
- live EEG acquisition through BrainFlow / PiEEG,
- live EMG acquisition from an Arduino UNO R4 Minima running the Upside Down Labs Chords firmware,
- DSI EEG streaming through Lab Streaming Layer,
- BIDS-style recording/export.

The main application is located in:

```bash
visual_interface/
```

---

## Quick start

Clone the repository:

```bash
git clone <repo-url>
cd BMI-SOFT-Data_Visualization
```

Launch the app:

```bash
cd visual_interface
./run_local.sh
```

Then open:

```text
http://127.0.0.1:5001
```

The script automatically creates a virtual environment at the repository root, installs the dependencies, and starts the Flask-SocketIO server.

---

## Manual setup

If you prefer to run the setup manually:

```bash
cd BMI-SOFT-Data_Visualization
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r visual_interface/requirements.txt
```

Then launch:

```bash
cd visual_interface
PORT=5001 python app.py
```

Open:

```text
http://127.0.0.1:5001
```

---

## First test without hardware

The easiest way to verify the app is working is to use simulation mode.

1. Open the web interface.
2. Enable **Simulation Mode**.
3. Select **EEG**.
4. Choose a number of channels, for example `8`.
5. Click **Start Stream**.

You should see synthetic EEG traces updating live.

---

## Generate sample data

From `visual_interface/`:

```bash
python scripts/generate_sample_eeg.py
```

This creates a sample JSON recording that can be loaded in the interface using Simulation Mode + file replay.

For EMG sample data:

```bash
python scripts/generate_sample_emg.py
```

---

## Repository structure

```text
BMI-SOFT-Data_Visualization/
├── README.md                     # Main setup and launch instructions
├── docs/
│   └── USER_GUIDE.md             # Detailed interface usage guide
├── eeg_realtime/                 # Small earlier EEG demo app
└── visual_interface/             # Main N-Pulse visual interface
    ├── app.py                    # Flask + Socket.IO backend
    ├── requirements.txt          # Python dependencies
    ├── run_local.sh              # Recommended local launcher
    ├── scripts/                  # Test data generation scripts
    ├── static/                   # Frontend JavaScript and CSS
    └── templates/                # HTML templates
```

---

## Hardware modes

The app can be launched without hardware. Hardware-specific libraries are loaded only when the corresponding mode is used.

### BrainFlow / PiEEG

Use this mode for BrainFlow-compatible EEG boards.

If needed, set the serial/SPI device before launch:

```bash
export PIEEG_SERIAL_PORT=/dev/spidev0.0
cd visual_interface
./run_local.sh
```

### DSI via LSL

Use this mode when the DSI headset is streamed through Lab Streaming Layer.

Make sure the DSI-to-LSL bridge is running first, then start the visual interface and select the DSI/LSL source in the UI.

### Arduino UNO R4 / Chords EMG

Use this mode for the Kraken EMG setup based on the Arduino UNO R4 Minima and the Upside Down Labs Chords firmware. The Arduino must be flashed with the Chords sketch that sends 6 analog channels at 500 Hz over USB serial at 230400 baud.

Before launching the app, connect the Arduino over USB and make sure no other program is using the same serial port. Do not keep the Chords web visualizer open at the same time, because only one program can read the Arduino serial stream.

On Linux, check the detected port with:

```bash
ls /dev/ttyACM* /dev/ttyUSB*
```

If needed, give your user access to serial ports:

```bash
sudo usermod -a -G dialout $USER
newgrp dialout
```

Then launch the app and select:

```text
Signal: EMG
Simulation Mode: OFF
Channels: 6
```

Click **Start Stream** to visualize the live EMG signals.

The backend auto-detects the Arduino port. To force a specific port, set:

```bash
export EMG_SERIAL_PORT=/dev/ttyACM0
export EMG_BAUD_RATE=230400
export EMG_SAMPLING_RATE=500
export EMG_CHANNELS=6
cd visual_interface
./run_local.sh
```

For simultaneous EEG + EMG visualization, select EEG as the main signal, keep hardware mode enabled, enable **Synchronized EEG + EMG**, and connect the Arduino for the EMG stream.

---

## Troubleshooting

### `python: command not found`

Use:

```bash
python3 --version
```

If Python exists but `python` does not, recreate the virtual environment:

```bash
cd BMI-SOFT-Data_Visualization
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r visual_interface/requirements.txt
```

### Port 5001 is already in use

Find the process:

```bash
lsof -i :5001
```

Kill it:

```bash
kill -9 <PID>
```

Or launch on another port:

```bash
cd visual_interface
PORT=5002 ./run_local.sh
```

Then open:

```text
http://127.0.0.1:5002
```

### The app starts but the chart is blank

Check:

- Simulation Mode is enabled if no hardware is connected.
- A file is uploaded if you expect file replay.
- The terminal does not show backend errors.
- You clicked **Start Stream**.

### BrainFlow import errors

The app should still launch in Simulation Mode even if BrainFlow has an issue. BrainFlow is only required for live BrainFlow/PiEEG acquisition.

Test BrainFlow separately:

```bash
source .venv/bin/activate
python -c "from brainflow.board_shim import BoardShim; print('BrainFlow OK')"
```

### LSL / pylsl errors

DSI mode requires Lab Streaming Layer to be available and the DSI LSL stream to be running. Simulation mode does not require LSL.

### EMG serial errors

If EMG hardware mode does not start, check that:

- the Arduino UNO R4 is connected and flashed with the Chords firmware,
- the Chords web visualizer and Arduino Serial Monitor are closed,
- the detected port exists, for example `/dev/ttyACM0`,
- your Linux user belongs to the `dialout` group,
- `pyserial` is installed in the active virtual environment.

You can force the port manually:

```bash
export EMG_SERIAL_PORT=/dev/ttyACM0
```

---

## Development workflow

Create a branch before modifying the app:

```bash
git switch -c my-feature-branch
```

Run the app locally:

```bash
cd visual_interface
./run_local.sh
```

Before committing, check what changed:

```bash
git status
git diff
```

Commit:

```bash
git add .
git commit -m "Describe the change"
```

Push:

```bash
git push -u origin my-feature-branch
```

---

## Main entry point

Use this command for local development:

```bash
cd visual_interface
./run_local.sh
```

Avoid using `flask run` directly unless the app has been explicitly refactored for that workflow. The Socket.IO server should be started through `app.py` or `run_local.sh`.
