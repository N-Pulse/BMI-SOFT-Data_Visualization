import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

params = BrainFlowInputParams()
BOARD_ID = BoardIds.SYNTHETIC_BOARD.value
board = BoardShim(BOARD_ID, params)

RUNNING = False
FS = BoardShim.get_sampling_rate(BOARD_ID)
EEG_CHS = BoardShim.get_eeg_channels(BOARD_ID)

def stream_loop():
    chunk_sec = 0.25
    samples = int(FS * chunk_sec)
    while RUNNING:
        data = board.get_current_board_data(samples)
        if data.size > 0:
            eeg = data[EEG_CHS, :].tolist()
            socketio.emit("eeg_chunk", {"fs": FS, "eeg": eeg})
        socketio.sleep(chunk_sec / 2)

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("start")
def handle_start():
    global RUNNING
    if RUNNING:
        return
    BoardShim.enable_dev_board_logger()  # optional
    board.prepare_session()
    board.start_stream()
    RUNNING = True
    socketio.start_background_task(stream_loop)

@socketio.on("stop")
def handle_stop():
    global RUNNING
    RUNNING = False
    try:
        board.stop_stream()
        board.release_session()
    except Exception:
        pass

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
