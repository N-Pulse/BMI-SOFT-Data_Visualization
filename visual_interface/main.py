# Import the required libraries
import logging
import json
from flask import Flask, render_template, request, jsonify, Response
import spidev
import gpiod
import threading
import time
from scipy.signal import butter, filtfilt, iirnotch
import numpy as np
from flask_socketio import SocketIO, emit
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
import os
import asyncio
import inspect
import sqlite3
import base64

# Configure logging for detailed information during execution
logging.basicConfig(level=logging.INFO)

# Initialize Flask app and SocketIO for WebSocket support
app = Flask(__name__)
socketio = SocketIO(app)

# Define the main route to serve the web interface
@app.route('/')
def index():
    return render_template('index.html')

# Main entry point of the application


