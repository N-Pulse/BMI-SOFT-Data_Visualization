// Socket.IO connection
const socket = io();

let isSimulationMode = false;
let chart = null;
let currentSignalType = 'eeg';

// Initialize the chart on page load
document.addEventListener('DOMContentLoaded', function() {
    currentSignalType = (window.initialSignalType || 'eeg');
    console.log('[v0] Page loaded, initializing chart');
    initializeChart();
    updateSignalUI();
});

function parseJsonResponse(response) {
    const contentType = response.headers.get('content-type') || '';
    const isJson = contentType.includes('application/json');
    
    if (!isJson) {
        return response.text().then(text => {
            const message = text || `Request failed with status ${response.status}`;
            if (response.ok) {
                return { message };
            }
            throw { message };
        });
    }
    
    return response.json().then(data => {
        if (!response.ok) {
            throw data;
        }
        return data;
    });
}

function initializeChart() {
    const ctx = document.getElementById('eegChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Channel 1', data: [], borderColor: 'rgb(255, 99, 132)', tension: 0.1 },
                { label: 'Channel 2', data: [], borderColor: 'rgb(54, 162, 235)', tension: 0.1 },
                { label: 'Channel 3', data: [], borderColor: 'rgb(255, 206, 86)', tension: 0.1 },
                { label: 'Channel 4', data: [], borderColor: 'rgb(75, 192, 192)', tension: 0.1 },
                { label: 'Channel 5', data: [], borderColor: 'rgb(153, 102, 255)', tension: 0.1 },
                { label: 'Channel 6', data: [], borderColor: 'rgb(255, 159, 64)', tension: 0.1 },
                { label: 'Channel 7', data: [], borderColor: 'rgb(199, 199, 199)', tension: 0.1 },
                { label: 'Channel 8', data: [], borderColor: 'rgb(83, 102, 255)', tension: 0.1 }
            ]
        },
        options: {
            responsive: true,
            animation: false,
            scales: {
                x: { display: false },
                y: { 
                    title: { display: true, text: getYAxisLabel() }
                }
            },
            plugins: {
                legend: { display: true, position: 'top' }
            }
        }
    });
    console.log('[v0] Chart initialized successfully');
}

// Toggle simulation mode
function toggleSimulation() {
    console.log('[v0] Toggle button clicked');
    const toggle = document.getElementById('simulationToggle');
    const label = document.getElementById('modeLabel');
    const fileUploadSection = document.getElementById('fileUploadSection');
    
    isSimulationMode = toggle.checked;
    
    console.log('[v0] Toggle checkbox checked:', toggle.checked);
    console.log('[v0] Toggling simulation mode to:', isSimulationMode);
    
    // Show/hide file upload section
    if (fileUploadSection) {
        fileUploadSection.style.display = isSimulationMode ? 'block' : 'none';
    }
    
    fetch('/toggle-simulation', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ simulation: isSimulationMode })
    })
        .then(response => {
            console.log('[v0] Response status:', response.status);
            return parseJsonResponse(response);
        })
        .then(data => {
            console.log('[v0] Server response:', data);
            console.log('[v0] Simulation mode toggled successfully. Mode:', data.mode);
            label.textContent = data.mode === 'simulation' ? 'Simulation Mode' : 'Hardware Mode';
            label.style.color = data.mode === 'simulation' ? '#ff9800' : '#4CAF50';
        })
        .catch(error => {
            console.error('[v0] Error toggling simulation:', error);
            // Revert toggle on error
            toggle.checked = !toggle.checked;
            isSimulationMode = !isSimulationMode;
            if (fileUploadSection) {
                fileUploadSection.style.display = !isSimulationMode ? 'block' : 'none';
            }
            alert('Error toggling simulation mode. Check console for details.');
        });
}

function setSignalType(type) {
    if (type === currentSignalType) {
        return;
    }
    
    fetch('/set-signal-type', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ signal_type: type })
    })
    .then(parseJsonResponse)
    .then(data => {
        currentSignalType = data.signal_type;
        console.log('[v0] Signal type set to', currentSignalType);
        updateSignalUI();
    })
    .catch(error => {
        console.error('[v0] Error changing signal type:', error);
        alert(error.message || 'Failed to change signal type');
    });
}

function updateSignalUI() {
    const eegBtn = document.getElementById('signalEegBtn');
    const emgBtn = document.getElementById('signalEmgBtn');
    const description = document.getElementById('signalDescription');
    
    if (!eegBtn || !emgBtn || !description) return;
    
    eegBtn.classList.toggle('active', currentSignalType === 'eeg');
    emgBtn.classList.toggle('active', currentSignalType === 'emg');
    
    description.textContent = currentSignalType === 'emg'
        ? 'Track muscle activation levels (EMG mode).'
        : 'Monitor electrical brain activity (EEG mode).';
    
    if (chart) {
        const prefix = currentSignalType.toUpperCase();
        chart.data.datasets.forEach((dataset, idx) => {
            dataset.label = `${prefix} Channel ${idx + 1}`;
        });
        chart.options.scales.y.title.text = getYAxisLabel();
        chart.update('none');
    }
}

function getYAxisLabel() {
    return currentSignalType === 'emg' ? 'Amplitude (mV)' : 'Amplitude (µV)';
}

// Start analysis
function startAnalysis() {
    console.log('[v0] Starting analysis...');
    fetch('/start-analysis', { method: 'POST' })
        .then(parseJsonResponse)
        .then(data => {
            console.log('[v0] Analysis started:', data);
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
        })
        .catch(error => {
            console.error('[v0] Error starting analysis:', error);
            alert(error.message || 'Failed to start analysis');
        });
}

// Stop analysis
function stopAnalysis() {
    console.log('[v0] Stopping analysis...');
    fetch('/stop-analysis', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            console.log('[v0] Analysis stopped:', data);
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        })
        .catch(error => console.error('[v0] Error stopping analysis:', error));
}

// Handle EEG file upload
function uploadEEGFile() {
    const fileInput = document.getElementById('eegFileInput');
    const fileStatus = document.getElementById('fileStatus');
    
    if (!fileInput.files || fileInput.files.length === 0) {
        fileStatus.textContent = 'No file selected';
        fileStatus.style.color = '#f44336';
        return;
    }
    
    const file = fileInput.files[0];
    
    // Check if file is JSON
    if (!file.name.endsWith('.json')) {
        fileStatus.textContent = 'Error: Please upload a JSON file';
        fileStatus.style.color = '#f44336';
        return;
    }
    
    fileStatus.textContent = 'Uploading...';
    fileStatus.style.color = '#ff9800';
    
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/upload-simulation-data', {
        method: 'POST',
        body: formData
    })
    .then(parseJsonResponse)
    .then(data => {
        if (data.status === 'success') {
            const metadata = data.metadata;
            fileStatus.innerHTML = `
                <strong>✓ File loaded successfully!</strong><br>
                Channels: ${metadata.num_channels}<br>
                Duration: ${metadata.duration_seconds}s<br>
                Samples: ${metadata.num_samples}<br>
                Sampling Rate: ${metadata.sampling_rate} Hz
            `;
            fileStatus.style.color = '#4CAF50';
            console.log('[v0] Simulation data loaded:', metadata);
        } else {
            fileStatus.textContent = `Error: ${data.message}`;
            fileStatus.style.color = '#f44336';
        }
    })
    .catch(error => {
        console.error('[v0] Error uploading file:', error);
        fileStatus.textContent = 'Error uploading file. Check console for details.';
        fileStatus.style.color = '#f44336';
    });
}

// Update settings
function updateSettings() {
    const settings = {
        enabled_channels: parseInt(document.getElementById('enabled_channels').value) || 8,
        ref_enabled: document.getElementById('ref_enabled').checked,
        biasout_enabled: document.getElementById('biasout_enabled').checked,
        baseline_correction_enabled: document.getElementById('baseline_correction_enabled').checked,
        bandpass_filter_enabled: document.getElementById('bandpass_filter_enabled').checked
    };
    
    console.log('[v0] Updating settings:', settings);
    
    // Update the display value
    document.getElementById('enabledChannelsValue').textContent = settings.enabled_channels;
    
    fetch('/update-settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
    })
    .then(response => response.json())
    .then(data => console.log('[v0] Settings updated:', data))
    .catch(error => console.error('[v0] Error updating settings:', error));
}

// Socket.IO event handlers
socket.on('connect', function() {
    console.log('[v0] Socket.IO connected successfully');
});

socket.on('disconnect', function() {
    console.log('[v0] Socket.IO disconnected');
});

socket.on('eeg_data', function(data) {
    console.log('[v0] Received eeg_data event with', data.channels ? data.channels.length : 0, 'channels');
    updateChart(data);
});

function updateChart(data) {
    if (!chart) {
        console.error('[v0] Chart not initialized');
        return;
    }
    
    if (!data.channels || !Array.isArray(data.channels)) {
        console.error('[v0] Invalid data format:', data);
        return;
    }
    
    const maxDataPoints = 250; // Keep 250 points visible (about 1 second at 250Hz)
    
    // Update each channel dataset
    data.channels.forEach((channelData, index) => {
        if (index < chart.data.datasets.length && Array.isArray(channelData)) {
            // Append new data points to existing data
            chart.data.datasets[index].data.push(...channelData);
            
            // Keep only the last maxDataPoints
            if (chart.data.datasets[index].data.length > maxDataPoints) {
                chart.data.datasets[index].data = chart.data.datasets[index].data.slice(-maxDataPoints);
            }
        }
    });
    
    // Update x-axis labels
    const dataLength = chart.data.datasets[0].data.length;
    chart.data.labels = Array.from({ length: dataLength }, (_, i) => i);
    
    // Update chart without animation for smooth real-time display
    chart.update('none');
}

// Start calibration
function startCalibration() {
    console.log('[v0] Starting calibration...');
    fetch('/calibrate', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            console.log('[v0] Calibration complete:', data);
            if (data.values) {
                alert('Calibration complete!\nBaseline values: ' + data.values.map(v => v.toFixed(2)).join(', '));
            } else {
                alert('Calibration complete!');
            }
        })
        .catch(error => {
            console.error('[v0] Error during calibration:', error);
            alert('Calibration failed. Check console for details.');
        });
}

// Export data
function exportData() {
    console.log('[v0] Exporting data...');
    window.location.href = '/export-data?num_rows=5000';
}
