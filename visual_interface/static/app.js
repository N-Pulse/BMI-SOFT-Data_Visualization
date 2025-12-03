// Socket.IO connection
const socket = io();

const channelColors = [
  'rgb(255, 99, 132)',
  'rgb(54, 162, 235)',
  'rgb(255, 206, 86)',
  'rgb(75, 192, 192)',
  'rgb(153, 102, 255)',
  'rgb(255, 159, 64)',
  'rgb(199, 199, 199)',
  'rgb(83, 102, 255)'
];

const headChannels = [0, 1, 2, 3];
const armChannels = [4, 5, 6, 7];

let channelVisibility = Array(8).fill(true);
let chart = null;
let bandPowerChart = null;
let featureTimeChart = null;
let featureHistory = [];
let isSimulationMode = false;
let currentSignalType = 'eeg';
let chartMode = 'overlap';
let isRecording = false;
let isRecordingPaused = false;
let markers = [];
let metadataFields = [];
let metadataDragIndex = null;
const BANDPOWER_Y_MAX = 2000; // fixed y-axis for band power chart
const waveformWindowSec = 8; // seconds of data to keep in the live view

let waveformBuffers = Array.from({ length: 8 }, () => []);
let lastSampleTimestamp = 0;
let stackedOffsets = [];

// Initialize the UI on page load
document.addEventListener('DOMContentLoaded', () => {
  currentSignalType = window.initialSignalType || 'eeg';
  const modeSelect = document.getElementById('waveformModeSelect');
  if (modeSelect) {
    chartMode = modeSelect.value || 'overlap';
  }
  buildChannelSelectors();
  initializeChart();
  initializeFeatureCharts();
  updateSignalUI();
  updateSettings();
  loadMetadata();
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

function escapeHtml(value) {
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  return String(value ?? '').replace(/[&<>"']/g, ch => map[ch] || ch);
}

function getEnabledChannels() {
  return parseInt(document.getElementById('enabled_channels').value, 10) || 8;
}

function syncWaveformBuffers(enabled) {
  while (waveformBuffers.length < channelColors.length) {
    waveformBuffers.push([]);
  }
  waveformBuffers = waveformBuffers.slice(0, channelColors.length);
  for (let i = 0; i < waveformBuffers.length; i += 1) {
    if (!Array.isArray(waveformBuffers[i])) {
      waveformBuffers[i] = [];
    }
  }
  // Ensure visibility array stays aligned with channel count
  channelVisibility = channelVisibility.slice(0, enabled);
  while (channelVisibility.length < enabled) {
    channelVisibility.push(true);
  }
}

function resetWaveformBuffers() {
  waveformBuffers = Array.from({ length: channelColors.length }, () => []);
  lastSampleTimestamp = 0;
}

function buildChannelSelectors() {
  const container = document.getElementById('channelList');
  if (!container) return;
  const enabled = getEnabledChannels();
  syncWaveformBuffers(enabled);
  container.innerHTML = '';
  for (let idx = 0; idx < enabled; idx += 1) {
    createChannelPill(idx, container);
  }
  if (chart) {
    chart.data.datasets.forEach((ds, idx) => {
      ds.hidden = idx >= enabled ? true : !channelVisibility[idx];
    });
    renderWaveform();
  }
}

function createChannelPill(index, container) {
  const pill = document.createElement('div');
  pill.className = 'channel-pill' + (channelVisibility[index] ? ' active' : '');
  pill.textContent = `Ch ${index + 1}`;
  pill.style.borderColor = channelColors[index];
  pill.onclick = () => toggleChannel(index, pill);
  container.appendChild(pill);
}

function toggleChannel(index, pill) {
  channelVisibility[index] = !channelVisibility[index];
  if (pill) {
    pill.classList.toggle('active', channelVisibility[index]);
  }
  if (chart && chart.data.datasets[index]) {
    chart.data.datasets[index].hidden = !channelVisibility[index];
    renderWaveform();
  }
}

const stackedLabelPlugin = {
  id: 'stackedLabels',
  afterDatasetsDraw(chartInstance) {
    if (chartMode !== 'stacked') return;
    const { ctx, chartArea, scales } = chartInstance;
    if (!chartArea || !scales?.y) return;
    ctx.save();
    ctx.fillStyle = '#cfd8dc';
    ctx.font = '12px Arial';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const enabled = getEnabledChannels();
    for (let i = 0; i < enabled; i += 1) {
      const dataset = chartInstance.data.datasets[i];
      if (!dataset || dataset.hidden) continue;
      const offset = stackedOffsets[i] || 0;
      const y = scales.y.getPixelForValue(offset);
      const labelText = (dataset.label || `Ch ${i + 1}`).replace(/^[A-Z]+\s+/, '');
      ctx.fillText(labelText, chartArea.left - 10, y);
    }
    ctx.restore();
  }
};

function buildChartOptions() {
  const range = getYAxisRange();
  return {
    responsive: true,
    animation: false,
    parsing: false,
    normalized: true,
    scales: {
      x: {
        type: 'linear',
        display: true,
        min: 0,
        max: waveformWindowSec,
        title: { display: true, text: 'Time (s)' },
        ticks: {
          maxTicksLimit: 6,
          callback: value => {
            const num = Number(value);
            return Number.isFinite(num) ? num.toFixed(1) : value;
          }
        }
      },
      y: {
        display: chartMode !== 'stacked',
        title: {
          display: chartMode !== 'stacked',
          text: getYAxisLabel()
        },
        min: range.min,
        max: range.max,
        grid: { color: 'rgba(255, 255, 255, 0.05)' }
      }
    },
    plugins: {
      legend: { display: chartMode === 'overlap', position: 'top' }
    },
    layout: chartMode === 'stacked' ? { padding: { left: 70, right: 10, top: 10, bottom: 10 } } : {}
  };
}

function initializeChart() {
  const canvas = document.getElementById('eegChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (chart) {
    chart.destroy();
  }
  const enabled = getEnabledChannels();
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: channelColors.map((color, idx) => ({
        label: `${currentSignalType.toUpperCase()} Channel ${idx + 1}`,
        data: [],
        borderColor: color,
        borderWidth: 1.4,
        tension: 0.05,
        pointRadius: 0,
        hidden: idx >= enabled ? true : !channelVisibility[idx]
      }))
    },
    options: buildChartOptions(),
    plugins: [stackedLabelPlugin]
  });
  renderWaveform();
}

function initializeFeatureCharts() {
  const timeCanvas = document.getElementById('featureTimeChart');
  if (timeCanvas) {
    featureTimeChart = new Chart(timeCanvas.getContext('2d'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Mean Power', data: [], borderColor: '#1e88e5', tension: 0.2, pointRadius: 0 },
          { label: 'Baseline', data: [], borderColor: '#666', borderDash: [4, 4], tension: 0, pointRadius: 0 }
        ]
      },
      options: {
        animation: false,
        scales: { x: { display: false }, y: { title: { display: true, text: 'Power (uV^2)' } } },
        plugins: { legend: { display: true } }
      }
    });
  }

  const barCtx = document.getElementById('featureBarChart').getContext('2d');
  bandPowerChart = new Chart(barCtx, {
    type: 'bar',
    data: {
      labels: ['Delta (1-4)', 'Theta (4-8)', 'Alpha (8-13)', 'Beta (13-30)', 'Gamma (30-45)'],
      datasets: [
        {
          label: 'Band power (avg last chunk)',
          data: [0, 0, 0, 0, 0],
          backgroundColor: ['#7e57c2', '#42a5f5', '#66bb6a', '#ffa726', '#ef5350']
        }
      ]
    },
    options: {
      animation: false,
      scales: {
        y: {
          beginAtZero: true,
          title: { display: true, text: 'Power (uV^2)' },
          max: BANDPOWER_Y_MAX
        }
      },
      plugins: { legend: { display: false } }
    }
  });
}

// Toggle simulation mode
function toggleSimulation() {
  const toggle = document.getElementById('simulationToggle');
  const label = document.getElementById('modeLabel');
  const fileUploadSection = document.getElementById('fileUploadSection');

  isSimulationMode = toggle.checked;

  if (fileUploadSection) {
    fileUploadSection.style.display = isSimulationMode ? 'block' : 'none';
  }

  fetch('/toggle-simulation', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ simulation: isSimulationMode })
  })
    .then(parseJsonResponse)
    .then(data => {
      label.textContent = data.mode === 'simulation' ? 'Simulation Mode' : 'Hardware Mode';
      label.style.color = data.mode === 'simulation' ? '#ff9800' : '#4CAF50';
    })
    .catch(error => {
      console.error('[client] Error toggling simulation:', error);
      toggle.checked = !toggle.checked;
      isSimulationMode = !isSimulationMode;
      if (fileUploadSection) {
        fileUploadSection.style.display = isSimulationMode ? 'block' : 'none';
      }
      alert('Error toggling simulation mode. Check console for details.');
    });
}

function setSignalType(type) {
  if (type === currentSignalType) return;

  fetch('/set-signal-type', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ signal_type: type })
  })
    .then(parseJsonResponse)
    .then(data => {
      currentSignalType = data.signal_type;
      updateSignalUI();
      resetWaveformBuffers();
      renderWaveform();
      updateSettings();
      loadMetadata();
    })
    .catch(error => {
      console.error('[client] Error changing signal type:', error);
      alert(error.message || 'Failed to change signal type');
    });
}

function updateSignalUI() {
  const eegBtn = document.getElementById('signalEegBtn');
  const emgBtn = document.getElementById('signalEmgBtn');
  const motionBtn = document.getElementById('signalMotionBtn');
  const description = document.getElementById('signalDescription');

  if (eegBtn && emgBtn && description) {
    eegBtn.classList.toggle('active', currentSignalType === 'eeg');
    emgBtn.classList.toggle('active', currentSignalType === 'emg');
    if (motionBtn) {
      motionBtn.classList.toggle('active', currentSignalType === 'motion');
    }
    description.textContent =
      currentSignalType === 'emg'
        ? 'Track muscle activation levels (EMG mode).'
        : currentSignalType === 'motion'
        ? 'Stream motion/IMU channels (synthetic in simulation).'
        : 'Monitor electrical brain activity (EEG mode).';
  }

  const modalitySelect = document.getElementById('recModality');
  if (modalitySelect) {
    modalitySelect.value = currentSignalType;
  }
  if (currentSignalType === 'motion') {
    const channelInput = document.getElementById('enabled_channels');
    if (channelInput) {
      channelInput.value = 3;
    }
    buildChannelSelectors();
  }

  if (chart) {
    const prefix = currentSignalType.toUpperCase();
    chart.data.datasets.forEach((dataset, idx) => {
      dataset.label = `${prefix} Channel ${idx + 1}`;
    });
    chart.options.scales.y.title.text = getYAxisLabel();
    renderWaveform();
  }
}

function getYAxisLabel() {
  if (currentSignalType === 'emg') return 'Amplitude (mV)';
  if (currentSignalType === 'motion') return 'Acceleration (a.u.)';
  return 'Amplitude (µV)';
}

function getYAxisRange() {
  if (currentSignalType === 'emg') return { min: -200, max: 200 };
  if (currentSignalType === 'motion') return { min: -5, max: 5 };
  // Keep EEG amplitude centered and stable
  return { min: -400, max: 400 };
}

function setChartMode(mode) {
  const nextMode = mode === 'stacked' ? 'stacked' : 'overlap';
  if (chartMode === nextMode) return;
  chartMode = nextMode;
  const select = document.getElementById('waveformModeSelect');
  if (select && select.value !== nextMode) {
    select.value = nextMode;
  }
  initializeChart();
}

// Start analysis
function startAnalysis() {
  fetch('/start-analysis', { method: 'POST' })
    .then(parseJsonResponse)
    .then(() => {
      resetWaveformBuffers();
      renderWaveform();
      document.getElementById('startBtn').disabled = true;
      document.getElementById('stopBtn').disabled = false;
    })
    .catch(error => {
      console.error('[client] Error starting analysis:', error);
      alert(error.message || 'Failed to start analysis');
    });
}

// Stop analysis
function stopAnalysis() {
  fetch('/stop-analysis', { method: 'POST' })
    .then(parseJsonResponse)
    .then(() => {
      document.getElementById('startBtn').disabled = false;
      document.getElementById('stopBtn').disabled = true;
    })
    .catch(error => console.error('[client] Error stopping analysis:', error));
}

// Handle EEG/EMG file upload
function uploadEEGFile() {
  const fileInput = document.getElementById('eegFileInput');
  const fileStatus = document.getElementById('fileStatus');

  if (!fileInput.files || fileInput.files.length === 0) {
    fileStatus.textContent = 'No file selected';
    fileStatus.style.color = '#f44336';
    return;
  }

  const file = fileInput.files[0];
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
      } else {
        fileStatus.textContent = `Error: ${data.message}`;
        fileStatus.style.color = '#f44336';
      }
    })
    .catch(error => {
      console.error('[client] Error uploading file:', error);
      fileStatus.textContent = 'Error uploading file. Check console for details.';
      fileStatus.style.color = '#f44336';
    });
}

// Update settings
function updateSettings() {
  const settings = {
    enabled_channels: parseInt(document.getElementById('enabled_channels').value, 10) || 8,
    ref_enabled: document.getElementById('ref_enabled').checked,
    biasout_enabled: document.getElementById('biasout_enabled').checked,
    baseline_correction_enabled: document.getElementById('baseline_correction_enabled').checked,
    bandpass_filter_enabled: document.getElementById('bandpass_filter_enabled').checked,
    smoothing_enabled: document.getElementById('smoothing_enabled').checked,
    sampling_rate: parseInt(document.getElementById('sampling_rate').value, 10) || 250,
    downsample_factor: parseInt(document.getElementById('downsample_factor').value, 10) || 1,
    lowcut: parseFloat(document.getElementById('lowcut').value) || 3,
    highcut: parseFloat(document.getElementById('highcut').value) || 45,
    order: parseInt(document.getElementById('order').value, 10) || 2
  };

  fetch('/update-settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings)
  })
    .then(parseJsonResponse)
    .then(() => buildChannelSelectors())
    .catch(error => console.error('[client] Error updating settings:', error));
}

// Recording lifecycle
function startRecording() {
  const subject = document.getElementById('recSubject').value || '01';
  const session = document.getElementById('recSession').value || '01';
  const task = document.getElementById('recTask').value || 'resting';
  const run = document.getElementById('recRun').value || '01';
  const modality = document.getElementById('recModality').value || currentSignalType;

  fetch('/start-recording', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ subject, session, task, run, modality })
  })
    .then(parseJsonResponse)
    .then(() => {
      isRecording = true;
      isRecordingPaused = false;
      markers = [];
      renderMarkers();
      document.getElementById('startRecBtn').disabled = true;
      document.getElementById('pauseRecBtn').disabled = false;
      document.getElementById('stopRecBtn').disabled = false;
      document.getElementById('pauseRecBtn').textContent = 'Pause';
    })
    .catch(error => {
      console.error('[client] Error starting recording:', error);
      alert(error.message || 'Failed to start recording');
    });
}

function pauseRecording() {
  if (!isRecording) return;
  const requestedPause = !isRecordingPaused;

  fetch('/pause-recording', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pause: requestedPause })
  })
    .then(parseJsonResponse)
    .then(() => {
      isRecordingPaused = requestedPause;
      document.getElementById('pauseRecBtn').textContent = isRecordingPaused ? 'Resume' : 'Pause';
    })
    .catch(error => console.error('[client] Error toggling pause:', error));
}

function stopRecording() {
  if (!isRecording) return;
  const subject = document.getElementById('recSubject').value || '01';
  const session = document.getElementById('recSession').value || '01';
  const task = document.getElementById('recTask').value || 'resting';
  const run = document.getElementById('recRun').value || '01';
  const modality = document.getElementById('recModality').value || currentSignalType;

  const finalizeStop = () =>
    fetch('/stop-recording', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subject, session, task, run, modality })
    })
    .then(async response => {
      if (response.ok && (response.headers.get('content-type') || '').includes('application/zip')) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `sub-${subject}_task-${task}_run-${run}_recording.zip`;
        a.click();
        window.URL.revokeObjectURL(url);
        return { status: 'ok' };
      }
      return parseJsonResponse(response);
    })
    .then(data => {
      if (data && data.status === 'error') {
        alert(data.message || 'Failed to save recording');
      }
      resetRecordingUI();
    })
    .catch(error => {
      console.error('[client] Error stopping recording:', error);
      alert(error.message || 'Failed to stop recording');
      resetRecordingUI();
    });

  // Persist metadata selections before saving BIDS outputs
  saveMetadata(true).finally(finalizeStop);
}

function resetRecordingUI() {
  isRecording = false;
  isRecordingPaused = false;
  document.getElementById('startRecBtn').disabled = false;
  document.getElementById('pauseRecBtn').disabled = true;
  document.getElementById('stopRecBtn').disabled = true;
  document.getElementById('pauseRecBtn').textContent = 'Pause';
}

function addMarker() {
  if (!isRecording) {
    alert('Start recording to add markers.');
    return;
  }
  const label = document.getElementById('markerLabel').value || 'event';
  const description = document.getElementById('markerDescription').value || '';

  fetch('/add-marker', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label, description })
  })
    .then(parseJsonResponse)
    .then(data => {
      if (data.marker) {
        markers.push(data.marker);
        renderMarkers();
        document.getElementById('markerLabel').value = '';
        document.getElementById('markerDescription').value = '';
      }
    })
    .catch(error => console.error('[client] Error adding marker:', error));
}

function renderMarkers() {
  const list = document.getElementById('markerList');
  if (!list) return;
  list.innerHTML = markers
    .map(m => `<div class="marker-item"><span>${m.label}</span><span>${m.offset.toFixed(2)}s</span></div>`)
    .join('');
}

// BIDS metadata editor helpers
function getFormContext() {
  return {
    subject: document.getElementById('recSubject').value || '01',
    session: document.getElementById('recSession').value || '01',
    task: document.getElementById('recTask').value || 'resting',
    run: document.getElementById('recRun').value || '01',
    modality: document.getElementById('recModality').value || currentSignalType,
    sampling_rate: parseInt(document.getElementById('sampling_rate').value, 10) || 250,
    channels: parseInt(document.getElementById('enabled_channels').value, 10) || 8
  };
}

function setMetadataStatus(message, isError = false) {
  const el = document.getElementById('metadataModalStatus');
  if (!el) return;
  el.textContent = message;
  el.style.color = isError ? '#f44336' : '#4CAF50';
}

function openMetadataModal() {
  const modal = document.getElementById('metadataModal');
  if (!modal) return;
  modal.classList.remove('hidden');
  loadMetadata();
}

function closeMetadataModal() {
  const modal = document.getElementById('metadataModal');
  if (!modal) return;
  modal.classList.add('hidden');
}

function normalizeField(field) {
  return {
    key: field?.key || '',
    value: field?.value ?? '',
    include: field?.include !== false
  };
}

function renderMetadataFields() {
  const container = document.getElementById('metadataModalFieldList');
  if (!container) return;
  if (!metadataFields.length) {
    container.innerHTML = '<div class="metadata-empty">No metadata loaded yet.</div>';
    return;
  }
  container.innerHTML = metadataFields
    .map(
      (field, idx) => `
        <div class="metadata-row ${idx % 2 === 1 ? 'alt' : ''}" data-index="${idx}" draggable="true" ondragstart="handleMetadataDragStart(${idx})" ondragover="handleMetadataDragOver(event, ${idx})" ondragleave="handleMetadataDragLeave(${idx})" ondrop="handleMetadataDrop(${idx})" ondragend="handleMetadataDragEnd()">
          <span class="meta-handle">☰</span>
          <input type="checkbox" class="meta-include" ${field.include ? 'checked' : ''} onchange="toggleMetadataInclude(${idx}, this.checked)">
          <input type="text" class="meta-key" value="${escapeHtml(field.key)}" placeholder="Key" oninput="updateMetadataKey(${idx}, this.value)">
          <input type="text" class="meta-value" value="${escapeHtml(field.value)}" placeholder="Value" oninput="updateMetadataValue(${idx}, this.value)">
          <div class="meta-actions">
            <button class="meta-delete" onclick="removeMetadataField(${idx})">Delete</button>
          </div>
        </div>
      `
    )
    .join('');
}

function addMetadataField() {
  metadataFields.push({ key: '', value: '', include: true });
  renderMetadataFields();
}

function removeMetadataField(idx) {
  metadataFields.splice(idx, 1);
  renderMetadataFields();
}

function toggleMetadataInclude(idx, include) {
  if (!metadataFields[idx]) return;
  metadataFields[idx].include = include;
}

function updateMetadataKey(idx, value) {
  if (!metadataFields[idx]) return;
  metadataFields[idx].key = value;
}

function updateMetadataValue(idx, value) {
  if (!metadataFields[idx]) return;
  metadataFields[idx].value = value;
}

function handleMetadataDragStart(idx) {
  metadataDragIndex = idx;
  const row = document.querySelector(`.metadata-row[data-index="${idx}"]`);
  if (row) row.classList.add('dragging');
}

function handleMetadataDragOver(event, idx) {
  event.preventDefault();
  const row = document.querySelector(`.metadata-row[data-index="${idx}"]`);
  if (row) row.classList.add('drag-target');
}

function handleMetadataDragLeave(idx) {
  const row = document.querySelector(`.metadata-row[data-index="${idx}"]`);
  if (row) row.classList.remove('drag-target');
}

function handleMetadataDrop(idx) {
  if (metadataDragIndex === null || idx === metadataDragIndex) return;
  const from = metadataDragIndex;
  let to = idx;
  if (from < to) to -= 1; // account for removal shift
  const [moved] = metadataFields.splice(from, 1);
  metadataFields.splice(to, 0, moved);
  metadataDragIndex = null;
  renderMetadataFields();
}

function handleMetadataDragEnd() {
  metadataDragIndex = null;
  document.querySelectorAll('.metadata-row').forEach(row => row.classList.remove('dragging', 'drag-target'));
}

function syncMetadataFromDOM() {
  const rows = document.querySelectorAll('#metadataModalFieldList .metadata-row');
  if (!rows || !rows.length) return;
  metadataFields = Array.from(rows).map(row => {
    const include = row.querySelector('.meta-include')?.checked ?? true;
    const key = row.querySelector('.meta-key')?.value || '';
    const value = row.querySelector('.meta-value')?.value || '';
    return { key, value, include };
  });
}

function loadMetadata() {
  const ctx = getFormContext();
  const params = new URLSearchParams({
    subject: ctx.subject,
    session: ctx.session,
    task: ctx.task,
    run: ctx.run,
    modality: ctx.modality,
    sampling_rate: ctx.sampling_rate,
    channels: ctx.channels
  });
  setMetadataStatus('Loading...', false);
  return fetch(`/metadata?${params.toString()}`)
    .then(parseJsonResponse)
    .then(data => {
      metadataFields = (data.fields || []).map(normalizeField);
      renderMetadataFields();
      setMetadataStatus('Metadata loaded');
    })
    .catch(error => {
      console.error('[client] Error loading metadata:', error);
      setMetadataStatus('Failed to load metadata', true);
    });
}

function saveMetadata(silent = false) {
  syncMetadataFromDOM();
  const ctx = getFormContext();
  const payload = {
    subject: ctx.subject,
    session: ctx.session,
    task: ctx.task,
    run: ctx.run,
    modality: ctx.modality,
    fields: metadataFields
  };
  if (!silent) {
    setMetadataStatus('Saving...', false);
  }
  return fetch('/metadata', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
    .then(parseJsonResponse)
    .then(data => {
      if (!silent) {
        setMetadataStatus('Metadata saved');
      }
      return data;
    })
    .catch(error => {
      console.error('[client] Error saving metadata:', error);
      if (!silent) {
        setMetadataStatus('Failed to save metadata', true);
      }
    });
}

// Socket.IO event handlers
socket.on('connect', () => console.log('[client] Socket.IO connected'));
socket.on('disconnect', () => console.log('[client] Socket.IO disconnected'));

socket.on('analysis_stopped', () => {
  document.getElementById('startBtn').disabled = false;
  document.getElementById('stopBtn').disabled = true;
});

socket.on('eeg_data', data => {
  updateChart(data);
});

function updateChart(data) {
  if (!chart || !data.channels || !Array.isArray(data.channels)) {
    return;
  }

  const enabledChannels = getEnabledChannels();
  const smoothingEnabled = document.getElementById('smoothing_enabled').checked;
  const samplingRate = data.sampling_rate || parseInt(document.getElementById('sampling_rate').value, 10) || 250;
  syncWaveformBuffers(enabledChannels);

  const processedChannels = [];
  data.channels.forEach((channelData, index) => {
    if (index >= enabledChannels || !Array.isArray(channelData)) return;

    let processed = channelData;
    if (smoothingEnabled) {
      processed = smoothSeries(channelData);
    }
    processedChannels[index] = processed;
  });

  const chunkLength = Math.max(...processedChannels.map(ch => (Array.isArray(ch) ? ch.length : 0)), 0);
  if (!chunkLength || samplingRate <= 0) return;

  const times = Array.from({ length: chunkLength }, (_, i) => lastSampleTimestamp + (i + 1) / samplingRate);
  if (times.length) {
    lastSampleTimestamp = times[times.length - 1];
  }
  const cutoff = Math.max(0, lastSampleTimestamp - waveformWindowSec);

  processedChannels.forEach((series, index) => {
    if (!Array.isArray(series) || index >= enabledChannels) return;
    const buffer = waveformBuffers[index];
    const count = Math.min(series.length, times.length);
    for (let i = 0; i < count; i += 1) {
      buffer.push({ x: times[i], y: series[i] });
    }
    // Trim to keep only the recent window
    while (buffer.length && buffer[0].x < cutoff) {
      buffer.shift();
    }
  });

  waveformBuffers.forEach(buffer => {
    if (!Array.isArray(buffer)) return;
    while (buffer.length && buffer[0].x < cutoff) {
      buffer.shift();
    }
  });

  renderWaveform();
  updateMetricsAndFeatures(processedChannels.slice(0, enabledChannels).map(ch => ch || []), samplingRate);
}

function smoothSeries(series) {
  const windowSize = 3;
  return series.map((val, idx) => {
    const start = Math.max(0, idx - windowSize + 1);
    const subset = series.slice(start, idx + 1);
    const mean = subset.reduce((sum, v) => sum + v, 0) / subset.length;
    return mean;
  });
}

function renderWaveform() {
  if (!chart) return;
  const enabled = getEnabledChannels();
  syncWaveformBuffers(enabled);
  const windowStart = Math.max(0, lastSampleTimestamp - waveformWindowSec);
  const layout = chartMode === 'stacked' ? computeStackedLayout(enabled, windowStart) : null;
  const offsets = chartMode === 'stacked' && layout ? layout.offsets : [];
  stackedOffsets = offsets;

  chart.data.datasets.forEach((dataset, idx) => {
    dataset.hidden = idx >= enabled ? true : !channelVisibility[idx];
    const buffer = waveformBuffers[idx] || [];
    const visiblePoints = buffer.filter(pt => pt.x >= windowStart);
    const renderPoints =
      chartMode === 'stacked'
        ? visiblePoints.map(pt => ({ x: pt.x - windowStart, y: pt.y + (offsets[idx] || 0) }))
        : visiblePoints.map(pt => ({ x: pt.x - windowStart, y: pt.y }));
    dataset.data = renderPoints;
    dataset.borderWidth = chartMode === 'stacked' ? 1.2 : 1.6;
  });

  chart.options.scales.x.min = 0;
  chart.options.scales.x.max = waveformWindowSec;
  if (chartMode === 'stacked' && layout) {
    chart.options.scales.y.display = false;
    chart.options.scales.y.min = layout.min;
    chart.options.scales.y.max = layout.max;
    chart.options.plugins.legend.display = false;
    chart.options.layout = { padding: { left: 70, right: 10, top: 10, bottom: 10 } };
  } else {
    const range = getYAxisRange();
    chart.options.scales.y.display = true;
    chart.options.scales.y.title.text = getYAxisLabel();
    chart.options.scales.y.min = range.min;
    chart.options.scales.y.max = range.max;
    chart.options.plugins.legend.display = true;
    chart.options.layout = { padding: { left: 0, right: 0, top: 0, bottom: 0 } };
  }
  chart.update('none');
}

function computeStackedLayout(enabledChannels, windowStart) {
  const defaultSpacing = getDefaultStackSpacing();
  let maxAbs = 0;
  waveformBuffers.slice(0, enabledChannels).forEach(buf => {
    buf.forEach(pt => {
      if (pt.x >= windowStart) {
        const absVal = Math.abs(pt.y);
        if (absVal > maxAbs) maxAbs = absVal;
      }
    });
  });

  const spacing = Math.max(defaultSpacing, maxAbs * 2.5 || defaultSpacing);
  const mid = (enabledChannels - 1) / 2;
  const offsets = channelColors.map((_, idx) => (mid - idx) * spacing);

  let minVal = Number.POSITIVE_INFINITY;
  let maxVal = Number.NEGATIVE_INFINITY;
  waveformBuffers.slice(0, enabledChannels).forEach((buf, idx) => {
    buf.forEach(pt => {
      if (pt.x < windowStart) return;
      const shifted = pt.y + offsets[idx];
      if (shifted < minVal) minVal = shifted;
      if (shifted > maxVal) maxVal = shifted;
    });
  });

  if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) {
    minVal = -spacing;
    maxVal = spacing;
  }
  const pad = spacing * 0.6;
  return { offsets, min: minVal - pad, max: maxVal + pad };
}

function getDefaultStackSpacing() {
  if (currentSignalType === 'emg') return 300;
  if (currentSignalType === 'motion') return 3;
  return 180;
}

function updateMetricsAndFeatures(channels, samplingRate) {
  const powerPerChannel = channels.map(ch => (ch.length ? ch.reduce((s, v) => s + v * v, 0) / ch.length : 0));
  const avgPower = powerPerChannel.length
    ? powerPerChannel.reduce((s, v) => s + v, 0) / powerPerChannel.length
    : 0;
  const activation = computeActivation(channels);
  const snr = computeSNR(channels);
  const quality = Math.max(0, Math.min(1, snr / 3));

  const powerUnits = currentSignalType === 'motion' ? 'a.u.^2' : 'uV^2';
  setText('avgPower', `${avgPower.toFixed(1)} ${powerUnits}`);
  const activationUnits = currentSignalType === 'emg' ? 'mV' : currentSignalType === 'motion' ? 'a.u.' : 'uV';
  setText('activation', `${activation.toFixed(2)} ${activationUnits}`);
  setText('snr', snr.toFixed(2));
  setText('impedance', `${Math.round(quality * 100)}%`);

  const qualityOverlay = document.getElementById('qualityOverlay');
  if (qualityOverlay) {
    qualityOverlay.style.display = quality < 0.35 ? 'block' : 'none';
  }

  updateFeatureCharts(channels, samplingRate, avgPower);
}

function computeActivation(channels) {
  if (!channels.length) return 0;
  const rmsValues = channels.map(ch => {
    if (!ch.length) return 0;
    const meanSquare = ch.reduce((s, v) => s + v * v, 0) / ch.length;
    return Math.sqrt(meanSquare);
  });
  const meanRms = rmsValues.reduce((s, v) => s + v, 0) / rmsValues.length;
  return meanRms;
}

function computeSNR(channels) {
  if (!channels.length) return 0;
  const snrValues = channels.map(ch => {
    if (!ch.length) return 0;
    const mean = ch.reduce((s, v) => s + v, 0) / ch.length;
    const variance = ch.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / ch.length;
    const std = Math.sqrt(variance);
    if (std === 0) return 0;
    return Math.abs(mean) / (std + 1e-6);
  });
  const meanSnr = snrValues.reduce((s, v) => s + v, 0) / snrValues.length;
  return meanSnr;
}

// Compute simple band powers over the latest chunk
function computeBandPowers(channels, samplingRate) {
  const bands = [
    { name: 'delta', low: 1, high: 4 },
    { name: 'theta', low: 4, high: 8 },
    { name: 'alpha', low: 8, high: 13 },
    { name: 'beta', low: 13, high: 30 },
    { name: 'gamma', low: 30, high: 45 }
  ];
  if (!channels || !channels.length || samplingRate <= 0) {
    return bands.map(() => 0);
  }
  const bandTotals = bands.map(() => 0);
  let counted = 0;

  channels.forEach(ch => {
    if (!ch || ch.length < 8) return;
    const psd = computePSD(ch, samplingRate);
    const freqs = psd.freqs;
    const power = psd.power;
    bands.forEach((band, idx) => {
      let sum = 0;
      let count = 0;
      for (let i = 0; i < freqs.length; i++) {
        if (freqs[i] >= band.low && freqs[i] <= band.high) {
          sum += power[i];
          count += 1;
        }
      }
      bandTotals[idx] += count > 0 ? sum / count : 0;
    });
    counted += 1;
  });

  if (counted === 0) return bands.map(() => 0);
  return bandTotals.map(v => v / counted);
}

// Naive PSD using DFT; adequate for short windows
function computePSD(series, samplingRate) {
  const N = series.length;
  const mean = series.reduce((s, v) => s + v, 0) / N;
  const centered = series.map(v => v - mean);
  const freqs = [];
  const power = [];
  for (let k = 0; k <= Math.floor(N / 2); k++) {
    let re = 0;
    let im = 0;
    for (let n = 0; n < N; n++) {
      const angle = (-2 * Math.PI * k * n) / N;
      re += centered[n] * Math.cos(angle);
      im += centered[n] * Math.sin(angle);
    }
    const mag = (re * re + im * im) / N;
    const freq = (samplingRate * k) / N;
    freqs.push(freq);
    power.push(mag);
  }
  return { freqs, power };
}

function updateFeatureCharts(channels, samplingRate, meanPower) {
  if (featureTimeChart) {
    featureHistory.push(meanPower || 0);
    if (featureHistory.length > 100) featureHistory = featureHistory.slice(-100);
    featureTimeChart.data.labels = featureHistory.map((_, i) => i);
    featureTimeChart.data.datasets[0].data = [...featureHistory];
    featureTimeChart.data.datasets[1].data = featureHistory.map(() => featureHistory.reduce((a, b) => a + b, 0) / (featureHistory.length || 1));
    featureTimeChart.update('none');
  }

  if (bandPowerChart && channels && channels.length) {
    const bandPowers = computeBandPowers(channels, samplingRate);
    bandPowerChart.data.datasets[0].data = bandPowers;
    bandPowerChart.options.scales.y.max = BANDPOWER_Y_MAX;
    bandPowerChart.update('none');
  }
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

// Start calibration
function startCalibration() {
  fetch('/calibrate', { method: 'POST' })
    .then(parseJsonResponse)
    .then(data => {
      if (data.values) {
        alert('Calibration complete!\nBaseline values: ' + data.values.map(v => v.toFixed(2)).join(', '));
      } else {
        alert('Calibration complete!');
      }
    })
    .catch(error => {
      console.error('[client] Error during calibration:', error);
      alert('Calibration failed. Check console for details.');
    });
}

// Export data
function exportData() {
  const samplingRate = parseInt(document.getElementById('sampling_rate').value, 10) || 250;
  const subject = document.getElementById('recSubject').value || '01';
  const session = document.getElementById('recSession').value || '01';
  const task = document.getElementById('recTask').value || 'resting';
  const run = document.getElementById('recRun').value || '01';
  const modality = document.getElementById('recModality').value || currentSignalType;
  const params = new URLSearchParams({
    num_rows: 5000,
    sampling_rate: samplingRate,
    subject,
    session,
    task,
    run,
    modality
  });
  window.location.href = `/export-data?${params.toString()}`;
}
