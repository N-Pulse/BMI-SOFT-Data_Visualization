// Socket.IO connection — real-time bidirectional link between browser and Flask server
const socket = io();

// ─────────────────────────────────────────────────────────────────────────────
// Channel capacity constants
// ─────────────────────────────────────────────────────────────────────────────

// Maximum number of EEG channels the UI supports.
// Standard high-density EEG caps (e.g. 10–20 system, actiCAP) go up to 64.
// The PiEEG board is limited to 8, but the DSI-24 headset uses 24 channels.
// Setting this to 64 makes the code forward-compatible with any device we add.
const MAX_EEG_CHANNELS = 64;

// Maximum EMG channels — the upcoming N-Pulse bracelet prototype is expected
// to provide 8–16 channels. 16 gives enough headroom for future revisions.
const MAX_EMG_CHANNELS = 16;

// ─────────────────────────────────────────────────────────────────────────────
// Colour palettes — generated programmatically so they scale to any channel count
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Generate an array of `count` perceptually distinct HSL colours.
 *
 * We use the golden-angle increment (≈137.5°) to distribute hues around the
 * colour wheel.  This maximises the perceptual distance between *adjacent*
 * channel indices — so channels 1 and 2 look very different, channels 1 and 3
 * also look different, etc.  It is the same technique used in many scientific
 * data-visualisation libraries (e.g. D3, matplotlib's tab20).
 *
 * @param {number} count      Number of colours to generate.
 * @param {number} hueOffset  Starting hue in degrees (0 = red).
 * @param {number} sat        Saturation % (0-100).
 * @param {number} lightBase  Base lightness % — odd indices get +8 for extra separation.
 * @returns {string[]}        Array of CSS `hsl(…)` strings.
 */
function generateColorPalette(count, hueOffset = 0, sat = 75, lightBase = 60) {
  return Array.from({ length: count }, (_, i) => {
    const hue   = Math.round((hueOffset + i * 137.508) % 360); // golden angle
    const light = lightBase + (i % 2) * 8;                     // alternate lightness
    return `hsl(${hue}, ${sat}%, ${light}%)`;
  });
}

// EEG channels — full hue wheel, 64 entries.
// Each channel gets a distinct colour even at high density (e.g. 64-channel cap).
const channelColors = generateColorPalette(MAX_EEG_CHANNELS, 0, 75, 60);

// EMG channels — offset by 160° so the EMG panel palette starts in the
// blue-green range.  This makes it immediately obvious which panel is EEG
// and which is EMG even when glancing at the screen quickly.
const emgChannelColors = generateColorPalette(MAX_EMG_CHANNELS, 160, 85, 62);

const headChannels = [0, 1, 2, 3];
const armChannels = [4, 5, 6, 7];

// ─────────────────────────────────────────────────────────────────────────────
// EEG chart state (primary panel)
// ─────────────────────────────────────────────────────────────────────────────

// One visibility flag per channel.  Initialised to all-visible for the full
// MAX_EEG_CHANNELS capacity; the UI shows only the 'enabled' subset.
let channelVisibility = Array(MAX_EEG_CHANNELS).fill(true);
let chart = null;
let bandPowerChart = null;
let featureTimeChart = null;
let featureHistory = [];
let isSimulationMode = false;
let currentSignalType = 'eeg';
let chartMode = 'stacked';
let isRecording = false;
let isRecordingPaused = false;
let markers = [];
let simulationEvents = [];
let simulationDurationSec = 0;
let simulationEventColumns = [];
let simulationSidecarSummary = {};
let simulationChannelsInfo = {};
let metadataFields = [];
let metadataDragIndex = null;
const BANDPOWER_Y_MAX = 2000; // fixed y-axis for band power chart
const waveformWindowSec = 8; // seconds of data to keep in the live view
const MAX_EVENT_OVERLAY_LINES = 120;
const MAX_RENDER_POINTS = 1200;
const FEATURE_UPDATE_INTERVAL_MS = 400;

// One rolling buffer per channel, pre-allocated to the maximum channel count.
// Each buffer holds { x: timeSeconds, y: amplitude } objects for the last
// waveformWindowSec seconds.  syncWaveformBuffers() keeps this in sync with
// the currently enabled channel count.
let waveformBuffers = Array.from({ length: MAX_EEG_CHANNELS }, () => []);
let lastSampleTimestamp = 0;
let stackedOffsets = [];
let currentWindowStartSec = 0;
let lastFeatureUpdateMs = 0;

// ── View mode: 'live' vs 'freeze' ────────────────────────────────────────────
// 'live'   – chart window auto-follows the newest incoming data (default)
// 'freeze' – chart is pinned at the moment Freeze was clicked.
//            For file replay this also pauses the server so no new data
//            arrives and you can analyse that exact moment.
//            Click Live (or ↩ Resume) to continue from where you stopped.
let viewMode = 'live';
let freezePositionSec = 0;  // timestamp (s) at which the chart was frozen

// ── File-replay player state ──────────────────────────────────────────────────
// Only relevant when streaming a .fif / .edf / .json / .xdf file.
// isFileReplayMode is set to true when the server emits 'playback_info'.
let isFileReplayMode    = false;
let playerTotalSec      = 0;
let playerTotalSamples  = 0;
let playerSamplingRate  = 250;
let playerScrubDragging = false;  // suppress progress ticks while dragging

// ─────────────────────────────────────────────────────────────────────────────
// EMG chart state (secondary panel — only active in dual-stream mode)
//
// These mirror the EEG variables but are fully independent so the two
// streams never interfere with each other.  The EMG panel can have a
// different number of channels, its own display mode, and its own rolling
// data buffers.
// ─────────────────────────────────────────────────────────────────────────────
let emgChart = null;
let emgChartMode = 'stacked';  // 'stacked' (default) or 'overlap'
let isDualStreamMode = false;
let emgEnabledChannels = 8;

// Rolling waveform buffers for the EMG panel — same structure as waveformBuffers.
// Each entry is an array of { x: timeSeconds, y: amplitude } objects.
// Pre-allocated to MAX_EMG_CHANNELS so resizing is never needed at runtime.
let emgWaveformBuffers = Array.from({ length: MAX_EMG_CHANNELS }, () => []);
let emgChannelVisibility = Array(MAX_EMG_CHANNELS).fill(true);
let emgLastSampleTimestamp = 0;
let emgStackedOffsets = [];

// Initialize the UI on page load
document.addEventListener('DOMContentLoaded', () => {
  currentSignalType = window.initialSignalType || 'eeg';

  const modeSelect = document.getElementById('waveformModeSelect');
  if (modeSelect) chartMode = modeSelect.value || 'stacked';

  // Ensure view-mode button state matches the JS default
  setViewMode('live');

  buildChannelSelectors();
  initializeChart();
  initializeFeatureCharts();
  updateSignalUI();
  updateSettings();
  loadMetadata();
  renderMarkers();
  renderSessionSummary();

  // If dual-stream mode was already active (e.g. page reload), set up the EMG panel.
  // In practice the server resets dual_stream_mode on startup, so this is mostly
  // a defensive initialisation for the UI state.
  if (isDualStreamMode) {
    buildEmgChannelSelectors();
    initializeEmgChart();
    const section = document.getElementById('emgChartSection');
    if (section) section.style.display = '';
  }
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

/**
 * Keep waveformBuffers and channelVisibility arrays in sync with the current
 * enabled channel count.
 *
 * Called every time the user changes the channel count (via updateSettings).
 * We grow the arrays when needed and shrink channelVisibility to the active
 * count so the grid does not render cells beyond what the hardware/simulation
 * is producing.
 *
 * @param {number} enabled  Current number of active EEG channels (1–64).
 */
function syncWaveformBuffers(enabled) {
  // Grow the buffer array up to the full capacity if necessary
  while (waveformBuffers.length < MAX_EEG_CHANNELS) waveformBuffers.push([]);
  waveformBuffers = waveformBuffers.slice(0, MAX_EEG_CHANNELS);
  for (let i = 0; i < waveformBuffers.length; i += 1) {
    if (!Array.isArray(waveformBuffers[i])) waveformBuffers[i] = [];
  }
  // Grow visibility array to cover all capacity slots (default: visible)
  while (channelVisibility.length < MAX_EEG_CHANNELS) channelVisibility.push(true);
}

/** Clear all EEG waveform buffers and reset the time counter. */
function resetWaveformBuffers() {
  waveformBuffers = Array.from({ length: MAX_EEG_CHANNELS }, () => []);
  lastSampleTimestamp = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Channel grid  —  replaces the old pill list
//
// Design rationale
// ─────────────────
// The old "Ch 1 … Ch 8" pill row works for 8 channels but becomes unusable
// at 24 or 64 channels (the pills overflow and you cannot tell channels apart).
//
// The new grid:
//   • Shows all enabled channels as small numbered squares in 8 columns.
//   • Each square uses the channel's trace colour as its border.
//   • Active (visible) channels are fully opaque; hidden ones are dimmed.
//   • Three bulk-control buttons (All / None / Invert) let a researcher
//     quickly isolate a region or reset visibility without clicking 64 times.
//   • Scales from 1 to 64 channels without any layout changes.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Rebuild the channel-visibility grid inside #channelList.
 *
 * Call this after the enabled channel count changes or on initial page load.
 * The function also syncs the underlying Chart.js dataset visibility so the
 * waveform chart immediately reflects the current grid state.
 */
function buildChannelSelectors() {
  const container = document.getElementById('channelList');
  if (!container) return;

  const enabled = getEnabledChannels();
  syncWaveformBuffers(enabled);

  // ── Bulk-control toolbar ──────────────────────────────────────────────────
  // Three small buttons that operate on all enabled channels at once.
  // Useful when a researcher wants to isolate one channel (None → click one)
  // or reset after exploring (All).
  container.innerHTML = `
    <div class="ch-grid-controls">
      <button onclick="selectAllChannels(true)"  title="Show all channels">All</button>
      <button onclick="selectAllChannels(false)" title="Hide all channels">None</button>
      <button onclick="invertChannelSelection()" title="Flip visibility of every channel">Invert</button>
    </div>
    <div id="chGrid" class="ch-grid"></div>
  `;

  // ── Channel cells ─────────────────────────────────────────────────────────
  const grid = container.querySelector('#chGrid');
  for (let idx = 0; idx < enabled; idx++) {
    const cell = document.createElement('div');
    cell.className = 'ch-cell' + (channelVisibility[idx] ? ' active' : '');
    cell.textContent = idx + 1;          // show channel number (1-based)
    cell.title = `Channel ${idx + 1} — click to toggle visibility`;
    cell.style.borderColor = channelColors[idx];  // matches the trace colour
    cell.dataset.index = idx;
    cell.onclick = () => toggleChannel(idx, cell);
    grid.appendChild(cell);
  }

  // Sync the Chart.js dataset hidden-flags to match the current visibility state
  if (chart) {
    chart.data.datasets.forEach((ds, idx) => {
      ds.hidden = idx >= enabled ? true : !channelVisibility[idx];
    });
    renderWaveform();
  }
}

/**
 * Toggle an individual EEG channel on or off.
 *
 * @param {number} index  Zero-based channel index.
 * @param {Element} cell  The grid cell DOM element (gets the 'active' class toggled).
 */
function toggleChannel(index, cell) {
  channelVisibility[index] = !channelVisibility[index];
  if (cell) cell.classList.toggle('active', channelVisibility[index]);
  if (chart && chart.data.datasets[index]) {
    chart.data.datasets[index].hidden = !channelVisibility[index];
    renderWaveform();
  }
}

/**
 * Show or hide all currently enabled EEG channels at once.
 *
 * @param {boolean} visible  true = show all, false = hide all.
 */
function selectAllChannels(visible) {
  const enabled = getEnabledChannels();
  for (let i = 0; i < enabled; i++) {
    channelVisibility[i] = visible;
    if (chart && chart.data.datasets[i]) {
      chart.data.datasets[i].hidden = !visible;
    }
  }
  // Refresh cell appearance in the grid
  document.querySelectorAll('#chGrid .ch-cell').forEach((cell, i) => {
    cell.classList.toggle('active', channelVisibility[i]);
  });
  renderWaveform();
}

/**
 * Flip the visibility of every enabled EEG channel.
 * Useful for quickly switching focus between two groups (e.g. frontal vs occipital).
 */
function invertChannelSelection() {
  const enabled = getEnabledChannels();
  for (let i = 0; i < enabled; i++) {
    channelVisibility[i] = !channelVisibility[i];
    if (chart && chart.data.datasets[i]) {
      chart.data.datasets[i].hidden = !channelVisibility[i];
    }
  }
  document.querySelectorAll('#chGrid .ch-cell').forEach((cell, i) => {
    cell.classList.toggle('active', channelVisibility[i]);
  });
  renderWaveform();
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

const eventOverlayPlugin = {
  id: 'eventOverlay',
  afterDatasetsDraw(chartInstance) {
    if (!simulationEvents.length) return;
    const { ctx, chartArea, scales } = chartInstance;
    if (!ctx || !chartArea || !scales?.x) return;

    const visible = getVisibleSimulationEvents(currentWindowStartSec, waveformWindowSec);
    if (!visible.length) return;

    ctx.save();
    ctx.strokeStyle = 'rgba(255, 193, 7, 0.85)';
    ctx.fillStyle = 'rgba(255, 193, 7, 0.95)';
    ctx.lineWidth = 1;
    ctx.font = '11px Arial';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';

    visible.forEach((evt, idx) => {
      const xRelative = evt.plotTime - currentWindowStartSec;
      const x = scales.x.getPixelForValue(xRelative);
      if (!Number.isFinite(x) || x < chartArea.left || x > chartArea.right) return;
      ctx.beginPath();
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();
      if (idx < 24) {
        const label = `${evt.label}`;
        ctx.fillText(label, x + 3, chartArea.top + 4 + ((idx % 8) * 12));
      }
    });

    ctx.restore();
  }
};

function getVisibleSimulationEvents(windowStart, windowLength) {
  if (!simulationEvents.length) return [];
  const windowEnd = windowStart + windowLength;
  const duration = Number(simulationDurationSec) || 0;
  const visible = [];

  simulationEvents.forEach(evt => {
    const onset = Number(evt.onset);
    if (!Number.isFinite(onset)) return;

    if (duration > 0) {
      let cycle = Math.floor((windowStart - onset) / duration);
      if (!Number.isFinite(cycle)) cycle = 0;
      let plotTime = onset + cycle * duration;
      while (plotTime < windowStart) {
        cycle += 1;
        plotTime = onset + cycle * duration;
      }
      while (plotTime <= windowEnd) {
        visible.push({ ...evt, plotTime });
        if (visible.length >= MAX_EVENT_OVERLAY_LINES) break;
        cycle += 1;
        plotTime = onset + cycle * duration;
      }
    } else if (onset >= windowStart && onset <= windowEnd) {
      visible.push({ ...evt, plotTime: onset });
    }
  });

  visible.sort((a, b) => a.plotTime - b.plotTime);
  return visible.slice(0, MAX_EVENT_OVERLAY_LINES);
}

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
        title: { display: true, text: 'Time' },
        ticks: {
          maxTicksLimit: 6,
          // Show absolute file time (m:ss) so the position is clear after a seek.
          // currentWindowStartSec is added to each relative tick value so that
          // e.g. after seeking to 2:41 the axis reads "2:41  2:43  2:45  2:47  2:49"
          // rather than "0  2  4  6  8".
          callback: value => {
            const abs = currentWindowStartSec + Number(value);
            if (!Number.isFinite(abs)) return value;
            const m   = Math.floor(abs / 60);
            const s   = Math.floor(abs % 60);
            return `${m}:${s.toString().padStart(2, '0')}`;
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
      // One dataset per channel, pre-created for all MAX_EEG_CHANNELS.
      // Datasets beyond 'enabled' start hidden; they unhide if the user raises
      // the channel count, so Chart.js never needs to be fully rebuilt.
      datasets: Array.from({ length: MAX_EEG_CHANNELS }, (_, idx) => ({
        label: `${currentSignalType.toUpperCase()} Channel ${idx + 1}`,
        data: [],
        borderColor: channelColors[idx],
        borderWidth: 1.4,
        tension: 0.05,
        pointRadius: 0,
        hidden: idx >= enabled ? true : !channelVisibility[idx]
      }))
    },
    options: buildChartOptions(),
    plugins: [stackedLabelPlugin, eventOverlayPlugin]
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

  // If the user switches to hardware mode while the hardware-mode warning is
  // showing, keep it visible (it's still relevant).  If they switch back to
  // simulation mode, hide it — dual-stream is now available again.
  if (isSimulationMode) {
    const hardwareMsg = document.getElementById('dualModeHardwareMsg');
    if (hardwareMsg) hardwareMsg.style.display = 'none';
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
      // If dual-stream mode is on, reset the EMG buffers too so the second
      // panel starts fresh at the same moment as the EEG panel.
      if (isDualStreamMode) {
        resetEmgWaveformBuffers();
        renderEmgWaveform();
      }
      document.getElementById('startBtn').disabled = true;
      document.getElementById('stopBtn').disabled = false;
      // Lock the dual-mode toggle while streaming — the server rejects changes mid-stream
      const dualToggle = document.getElementById('dualModeToggle');
      if (dualToggle) dualToggle.disabled = true;
      // Reset to live view whenever a new stream starts
      setViewMode('live');
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
      // Re-enable the dual-mode toggle now that the stream has stopped
      const dualToggle = document.getElementById('dualModeToggle');
      if (dualToggle) dualToggle.disabled = false;
      // Hide the file-replay player bar and reset replay state
      playerTeardown();
      // Snap back to live view
      setViewMode('live');
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
  const lowerName = file.name.toLowerCase();
  const isSupported =
    lowerName.endsWith('.json') ||
    lowerName.endsWith('.fif') ||
    lowerName.endsWith('.edf') ||
    lowerName.endsWith('.xdf') ||
    lowerName.includes('.xdf.');

  if (!isSupported) {
    fileStatus.textContent = 'Error: Supported formats are .json, .fif, .edf, .xdf';
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
        simulationDurationSec = Number(metadata.duration_seconds) || 0;
        fileStatus.innerHTML = `
          <strong>✓ File loaded successfully!</strong><br>
          Channels: ${metadata.num_channels}<br>
          Duration: ${metadata.duration_seconds}s<br>
          Samples: ${metadata.num_samples}<br>
          Sampling Rate: ${metadata.sampling_rate} Hz
        `;
        fileStatus.style.color = '#4CAF50';
        renderWaveform();
        renderMarkers();
        renderSessionSummary();
      } else {
        fileStatus.textContent = `Error: ${data.message}`;
        fileStatus.style.color = '#f44336';
      }
    })
    .catch(error => {
      console.error('[client] Error uploading file:', error);
      const message =
        (error && typeof error === 'object' && error.message)
          ? error.message
          : 'Upload failed. Check server logs.';
      fileStatus.textContent = `Error: ${message}`;
      fileStatus.style.color = '#f44336';
    });
}

function uploadSessionFolder() {
  const folderInput = document.getElementById('sessionFolderInput');
  const fileStatus = document.getElementById('fileStatus');
  const eventStatus = document.getElementById('eventFileStatus');
  if (!folderInput || !fileStatus) return;

  const files = folderInput.files ? Array.from(folderInput.files) : [];
  if (!files.length) {
    fileStatus.textContent = 'No folder files selected';
    fileStatus.style.color = '#f44336';
    return;
  }

  fileStatus.textContent = `Uploading folder (${files.length} files)...`;
  fileStatus.style.color = '#ff9800';
  if (eventStatus) {
    eventStatus.textContent = 'Parsing session files...';
    eventStatus.style.color = '#ff9800';
  }

  const formData = new FormData();
  files.forEach(file => {
    const relative = file.webkitRelativePath || file.name;
    formData.append('files', file, relative);
  });

  fetch('/upload-simulation-session', {
    method: 'POST',
    body: formData
  })
    .then(parseJsonResponse)
    .then(data => {
      if (data.status !== 'success') {
        fileStatus.textContent = `Error: ${data.message}`;
        fileStatus.style.color = '#f44336';
        if (eventStatus) {
          eventStatus.textContent = '';
        }
        return;
      }

      const metadata = data.metadata || {};
      simulationDurationSec = Number(metadata.duration_seconds) || 0;
      simulationEvents = Array.isArray(data.events)
        ? data.events.map(evt => ({
            onset: Number(evt.onset) || 0,
            duration: Number(evt.duration) || 0,
            label: evt.label || 'event',
            description: evt.description || '',
            code: evt.code || ''
          }))
        : [];
      simulationEventColumns = Array.isArray(data.event_columns) ? data.event_columns : [];
      simulationSidecarSummary = data.events_sidecar_summary || {};
      simulationChannelsInfo = data.channels_info || {};

      fileStatus.innerHTML = `
        <strong>✓ Session loaded</strong><br>
        Signal: ${escapeHtml(data.detected_files?.signal || 'n/a')}<br>
        Channels: ${metadata.num_channels || '--'}<br>
        Duration: ${metadata.duration_seconds || '--'}s<br>
        Sampling Rate: ${metadata.sampling_rate || '--'} Hz
      `;
      fileStatus.style.color = '#4CAF50';

      if (eventStatus) {
        eventStatus.innerHTML = `
          <strong>✓ Events parsed</strong><br>
          Events: ${Number(data.events_loaded) || 0}<br>
          Events sidecar: ${data.has_events_sidecar ? 'yes' : 'no'}
        `;
        eventStatus.style.color = '#4CAF50';
      }

      renderMarkers();
      renderWaveform();
      renderSessionSummary();
    })
    .catch(error => {
      console.error('[client] Error uploading session folder:', error);
      const message =
        (error && typeof error === 'object' && error.message)
          ? error.message
          : 'Upload failed. Check server logs.';
      fileStatus.textContent = `Error: ${message}`;
      fileStatus.style.color = '#f44336';
      if (eventStatus) {
        eventStatus.textContent = '';
      }
    });
}

function uploadEventsFile() {
  const fileInput = document.getElementById('eventFileInput');
  const fileStatus = document.getElementById('eventFileStatus');
  if (!fileInput || !fileStatus) return;

  if (!fileInput.files || fileInput.files.length === 0) {
    fileStatus.textContent = 'No events file selected';
    fileStatus.style.color = '#f44336';
    return;
  }

  const file = fileInput.files[0];
  const lowerName = file.name.toLowerCase();
  if (!(lowerName.endsWith('.tsv') || lowerName.endsWith('.json'))) {
    fileStatus.textContent = 'Error: Supported event formats are .tsv and .json';
    fileStatus.style.color = '#f44336';
    return;
  }

  fileStatus.textContent = 'Uploading events...';
  fileStatus.style.color = '#ff9800';

  const formData = new FormData();
  formData.append('file', file);

  fetch('/upload-simulation-events', {
    method: 'POST',
    body: formData
  })
    .then(parseJsonResponse)
    .then(data => {
      if (data.status === 'success') {
        if (Array.isArray(data.events)) {
          simulationEvents = data.events
            .map(evt => ({
              onset: Number(evt.onset) || 0,
              duration: Number(evt.duration) || 0,
              label: evt.label || 'event',
              description: evt.description || '',
              code: evt.code || ''
            }))
            .sort((a, b) => a.onset - b.onset);
        }
        simulationEventColumns = Array.isArray(data.columns) ? data.columns : simulationEventColumns;
        simulationSidecarSummary = data.events_sidecar_summary || simulationSidecarSummary;
        fileStatus.innerHTML = `
          <strong>✓ ${escapeHtml(data.message || 'Events loaded')}</strong><br>
          Events loaded: ${Number(data.events_loaded) || 0}
        `;
        fileStatus.style.color = '#4CAF50';
        renderMarkers();
        renderWaveform();
        renderSessionSummary();
      } else {
        fileStatus.textContent = `Error: ${data.message}`;
        fileStatus.style.color = '#f44336';
      }
    })
    .catch(error => {
      console.error('[client] Error uploading events file:', error);
      const message =
        (error && typeof error === 'object' && error.message)
          ? error.message
          : 'Upload failed. Check server logs.';
      fileStatus.textContent = `Error: ${message}`;
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
  const sections = [];

  if (simulationEvents.length) {
    const preview = simulationEvents.slice(0, 40);
    const simRows = preview
      .map(evt => {
        const onset = (Number(evt.onset) || 0).toFixed(2);
        const duration = Number(evt.duration || 0).toFixed(2);
        const codePart = evt.code ? ` [${evt.code}]` : '';
        const descPart = evt.description ? ` - ${evt.description}` : '';
        const left = `${evt.label}${codePart}${descPart}`;
        const right = `${onset}s / ${duration}s`;
        return `<div class="marker-item"><span>${escapeHtml(left)}</span><span>${escapeHtml(right)}</span></div>`;
      })
      .join('');
    const extra = simulationEvents.length > preview.length
      ? `<div class="marker-item"><span>...</span><span>+${simulationEvents.length - preview.length} more</span></div>`
      : '';
    sections.push(`<div class="marker-item"><strong>Loaded events</strong><span>${simulationEvents.length}</span></div>${simRows}${extra}`);
  }

  if (markers.length) {
    const recRows = markers
      .map(m => `<div class="marker-item"><span>${escapeHtml(m.label)}</span><span>${Number(m.offset || 0).toFixed(2)}s</span></div>`)
      .join('');
    sections.push(`<div class="marker-item"><strong>Manual markers</strong><span>${markers.length}</span></div>${recRows}`);
  }

  list.innerHTML = sections.length
    ? sections.join('')
    : '<div class="marker-item"><span>No events/markers loaded</span><span>--</span></div>';
}

function renderSessionSummary() {
  const panel = document.getElementById('sessionInfoPanel');
  if (!panel) return;

  const parts = [];
  if (simulationEventColumns.length) {
    parts.push(`Event columns: ${escapeHtml(simulationEventColumns.join(', '))}`);
  }

  const previewNames = simulationChannelsInfo?.channel_names_preview || [];
  if (Array.isArray(previewNames) && previewNames.length) {
    const suffix = (simulationChannelsInfo.row_count || 0) > previewNames.length
      ? ` ... (+${simulationChannelsInfo.row_count - previewNames.length})`
      : '';
    parts.push(`Channels: ${escapeHtml(previewNames.join(', '))}${escapeHtml(suffix)}`);
  }

  if (simulationSidecarSummary && Object.keys(simulationSidecarSummary).length) {
    if (simulationSidecarSummary.protocol_version !== undefined) {
      parts.push(`Protocol version: ${escapeHtml(simulationSidecarSummary.protocol_version)}`);
    }
    if (simulationSidecarSummary.lsl_marker?.format) {
      parts.push(`LSL code format: ${escapeHtml(simulationSidecarSummary.lsl_marker.format)}`);
    }
  }

  if (!parts.length) {
    panel.innerHTML = '<strong>Session info</strong><br>No extra session metadata loaded yet.';
    return;
  }

  panel.innerHTML = `<strong>Session info</strong><br>${parts.join('<br>')}`;
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

// ═════════════════════════════════════════════════════════════════════════════
// VIEW MODE  —  Live vs Freeze
//
//   Live   – chart auto-scrolls with incoming data (default)
//   Freeze – chart is pinned at the moment Freeze was clicked.
//            For file replay this also pauses the server so no new samples
//            arrive, letting you analyse that exact moment without the signal
//            running away.  Clicking Live resumes from exactly where it paused.
//
//   For live hardware there is no server pause, but the chart stops updating
//   so you can still measure peaks, latencies etc. on the frozen view.
// ═════════════════════════════════════════════════════════════════════════════

/**
 * Switch between 'live' and 'freeze' view modes.
 *
 * @param {string} mode  'live' or 'freeze'
 */
function setViewMode(mode) {
  viewMode = mode;

  const liveBtn   = document.getElementById('viewModeLiveBtn');
  const freezeBtn = document.getElementById('viewModeFreezeBtn');

  if (liveBtn)   liveBtn.classList.toggle('active',   mode === 'live');
  if (freezeBtn) freezeBtn.classList.toggle('active', mode === 'freeze');

  if (mode === 'freeze') {
    // Remember the timestamp so renderWaveform() shows this exact window
    freezePositionSec = Math.max(0, lastSampleTimestamp - waveformWindowSec);

    // For file replay: tell the server to stop emitting so the chart really
    // stays at this exact moment and resumes from here on Live.
    if (isFileReplayMode) {
      fetch('/pause-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paused: true })
      }).catch(err => console.error('[viewMode] Pause failed:', err));
    }
  } else {
    // Resuming — for file replay tell the server to continue streaming
    if (isFileReplayMode) {
      fetch('/pause-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paused: false })
      }).catch(err => console.error('[viewMode] Resume failed:', err));
    }
  }

  renderWaveform();
}

// ═════════════════════════════════════════════════════════════════════════════
// FILE-REPLAY PLAYER BAR
//
// Shown automatically when the server starts replaying a file.
// Lets the user seek to any position in the file.
// Pause/Resume is the Freeze/Live toggle above.
//
//   server ──playback_info──►    playerSetup()       (once at stream start)
//   server ──playback_progress──► playerTick()       (every ~100 ms chunk)
//   drag scrubber ──► onPlayerScrubberCommit() ──► POST /seek-stream
// ═════════════════════════════════════════════════════════════════════════════

/** Show the player bar and store file metadata received from the server. */
function playerSetup(info) {
  isFileReplayMode   = true;
  playerTotalSamples = info.total_samples  || 0;
  playerTotalSec     = info.total_time_sec || 0;
  playerSamplingRate = info.sampling_rate  || 250;

  const bar      = document.getElementById('playerBar');
  const scrubber = document.getElementById('playerScrubber');
  if (!bar || !scrubber) return;

  scrubber.min   = 0;
  scrubber.max   = playerTotalSamples;
  scrubber.value = 0;
  bar.style.display = 'flex';
  playerUpdateTime(0);
}

/** Hide the player bar and reset state. Called on Stop. */
function playerTeardown() {
  isFileReplayMode    = false;
  playerScrubDragging = false;
  const bar = document.getElementById('playerBar');
  if (bar) bar.style.display = 'none';
}

/**
 * Advance the scrubber thumb as file replay progresses.
 * Ignored while the user is dragging to avoid fighting their input.
 */
function playerTick(data) {
  if (!isFileReplayMode || playerScrubDragging) return;
  const scrubber = document.getElementById('playerScrubber');
  if (scrubber) scrubber.value = data.current_sample;
  playerUpdateTime(data.current_time_sec, data.total_time_sec);
}

/** Update the time label (m:ss / m:ss). */
function playerUpdateTime(current, total) {
  const fmt = s => {
    const m = Math.floor((s || 0) / 60);
    const sec = Math.floor((s || 0) % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };
  const el = document.getElementById('playerTime');
  if (el) el.textContent = `${fmt(current)} / ${fmt(total ?? playerTotalSec)}`;
}

/**
 * Live preview while the user drags the scrubber.
 * Only updates the time label — no server request yet.
 */
function onPlayerScrubberDrag(value) {
  playerScrubDragging = true;
  const timeSec = playerSamplingRate > 0 ? parseInt(value, 10) / playerSamplingRate : 0;
  playerUpdateTime(timeSec);
}

/**
 * Seek when the user releases the scrubber thumb.
 *
 * The operations MUST be sequential to avoid a race condition where the
 * server resumes before the seek takes effect (causing it to emit chunks
 * from the wrong position into the freshly-cleared buffer):
 *
 *   1. Pause  – stop the server emitting so no stale chunks arrive
 *   2. Reset  – clear client buffers (safe: server is idle)
 *   3. Seek   – tell the server the new file position
 *   4. Resume – start streaming from the new position
 *
 * Result: ~200–400 ms blank chart, then signal from the exact seek point.
 */
function onPlayerScrubberCommit(value) {
  playerScrubDragging = false;
  const sample      = parseInt(value, 10);
  // Convert sample index → seconds so we can anchor the chart's time axis.
  const seekTimeSec = playerSamplingRate > 0 ? sample / playerSamplingRate : 0;

  const postJson = (url, body) => fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  }).then(parseJsonResponse);

  // Sequential: pause → clear buffers → seek → resume.
  // Keeping these strictly in order prevents the server from sending stale
  // chunks into the freshly-cleared buffer between steps.
  postJson('/pause-stream', { paused: true })
    .then(() => {
      // Clear buffer data so old signal disappears immediately.
      // Crucially: restore lastSampleTimestamp to the seek position AFTER the
      // reset (which would zero it).  This anchors the chart's time axis at the
      // correct file time — e.g. 161 s for 2:41 — so the x-axis labels show
      // "2:41 – 2:49" instead of "0 – 8" when data from the new position arrives.
      resetWaveformBuffers();            // → lastSampleTimestamp = 0
      lastSampleTimestamp = seekTimeSec; // → restore to file position (e.g. 161 s)
      if (isDualStreamMode) resetEmgWaveformBuffers();
      renderWaveform();                  // show blank chart at the correct time

      return postJson('/seek-stream', { sample });
    })
    .then(() => {
      // Resume from the new position; update UI to Live directly
      // (avoids a redundant second /pause-stream call that setViewMode would add).
      viewMode = 'live';
      document.getElementById('viewModeLiveBtn')?.classList.add('active');
      document.getElementById('viewModeFreezeBtn')?.classList.remove('active');
      return postJson('/pause-stream', { paused: false });
    })
    .catch(err => console.error('[player] Seek failed:', err));
}

// ═════════════════════════════════════════════════════════════════════════════
// DUAL-STREAM MODE  —  synchronized EEG + EMG functions
// ═════════════════════════════════════════════════════════════════════════════

/**
 * Toggle the synchronized EEG + EMG dual-stream mode on/off.
 *
 * When enabled:
 *  • Tells the server to start a second EMG acquisition thread on /start-analysis.
 *  • Shows the EMG chart panel and the EMG channel controls.
 *  • Initialises (or destroys) the second Chart.js instance.
 *
 * The server rejects the toggle while streaming is active, so the button
 * is disabled between Start and Stop.
 */
function toggleDualMode() {
  const toggle = document.getElementById('dualModeToggle');
  const hardwareMsg = document.getElementById('dualModeHardwareMsg');

  // Dual-stream is simulation-only for now — the hardware EMG bracelet
  // integration is not yet wired up on the server side.  If the user tries
  // to enable the toggle in hardware mode, revert it immediately and show
  // a clear inline message instead of silently producing a blank EMG panel.
  if (toggle.checked && !isSimulationMode) {
    toggle.checked = false;          // revert the visual state
    if (hardwareMsg) hardwareMsg.style.display = '';
    return;
  }
  // Hide the hardware-mode warning whenever the toggle is turned off or
  // the user is already in simulation mode.
  if (hardwareMsg) hardwareMsg.style.display = 'none';

  isDualStreamMode = toggle.checked;
  emgEnabledChannels = parseInt(document.getElementById('emg_channels')?.value || '8', 10);

  fetch('/toggle-dual-mode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dual_mode: isDualStreamMode, emg_channels: emgEnabledChannels })
  })
    .then(parseJsonResponse)
    .then(() => {
      // Show/hide the EMG panel and channel-count input
      const section = document.getElementById('emgChartSection');
      const config  = document.getElementById('emgChannelConfig');
      if (section) section.style.display = isDualStreamMode ? '' : 'none';
      if (config)  config.style.display  = isDualStreamMode ? '' : 'none';

      if (isDualStreamMode) {
        // Build or rebuild the EMG chart with the current channel count
        buildEmgChannelSelectors();
        initializeEmgChart();
      } else {
        // Clean up the Chart.js instance to free canvas memory
        if (emgChart) { emgChart.destroy(); emgChart = null; }
        resetEmgWaveformBuffers();
      }
    })
    .catch(err => {
      console.error('[client] Error toggling dual mode:', err);
      // Roll back the checkbox on failure (e.g. stream was still running)
      toggle.checked = !isDualStreamMode;
      isDualStreamMode = toggle.checked;
      alert(err.message || 'Cannot change dual-stream mode while streaming is active.');
    });
}

/**
 * Called when the EMG channel count input changes in dual-stream mode.
 * Sends the new count to the server and rebuilds the EMG chart.
 */
function updateDualModeSettings() {
  emgEnabledChannels = parseInt(document.getElementById('emg_channels')?.value || '8', 10);
  if (!isDualStreamMode) return;

  fetch('/toggle-dual-mode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dual_mode: true, emg_channels: emgEnabledChannels })
  })
    .then(parseJsonResponse)
    .then(() => {
      buildEmgChannelSelectors();
      initializeEmgChart();
    })
    .catch(err => console.error('[client] Error updating EMG channel count:', err));
}

/** Returns the current number of active EMG channels. */
function getEmgEnabledChannels() {
  return emgEnabledChannels;
}

/** Clear all EMG waveform buffers and reset the time counter. */
function resetEmgWaveformBuffers() {
  emgWaveformBuffers = Array.from({ length: MAX_EMG_CHANNELS }, () => []);
  emgLastSampleTimestamp = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// EMG chart initialisation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create (or recreate) the Chart.js instance for the EMG panel.
 *
 * The chart is an independent line chart with one dataset per EMG channel.
 * Default display mode is 'stacked' — each channel is offset vertically so
 * individual muscle activations are easy to read (same as EEG stacked mode).
 */
function initializeEmgChart() {
  const canvas = document.getElementById('emgChart');
  if (!canvas) return;

  if (emgChart) { emgChart.destroy(); }

  const enabled = getEmgEnabledChannels();
  emgChart = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: {
      // One dataset per channel, pre-created for all MAX_EMG_CHANNELS.
      // Datasets beyond 'enabled' are hidden; they become visible if the user
      // increases the channel count without reloading the page.
      datasets: Array.from({ length: MAX_EMG_CHANNELS }, (_, idx) => ({
        label: `EMG Ch ${idx + 1}`,
        data: [],
        borderColor: emgChannelColors[idx] || '#aaa',
        borderWidth: 1.3,
        tension: 0.05,
        pointRadius: 0,
        hidden: idx >= enabled ? true : !emgChannelVisibility[idx]
      }))
    },
    options: buildEmgChartOptions(),
    plugins: [emgStackedLabelPlugin]
  });
}

/** Build Chart.js options for the EMG panel based on the current display mode. */
function buildEmgChartOptions() {
  const range = { min: -300, max: 300 };  // EMG amplitude range in µV
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
          callback: value => Number.isFinite(Number(value)) ? Number(value).toFixed(1) : value
        }
      },
      y: {
        display: emgChartMode !== 'stacked',
        title: { display: emgChartMode !== 'stacked', text: 'Amplitude (µV)' },
        min: range.min,
        max: range.max,
        grid: { color: 'rgba(255, 255, 255, 0.05)' }
      }
    },
    plugins: {
      legend: { display: emgChartMode === 'overlap', position: 'top' }
    },
    layout: emgChartMode === 'stacked'
      ? { padding: { left: 70, right: 10, top: 10, bottom: 10 } }
      : {}
  };
}

/**
 * Custom Chart.js plugin that draws channel labels on the left margin of the
 * EMG chart when in stacked mode — mirrors the EEG stackedLabelPlugin.
 */
const emgStackedLabelPlugin = {
  id: 'emgStackedLabels',
  afterDatasetsDraw(chartInstance) {
    if (emgChartMode !== 'stacked') return;
    const { ctx, chartArea, scales } = chartInstance;
    if (!chartArea || !scales?.y) return;
    ctx.save();
    ctx.fillStyle = '#80cbc4';  // teal tint to match EMG colour palette
    ctx.font = '12px Arial';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const enabled = getEmgEnabledChannels();
    for (let i = 0; i < enabled; i++) {
      const dataset = chartInstance.data.datasets[i];
      if (!dataset || dataset.hidden) continue;
      const offset = emgStackedOffsets[i] || 0;
      const y = scales.y.getPixelForValue(offset);
      ctx.fillText(`EMG ${i + 1}`, chartArea.left - 10, y);
    }
    ctx.restore();
  }
};

/** Switch between 'stacked' and 'overlap' display modes for the EMG chart. */
function setEmgChartMode(mode) {
  emgChartMode = mode === 'stacked' ? 'stacked' : 'overlap';
  initializeEmgChart();
}

// ─────────────────────────────────────────────────────────────────────────────
// EMG channel selector grid  (same pattern as the EEG grid)
//
// The EMG bracelet prototype is expected to have 8–16 channels.  Using the
// same compact grid as EEG keeps the two panels visually consistent and
// avoids any layout issues when the channel count changes at runtime.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Rebuild the EMG channel-visibility grid inside #emgChannelList.
 *
 * Called when dual-stream mode is enabled or when the EMG channel count
 * changes.  Uses the teal/green emgChannelColors palette so EMG cells are
 * visually distinct from the EEG grid above.
 */
function buildEmgChannelSelectors() {
  const container = document.getElementById('emgChannelList');
  if (!container) return;
  const enabled = getEmgEnabledChannels();

  // Grow the visibility array to cover all palette slots
  while (emgChannelVisibility.length < emgChannelColors.length) {
    emgChannelVisibility.push(true);
  }

  // ── Bulk-control toolbar (same three buttons as EEG grid) ─────────────────
  container.innerHTML = `
    <div class="ch-grid-controls">
      <button onclick="selectAllEmgChannels(true)"  title="Show all EMG channels">All</button>
      <button onclick="selectAllEmgChannels(false)" title="Hide all EMG channels">None</button>
      <button onclick="invertEmgChannelSelection()" title="Flip EMG channel visibility">Invert</button>
    </div>
    <div id="emgChGrid" class="ch-grid"></div>
  `;

  // ── Channel cells ──────────────────────────────────────────────────────────
  const grid = container.querySelector('#emgChGrid');
  for (let idx = 0; idx < enabled; idx++) {
    const cell = document.createElement('div');
    cell.className = 'ch-cell' + (emgChannelVisibility[idx] ? ' active' : '');
    cell.textContent = idx + 1;
    cell.title = `EMG Channel ${idx + 1} — click to toggle`;
    cell.style.borderColor = emgChannelColors[idx] || '#aaa';
    cell.dataset.index = idx;
    cell.onclick = () => toggleEmgChannel(idx, cell);
    grid.appendChild(cell);
  }
}

/**
 * Toggle a single EMG channel on or off.
 *
 * @param {number}  index  Zero-based channel index.
 * @param {Element} cell   Grid cell DOM element.
 */
function toggleEmgChannel(index, cell) {
  emgChannelVisibility[index] = !emgChannelVisibility[index];
  if (cell) cell.classList.toggle('active', emgChannelVisibility[index]);
  if (emgChart && emgChart.data.datasets[index]) {
    emgChart.data.datasets[index].hidden = !emgChannelVisibility[index];
    emgChart.update('none');
  }
}

/**
 * Show or hide all enabled EMG channels at once.
 *
 * @param {boolean} visible  true = show all, false = hide all.
 */
function selectAllEmgChannels(visible) {
  const enabled = getEmgEnabledChannels();
  for (let i = 0; i < enabled; i++) {
    emgChannelVisibility[i] = visible;
    if (emgChart && emgChart.data.datasets[i]) {
      emgChart.data.datasets[i].hidden = !visible;
    }
  }
  document.querySelectorAll('#emgChGrid .ch-cell').forEach((cell, i) => {
    cell.classList.toggle('active', emgChannelVisibility[i]);
  });
  if (emgChart) emgChart.update('none');
}

/**
 * Flip the visibility of every enabled EMG channel.
 * Useful for quickly isolating one channel (None → click one)
 * or comparing two groups of muscles.
 */
function invertEmgChannelSelection() {
  const enabled = getEmgEnabledChannels();
  for (let i = 0; i < enabled; i++) {
    emgChannelVisibility[i] = !emgChannelVisibility[i];
    if (emgChart && emgChart.data.datasets[i]) {
      emgChart.data.datasets[i].hidden = !emgChannelVisibility[i];
    }
  }
  document.querySelectorAll('#emgChGrid .ch-cell').forEach((cell, i) => {
    cell.classList.toggle('active', emgChannelVisibility[i]);
  });
  if (emgChart) emgChart.update('none');
}

// ─────────────────────────────────────────────────────────────────────────────
// EMG data handler — called on every 'emg_data' Socket.IO event
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Receive a chunk of EMG data from the server and append it to the rolling
 * waveform buffers, then re-render the EMG chart.
 *
 * Mirrors updateChart() for EEG but uses the separate emgWaveformBuffers and
 * the emgChart instance so the two panels never interfere.
 *
 * @param {Object} data  Socket.IO payload:
 *   { channels: number[][], sampling_rate: number, signal_type: 'emg', ... }
 */
function updateEmgChart(data) {
  if (!emgChart || !data.channels || !Array.isArray(data.channels)) return;

  const enabled = getEmgEnabledChannels();
  const samplingRate = data.sampling_rate || 250;

  // Resize buffer array if channel count changed at runtime
  while (emgWaveformBuffers.length < emgChannelColors.length) emgWaveformBuffers.push([]);

  // Build time-stamped sample arrays for each channel
  const processedChannels = [];
  data.channels.forEach((channelData, index) => {
    if (index >= enabled || !Array.isArray(channelData)) return;
    processedChannels[index] = channelData;
  });

  const chunkLength = Math.max(...processedChannels.map(ch => (Array.isArray(ch) ? ch.length : 0)), 0);
  if (!chunkLength || samplingRate <= 0) return;

  // Assign timestamps to each sample, continuing from the last known time
  const times = Array.from({ length: chunkLength }, (_, i) =>
    emgLastSampleTimestamp + (i + 1) / samplingRate
  );
  emgLastSampleTimestamp = times[times.length - 1];
  const cutoff = Math.max(0, emgLastSampleTimestamp - waveformWindowSec);

  // Append new samples and trim old ones outside the rolling window
  processedChannels.forEach((series, index) => {
    if (!Array.isArray(series) || index >= enabled) return;
    const buffer = emgWaveformBuffers[index];
    const count = Math.min(series.length, times.length);
    for (let i = 0; i < count; i++) {
      buffer.push({ x: times[i], y: series[i] });
    }
    while (buffer.length && buffer[0].x < cutoff) buffer.shift();
  });

  renderEmgWaveform();
  updateEmgMetrics(processedChannels.slice(0, enabled));
}

/**
 * Re-render the EMG chart from the current contents of emgWaveformBuffers.
 * Called after every incoming data chunk and whenever display settings change.
 */
function renderEmgWaveform() {
  if (!emgChart) return;
  const enabled = getEmgEnabledChannels();

  // Freeze-aware window calculation — mirrors the EEG panel logic exactly.
  // When frozen, the window is anchored at freezePositionSec so both charts
  // display the same time range and neither panel scrolls while paused.
  // In live mode we use the bufferStart trick so that after a seek (or on
  // fresh start) the signal fills left-to-right instead of appearing at the
  // right edge of an otherwise empty chart.
  let windowStart;
  if (viewMode === 'freeze') {
    windowStart = freezePositionSec;
  } else {
    let bufferStart = Infinity;
    for (let i = 0; i < enabled; i++) {
      const buf = emgWaveformBuffers[i];
      if (buf && buf.length > 0 && buf[0].x < bufferStart) bufferStart = buf[0].x;
    }
    windowStart = bufferStart < Infinity
      ? Math.max(bufferStart, emgLastSampleTimestamp - waveformWindowSec)
      : Math.max(0, emgLastSampleTimestamp - waveformWindowSec);
  }

  // Compute per-channel vertical offsets for stacked mode
  let offsets = [];
  if (emgChartMode === 'stacked') {
    const layout = computeEmgStackedLayout(enabled, windowStart);
    offsets = layout.offsets;
    emgStackedOffsets = offsets;
    emgChart.options.scales.y.display = false;
    emgChart.options.scales.y.min = layout.min;
    emgChart.options.scales.y.max = layout.max;
    emgChart.options.plugins.legend.display = false;
    emgChart.options.layout = { padding: { left: 70, right: 10, top: 10, bottom: 10 } };
  } else {
    emgStackedOffsets = [];
    emgChart.options.scales.y.display = true;
    emgChart.options.scales.y.title.text = 'Amplitude (µV)';
    emgChart.options.scales.y.min = -300;
    emgChart.options.scales.y.max = 300;
    emgChart.options.plugins.legend.display = true;
    emgChart.options.layout = {};
  }

  emgChart.data.datasets.forEach((dataset, idx) => {
    dataset.hidden = idx >= enabled ? true : !emgChannelVisibility[idx];
    const buffer = emgWaveformBuffers[idx] || [];
    const visiblePoints = buffer.filter(pt => pt.x >= windowStart);
    const decimated = decimatePoints(visiblePoints, MAX_RENDER_POINTS);
    dataset.data = emgChartMode === 'stacked'
      ? decimated.map(pt => ({ x: pt.x - windowStart, y: pt.y + (offsets[idx] || 0) }))
      : decimated.map(pt => ({ x: pt.x - windowStart, y: pt.y }));
    dataset.borderWidth = emgChartMode === 'stacked' ? 1.2 : 1.5;
  });

  emgChart.options.scales.x.min = 0;
  emgChart.options.scales.x.max = waveformWindowSec;
  emgChart.update('none');
}

/**
 * Compute vertical offsets for the stacked EMG layout.
 * Uses the same algorithm as computeStackedLayout for EEG.
 */
function computeEmgStackedLayout(enabledChannels, windowStart) {
  const defaultSpacing = 250;
  let maxAbs = 0;

  emgWaveformBuffers.slice(0, enabledChannels).forEach(buf => {
    buf.forEach(pt => {
      if (pt.x >= windowStart && Math.abs(pt.y) > maxAbs) maxAbs = Math.abs(pt.y);
    });
  });

  const spacing = Math.max(defaultSpacing, maxAbs * 2.5 || defaultSpacing);
  const mid = (enabledChannels - 1) / 2;
  const offsets = emgChannelColors.map((_, idx) => (mid - idx) * spacing);

  let minVal = Infinity, maxVal = -Infinity;
  emgWaveformBuffers.slice(0, enabledChannels).forEach((buf, idx) => {
    buf.forEach(pt => {
      if (pt.x < windowStart) return;
      const v = pt.y + offsets[idx];
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    });
  });

  if (!isFinite(minVal) || !isFinite(maxVal)) { minVal = -spacing; maxVal = spacing; }
  const pad = spacing * 0.6;
  return { offsets, min: minVal - pad, max: maxVal + pad };
}

/**
 * Update the live EMG metrics cards (RMS and MAV averages across channels).
 * These numbers give a quick indication of overall muscle activation level.
 *
 * @param {number[][]} channels  Array of per-channel sample arrays for the latest chunk.
 */
function updateEmgMetrics(channels) {
  if (!channels || !channels.length) return;

  let totalRms = 0, totalMav = 0, count = 0;
  channels.forEach(ch => {
    if (!Array.isArray(ch) || !ch.length) return;
    const rms = Math.sqrt(ch.reduce((s, v) => s + v * v, 0) / ch.length);
    const mav = ch.reduce((s, v) => s + Math.abs(v), 0) / ch.length;
    totalRms += rms;
    totalMav += mav;
    count++;
  });

  if (count) {
    setText('emgRms', `${(totalRms / count).toFixed(2)} µV`);
    setText('emgMav', `${(totalMav / count).toFixed(2)} µV`);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Socket.IO event handlers
// ─────────────────────────────────────────────────────────────────────────────
socket.on('connect', () => console.log('[client] Socket.IO connected'));
socket.on('disconnect', () => console.log('[client] Socket.IO disconnected'));

socket.on('analysis_stopped', () => {
  document.getElementById('startBtn').disabled = false;
  document.getElementById('stopBtn').disabled = true;
  // Re-enable the dual-mode toggle so the user can change settings
  const dualToggle = document.getElementById('dualModeToggle');
  if (dualToggle) dualToggle.disabled = false;
  // Hide the player bar — it only makes sense while a file is streaming
  playerTeardown();
});

// ── File-replay socket events ─────────────────────────────────────────────────

// Emitted once when the server starts replaying a file.
// Shows the player bar and stores file duration / sampling rate.
socket.on('playback_info', data => {
  playerSetup(data);
});

// Emitted every ~100 ms during file replay to advance the scrubber.
// Ignored while the user is dragging the scrubber thumb.
socket.on('playback_progress', data => {
  playerTick(data);
});

// Primary stream — drives the EEG chart panel
socket.on('eeg_data', data => {
  updateChart(data);
});

// Secondary stream — drives the EMG chart panel in dual-stream mode.
// The server only emits this event when dual_stream_mode is True.
socket.on('emg_data', data => {
  // Mirror the EEG Freeze behaviour — when the view is frozen, discard
  // incoming EMG chunks so both panels stop updating at the same moment.
  if (viewMode === 'freeze') return;
  updateEmgChart(data);
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

  // Compute the start of the visible window.
  //
  // Freeze mode: pinned at the timestamp when Freeze was clicked.
  //
  // Live mode:
  //   Normally windowStart = lastSampleTimestamp - 8 s (trailing edge).
  //   But after a seek (or at stream start) the buffer is almost empty, so
  //   the first new sample would appear at the RIGHT edge and fill leftward —
  //   the opposite of what feels natural.
  //
  //   Fix: find the oldest point currently in any buffer (bufferStart).
  //   windowStart = max(bufferStart, lastSampleTimestamp - 8).
  //   While the buffer spans < 8 s, bufferStart wins and data fills left→right.
  //   Once the buffer is full the two terms are equal and it scrolls normally.
  let windowStart;
  if (viewMode === 'freeze') {
    windowStart = freezePositionSec;
  } else {
    let bufferStart = Infinity;
    for (let i = 0; i < enabled; i++) {
      const buf = waveformBuffers[i];
      if (buf.length > 0 && buf[0].x < bufferStart) bufferStart = buf[0].x;
    }
    windowStart = bufferStart < Infinity
      ? Math.max(bufferStart, lastSampleTimestamp - waveformWindowSec)
      : Math.max(0, lastSampleTimestamp - waveformWindowSec);
  }
  currentWindowStartSec = windowStart;
  const layout = chartMode === 'stacked' ? computeStackedLayout(enabled, windowStart) : null;
  const offsets = chartMode === 'stacked' && layout ? layout.offsets : [];
  stackedOffsets = offsets;

  chart.data.datasets.forEach((dataset, idx) => {
    dataset.hidden = idx >= enabled ? true : !channelVisibility[idx];
    const buffer = waveformBuffers[idx] || [];
    const visiblePoints = buffer.filter(pt => pt.x >= windowStart);
    const decimatedPoints = decimatePoints(visiblePoints, MAX_RENDER_POINTS);
    const renderPoints =
      chartMode === 'stacked'
        ? decimatedPoints.map(pt => ({ x: pt.x - windowStart, y: pt.y + (offsets[idx] || 0) }))
        : decimatedPoints.map(pt => ({ x: pt.x - windowStart, y: pt.y }));
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

function decimatePoints(points, maxPoints) {
  if (!Array.isArray(points) || points.length <= maxPoints) {
    return points || [];
  }
  const step = Math.ceil(points.length / maxPoints);
  const reduced = [];
  for (let i = 0; i < points.length; i += step) {
    reduced.push(points[i]);
  }
  const lastPoint = points[points.length - 1];
  if (reduced[reduced.length - 1] !== lastPoint) {
    reduced.push(lastPoint);
  }
  return reduced;
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

  const now = Date.now();
  if (now - lastFeatureUpdateMs >= FEATURE_UPDATE_INTERVAL_MS) {
    lastFeatureUpdateMs = now;
    updateFeatureCharts(channels, samplingRate, avgPower);
  }
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
