// Global Variables
let ctx, eegChart;
const socket = window.io(); // Declare the io variable

// Color Configuration for EEG Channels
const colors = {
    ref: 'red',
    biasout: 'black',
    ch1: 'yellow',
    ch2: 'orange',
    ch3: 'brown',
    ch4: 'green',
    ch5: 'purple',
    ch6: 'blue',
    ch7: 'grey',
    ch8: 'white'
};

// Initialize the Chart When DOM is Ready
document.addEventListener("DOMContentLoaded", () => {
  ctx = document.getElementById("eegChart").getContext("2d")
  updateSettings() // Initial settings load
  setupColorBoxListeners() // Enable color-box click events
})

// WebSocket Event Listener for Updating the Chart with Real-Time Data
socket.on("update_data", (data) => {
  if (!eegChart) return

  if (eegChart.data.labels.length > 100) {
    eegChart.data.labels.shift()
    eegChart.data.datasets.forEach((dataset) => dataset.data.shift())
  }
  eegChart.data.labels.push(Date.now())
  eegChart.data.datasets.forEach((dataset, index) => {
    if (data.raw && data.raw.length > index) {
      dataset.data.push({ x: Date.now(), y: data.raw[index] })
    }
  })
  eegChart.update()
})

// Listen for errors
socket.on("error", (data) => {
  alert("Error: " + data.message)
})

// Listen for analysis stopped
socket.on("analysis_stopped", () => {
  disableControls(false)
})

// Create and Update the Chart Based on Current Settings
function createChart() {
  if (eegChart) eegChart.destroy()

  const datasets = []
  const enabledChannels = Number.parseInt(document.getElementById("enabled_channels").value, 10)

  // Add Channels Based on Settings
  if (document.getElementById("ref_enabled").checked) {
    datasets.push({ label: "REF", data: [], borderColor: colors.ref, fill: false })
  }
  if (document.getElementById("biasout_enabled").checked) {
    datasets.push({ label: "BIASOUT", data: [], borderColor: colors.biasout, fill: false })
  }

  for (let i = 0; i < enabledChannels; i++) {
    datasets.push({ label: `Ch${i + 1}`, data: [], borderColor: colors[`ch${i + 1}`], fill: false })
  }

  // Create Chart.js Line Chart
  eegChart = new window.Chart(ctx, {
    type: "line",
    data: { labels: [], datasets: datasets },
    options: {
      animation: false,
      scales: {
        x: { type: "linear" },
        y: { type: "linear" },
      },
    },
  })
}

// Update Settings from the Front-End Controls
function updateSettings() {
    const enabledChannels = document.getElementById('enabled_channels').value;
    document.getElementById('enabledChannelsValue').innerText = enabledChannels;
    createChart();

    // Send Updated Settings to the Server
    fetch('/update-settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            baseline_correction_enabled: document.getElementById('baseline_correction_enabled').checked,
            enabled_channels: enabledChannels,
            ref_enabled: document.getElementById('ref_enabled').checked,
            biasout_enabled: document.getElementById('biasout_enabled').checked,
            bandpass_filter_enabled: document.getElementById('bandpass_filter_enabled').checked,
        })
    });
}

// Start the Real-Time EEG Analysis
function startAnalysis() {
  disableControls(true)
  fetch("/start-analysis", { method: "POST" })
    .then((response) => response.json())
    .then((data) => console.log(data))
    .catch((err) => {
      console.error("Failed to start analysis:", err)
      disableControls(false)
    })
}

// Stop the Real-Time EEG Analysis
function stopAnalysis() {
  disableControls(false)
  fetch("/stop-analysis", { method: "POST" })
    .then((response) => response.json())
    .then((data) => console.log(data))
    .catch((err) => console.error("Failed to stop analysis:", err))
}

// Start Calibration Process
function startCalibration() {
  disableControls(true)
  const btn = document.getElementById("calibrateBtn")
  btn.innerText = "Calibrating..."
  fetch("/calibrate", { method: "POST" })
    .then((response) => response.json())
    .then((data) => {
      alert("Calibration completed: " + JSON.stringify(data.values))
      btn.innerText = "Start Calibration"
      disableControls(false)
    })
    .catch((error) => {
      alert("Calibration failed: " + error)
      btn.innerText = "Start Calibration"
      disableControls(false)
    })
}

// Export EEG Data to CSV
function exportData() {
  const numRows = prompt("Enter the number of rows to export:", 5000)
  fetch(`/export-data?num_rows=${numRows}`)
    .then((response) => response.blob())
    .then((blob) => {
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.style.display = "none"
      a.href = url
      a.download = "eeg_data.csv"
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
    })
    .catch((err) => console.error("Error exporting data:", err))
}

// Helper to Enable or Disable Controls
function disableControls(disable) {
  document.getElementById("startBtn").disabled = disable
  document.getElementById("stopBtn").disabled = !disable
  document.getElementById("calibrateBtn").disabled = disable
  document.getElementById("exportBtn").disabled = disable
  document.getElementById("enabled_channels").disabled = disable
  document.getElementById("ref_enabled").disabled = disable
  document.getElementById("biasout_enabled").disabled = disable
  document.getElementById("bandpass_filter_enabled").disabled = disable
}

// Setup Color Box Listeners for Hiding/Showing Channels
function setupColorBoxListeners() {
  const colorBoxes = document.querySelectorAll(".color-box")
  colorBoxes.forEach((box) => {
    box.addEventListener("click", function () {
      const label = this.getAttribute("data-label")
      const dataset = eegChart.data.datasets.find((ds) => ds.label === label)
      if (dataset) {
        dataset.hidden = !dataset.hidden
        eegChart.update()
      }
    })
  })
}


