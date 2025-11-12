const socket = io();
let eegChart;
const MAX_POINTS = 2000;

document.addEventListener("DOMContentLoaded", () => {
  initChart();
});

function initChart() {
  const ctx = document.getElementById("eegChart");
  eegChart = new Chart(ctx, {
    type: "line",
    data: { labels: Array(MAX_POINTS).fill(""), datasets: [] },
    options: {
      animation: false,
      responsive: true,
      elements: { point: { radius: 0 } },
      maintainAspectRatio: false,
      scales: { y: { title: { display: true, text: "µV (approx)" } } }
    }
  });
}

function setButtons(running) {
  document.getElementById("startBtn").disabled = running;
  document.getElementById("stopBtn").disabled = !running;
}
function writeStatus(msg) { document.getElementById("status").innerText = msg; }

function startAnalysis() {
  socket.emit("start");
  setButtons(true);
  writeStatus("Streaming started…");
}
function stopAnalysis() {
  socket.emit("stop");
  setButtons(false);
  writeStatus("Streaming stopped.");
}

// make them callable from inline onclick
window.startAnalysis = startAnalysis;
window.stopAnalysis = stopAnalysis;

socket.on("eeg_chunk", (msg) => {
  const eeg = msg.eeg;
  const chCount = eeg.length;

  if (eegChart.data.datasets.length === 0) {
    const colors = ["#e6194B","#f58231","#ffe119","#3cb44b",
                    "#0082c8","#911eb4","#a9a9a9","#000000"];
    for (let i = 0; i < chCount; i++) {
      eegChart.data.datasets.push({
        label: `Ch ${i + 1}`,
        data: [],
        borderColor: colors[i % colors.length],
        borderWidth: 1,
        fill: false
      });
    }
  }

  for (let i = 0; i < chCount; i++) {
    const arr = eegChart.data.datasets[i].data;
    eeg[i].forEach(v => arr.push(v));
    while (arr.length > MAX_POINTS) arr.shift();
  }
  eegChart.update("none");
});
