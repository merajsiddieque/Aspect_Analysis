const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const recordBtn = document.getElementById("recordBtn");
const sendAudioBtn = document.getElementById("sendAudioBtn");
const status = document.getElementById("status");
const ctx = document.getElementById("aspectChart").getContext("2d");

let chart;
let mediaRecorder;
let recordedChunks = [];

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return alert("Please select a file.");
  await uploadAndDisplay(file);
});

recordBtn.addEventListener("click", async () => {
  if (!navigator.mediaDevices.getUserMedia) {
    alert("Your browser does not support audio recording.");
    return;
  }

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);

  recordedChunks = [];
  mediaRecorder.ondataavailable = (e) => recordedChunks.push(e.data);
  mediaRecorder.onstop = () => {
    sendAudioBtn.disabled = false;
    status.textContent = "🎙️ Recording stopped. Ready to upload.";
  };

  mediaRecorder.start();
  status.textContent = "🎙️ Recording... click again to stop.";
  recordBtn.textContent = "⏹️ Stop Recording";

  recordBtn.onclick = () => {
    mediaRecorder.stop();
    recordBtn.textContent = "🎙️ Start Recording";
    recordBtn.onclick = recordAudio; // reset
  };
});

sendAudioBtn.addEventListener("click", async () => {
  const audioBlob = new Blob(recordedChunks, { type: "audio/webm" });
  const formData = new FormData();
  formData.append("audio", audioBlob, "speech.webm");

  status.textContent = "🔊 Uploading audio for transcription...";

  const res = await fetch("http://127.0.0.1:5000/upload_audio", {
    method: "POST",
    body: formData
  });

  const text = await res.text();
  displayChart(text);
});

async function uploadAndDisplay(file) {
  const formData = new FormData();
  formData.append("file", file);
  status.textContent = "⏳ Uploading and processing...";

  const response = await fetch("http://127.0.0.1:5000/upload", {
    method: "POST",
    body: formData
  });

  const text = await response.text();
  displayChart(text);
}

function displayChart(text) {
  const aspectCounts = {};
  text.split("\n").forEach((line) => {
    const aspect = line.trim();
    if (aspect) aspectCounts[aspect] = (aspectCounts[aspect] || 0) + 1;
  });

  const labels = Object.keys(aspectCounts);
  const values = Object.values(aspectCounts);
  const total = values.reduce((sum, val) => sum + val, 0);

  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: "pie",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: [
          "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF", "#FF9F40"
        ],
      }]
    },
    options: {
      plugins: {
        datalabels: {
          formatter: (value) => ((value / total) * 100).toFixed(1) + "%",
          color: "#fff",
          font: { weight: "bold", size: 14 }
        },
        legend: {
          position: 'right',
          labels: { font: { size: 14 } }
        }
      }
    },
    plugins: [ChartDataLabels]
  });

  status.textContent = "✅ Chart updated!";
}
