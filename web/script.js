const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const recordBtn = document.getElementById("recordBtn");
const sendAudioBtn = document.getElementById("sendAudioBtn");
const status = document.getElementById("status");
const ctx = document.getElementById("aspectChart").getContext("2d");

let chart;
let mediaRecorder;
let recordedChunks = [];
let isRecording = false;

// Initialize pie chart
function initChart() {
    chart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'right' },
                datalabels: {
                    formatter: (value, ctx) => {
                        const sum = ctx.dataset.data.reduce((a, b) => a + b, 0);
                        return ((value * 100) / sum).toFixed(1) + "%";
                    },
                    color: '#fff',
                    font: { weight: 'bold', size: 14 }
                }
            }
        },
        plugins: [ChartDataLabels]
    });
}

// Update chart with aspect prediction data
function updateChart(text) {
    const aspectCounts = {};
    text.split("\n").forEach(line => {
        const aspect = line.trim();
        if (aspect) {
            aspectCounts[aspect] = (aspectCounts[aspect] || 0) + 1;
        }
    });

    chart.data.labels = Object.keys(aspectCounts);
    chart.data.datasets[0].data = Object.values(aspectCounts);
    chart.update();
}

// Handle file upload
uploadBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) {
        status.textContent = "⚠️ Please select a file first";
        return;
    }

    status.textContent = "⏳ Processing file...";

    try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("Upload failed");

        const text = await response.text();
        updateChart(text);
        status.textContent = "✅ File processed successfully!";
    } catch (error) {
        console.error("Error:", error);
        status.textContent = "❌ Error processing file";
    }
});

// Toggle recording
async function toggleRecording() {
    if (!isRecording) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            recordedChunks = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) recordedChunks.push(e.data);
            };

            mediaRecorder.onstop = async () => {
                stream.getTracks().forEach(track => track.stop());
                sendAudioBtn.disabled = false;

                const blob = new Blob(recordedChunks, { type: "audio/webm" });
                await handleAudioUpload(blob);
            };

            mediaRecorder.start(100);
            isRecording = true;
            recordBtn.textContent = "⏹️ Stop Recording";
            recordBtn.classList.add("recording");
            sendAudioBtn.disabled = true;
            status.textContent = "🎙️ Recording...";
        } catch (error) {
            console.error("Microphone error:", error);
            status.textContent = "❌ Microphone access denied";
        }
    } else {
        mediaRecorder.stop();
        isRecording = false;
        recordBtn.textContent = "🎙️ Start Recording";
        recordBtn.classList.remove("recording");
        status.textContent = "🎙️ Recording stopped and uploading...";
    }
}

// Handle audio upload, conversion, transcription and classification
async function handleAudioUpload(blob) {
    status.textContent = "🔊 Processing audio...";
    
    try {
        // Step 1: Upload the audio file
        const formData = new FormData();
        formData.append("audio", blob, "input_audio.webm");

        const uploadResponse = await fetch("/upload-audio", {
            method: "POST",
            body: formData
        });

        if (!uploadResponse.ok) {
            throw new Error("Audio upload failed");
        }

        // Step 2: Process ASR (this now includes conversion and transcription)
        status.textContent = "💬 Transcribing audio...";
        const asrResponse = await fetch("/upload-asr", {
            method: "POST"
        });

        if (!asrResponse.ok) {
            throw new Error("ASR processing failed");
        }

        // Step 3: Get the classification results
        status.textContent = "🧠 Analyzing sentiment...";
        const text = await asrResponse.text();
        updateChart(text);
        status.textContent = "✅ Audio processed and analyzed successfully!";
    } catch (error) {
        console.error("Audio processing error:", error);
        status.textContent = "❌ Error processing audio";
    } finally {
        sendAudioBtn.disabled = false;
    }
}

// Remove the separate sendAudioBtn click handler since we're handling everything in one flow

// Start chart on load
initChart();
recordBtn.addEventListener("click", toggleRecording);