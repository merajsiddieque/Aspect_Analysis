document.addEventListener("DOMContentLoaded", () => {
  const uploadBtn = document.getElementById("uploadBtn");
  const fileInput = document.getElementById("fileInput");
  const status = document.getElementById("status");
  const ctx = document.getElementById("aspectChart").getContext("2d");
  const micBtn = document.getElementById("micBtn");
  const speechText = document.getElementById("speechText");
  const uploadSpeechBtn = document.getElementById("uploadSpeechBtn");

  let chart;
  let recognition;
  let isListening = false;

  // ========== Upload File Button ==========
  uploadBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];

    if (!file) {
      alert("Please select a .txt file first.");
      return;
    }

    await uploadAndDisplay(file);
  });

  // ========== Upload Speech Text Button ==========
  uploadSpeechBtn.addEventListener("click", async () => {
    const text = speechText.value.trim();

    if (!text) {
      alert("Please speak something or type in the box.");
      return;
    }

    const blob = new Blob([text], { type: "text/plain" });
    const file = new File([blob], "speech.txt");

    await uploadAndDisplay(file);
  });

  // ========== Handle Speech Recognition ==========
  micBtn.addEventListener("click", () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert("Speech recognition not supported in this browser.");
      return;
    }

    if (!recognition) {
      recognition = new SpeechRecognition();
      recognition.lang = 'hi-IN';
      recognition.continuous = true;
      recognition.interimResults = false;

      recognition.onresult = (event) => {
        const transcript = event.results[event.results.length - 1][0].transcript.trim();
        if (transcript) {
          speechText.value += (speechText.value ? "\n" : "") + transcript;
        }
        status.textContent = "🎙️ Listening...";
      };

      recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        status.textContent = "❌ Speech recognition error.";
      };

      recognition.onend = () => {
        if (isListening) recognition.start();
      };
    }

    if (!isListening) {
      recognition.start();
      isListening = true;
      status.textContent = "🎙️ Listening... (click Upload when done)";
      micBtn.textContent = "🛑 Stop Listening";
    } else {
      recognition.stop();
      isListening = false;
      status.textContent = "🛑 Speech stopped. Click upload.";
      micBtn.textContent = "🎤 Start Speaking";
    }
  });

  // ========== Upload and Display Pie Chart ==========
  async function uploadAndDisplay(file) {
    const formData = new FormData();
    formData.append("file", file);

    status.textContent = "⏳ Uploading and processing...";

    try {
      const response = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Server error");

      const text = await response.text();
      const aspectCounts = {};

      text.split("\n").forEach((line) => {
        const aspect = line.trim();
        if (aspect) {
          aspectCounts[aspect] = (aspectCounts[aspect] || 0) + 1;
        }
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
              formatter: (value) => {
                const percentage = ((value / total) * 100).toFixed(1);
                return percentage + "%";
              },
              color: "#fff",
              font: {
                weight: "bold",
                size: 14
              }
            },
            legend: {
              position: 'right',
              labels: {
                font: {
                  size: 14
                }
              }
            }
          }
        },
        plugins: [ChartDataLabels]
      });

      status.textContent = "✅ Chart updated!";
    } catch (err) {
      console.error(err);
      status.textContent = "❌ Error processing file.";
    }
  }
});
