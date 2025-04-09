const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const status = document.getElementById("status");
const ctx = document.getElementById("aspectChart").getContext("2d");

let chart;

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];

  if (!file) {
    alert("Please select a .txt file first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  status.innerHTML = "⏳ Uploading and processing...";

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
            formatter: (value, ctx) => {
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

    status.innerHTML = "✅ Predictions displayed as percentage pie chart.";
  } catch (err) {
    console.error(err);
    status.innerHTML = "❌ Error processing file.";
  }
});
