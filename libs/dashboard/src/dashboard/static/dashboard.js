let forecastChartInstance = null;

async function initForecastChart(csvUrl) {
    if (!csvUrl) {
        const urlParams = new URLSearchParams(window.location.search);
        const appId = urlParams.get('app_id');
        csvUrl = appId ? `/data/forecast_data.csv?app_id=${appId}` : '/data/forecast_data.csv';
    }

    try {
        const response = await fetch(csvUrl);
        const csvText = await response.text();

        Papa.parse(csvText, {
            header: true,
            skipEmptyLines: true,
            complete: function(results) {
                const rows = results.data
                    .map(row => ({
                        label: row.day || row.date || row.timestamp || row.Date,
                        value: parseFloat(
                            row.predicted_mean ||
                            row.kpi ||
                            row.mean_players ||
                            row.y
                        )
                    }))
                    .filter(row => row.label && !Number.isNaN(row.value));

                const labels = rows.map(row => row.label);
                const playerCounts = rows.map(row => row.value);

                renderChart(labels, playerCounts);
            }
        });
    } catch (error) {
        console.error("Error loading CSV file:", error);
    }
}

function renderChart(labels, dataPoints) {
    const ctx = document.getElementById('forecastChart').getContext('2d');

    if (forecastChartInstance) {
        forecastChartInstance.destroy();
    }

    forecastChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Forecasted Players',
                data: dataPoints,
                borderColor: '#67c1f5',
                backgroundColor: 'rgba(103, 193, 245, 0.15)',
                borderWidth: 2,
                fill: true,
                tension: 0.35,
                pointRadius: 2.5,
                pointHoverRadius: 5,
                pointBackgroundColor: '#1a9fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: '#171d25',
                    titleColor: '#ffffff',
                    bodyColor: '#67c1f5',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        font: { size: 10 },
                        autoSkip: false,
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        font: { size: 10 }
                    }
                }
            }
        }
    });
}

function toggleDropdown() {
    const menu = document.getElementById('games-dropdown-menu');
    if (menu) {
        menu.classList.toggle('hidden');
    }
}

function openTrackGameModal() {
    const menu = document.getElementById('games-dropdown-menu');
    if (menu) menu.classList.add('hidden');

    const modal = document.getElementById('track-game-modal');
    if (modal) {
        modal.classList.remove('hidden');
        const input = document.getElementById('track-appid-input');
        if (input) input.focus();
    }
}

function closeTrackGameModal() {
    const modal = document.getElementById('track-game-modal');
    if (modal) modal.classList.add('hidden');
}

function handleTrackGameSubmit(event) {
    event.preventDefault();

    const input = document.getElementById('track-appid-input');
    if (input && input.value.trim()) {
        const appId = input.value.trim();
        window.location.href = '/?app_id=' + encodeURIComponent(appId);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    initForecastChart();
});

document.addEventListener('click', (e) => {
    const container = document.getElementById('game-selector-container');
    const menu = document.getElementById('games-dropdown-menu');

    if (container && menu && !container.contains(e.target)) {
        menu.classList.add('hidden');
    }
});