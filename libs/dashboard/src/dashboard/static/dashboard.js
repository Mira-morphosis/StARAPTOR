let forecastChartInstance = null;

/**
 * Carica e analizza il file CSV per il grafico di previsione
 */
async function initForecastChart(csvUrl = '/data/forecast_data.csv') {
    try {
        const response = await fetch(csvUrl);
        const csvText = await response.text();

        Papa.parse(csvText, {
            header: true,
            skipEmptyLines: true,
            complete: function(results) {
                const data = results.data;
                const labels = data.map(row => row.day || row.date || row.Date);
                const playerCounts = data.map(row => parseFloat(row.predicted_mean || row.mean_players || row.y));

                renderChart(labels, playerCounts);
            }
        });
    } catch (error) {
        console.error("Errore nel caricamento del file CSV:", error);
    }
}

/**
 * Inizializza / Aggiorna Chart.js
 */
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
                                      ticks: { color: 'rgba(255, 255, 255, 0.7)', font: { size: 10 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                                      ticks: { color: 'rgba(255, 255, 255, 0.7)', font: { size: 10 } }
                }
            }
        }
    });
}

/**
 * Toggle Dropdown Selezione Gioco
 */
function toggleDropdown() {
    const menu = document.getElementById('games-dropdown-menu');
    if (menu) {
        menu.classList.toggle('hidden');
    }
}

// Event Listeners Globali
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
