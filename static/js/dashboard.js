// ══════════════════════════════════════════════════════════════════════════
//  FraudGuard AI — Dashboard Chart Visualizations
//  Loaded only on /dashboard page
// ══════════════════════════════════════════════════════════════════════════

// Chart.js global defaults for dark theme
Chart.defaults.color = '#a0aec0';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";

const CHART_COLORS = {
    fraud:    '#ff4757',
    low:      '#00d4aa',
    moderate: '#ffa502',
    high:     '#ff793f',
    critical: '#ff4757',
    primary:  '#6c63ff',
    accent:   '#00d4aa',
};

const TIER_COLORS = {
    LOW:      '#00d4aa',
    MODERATE: '#ffa502',
    HIGH:     '#ff793f',
    CRITICAL: '#ff4757',
};

// ── Active chart instances (for cleanup) ──────────────────────────────────
let fraudRateChart = null;
let tierChart      = null;
let hourlyChart    = null;
let categoryChart  = null;


// ── Main init ─────────────────────────────────────────────────────────────
async function initDashboard() {
    try {
        const res  = await fetch('/api/analytics');
        const data = await res.json();

        updateDashStats(data);
        renderFraudRateChart(data);
        renderTierChart(data);
        renderHourlyChart(data);
        renderCategoryChart(data);
    } catch (err) {
        console.error('Dashboard load failed:', err);
    }
}


// ── Stat Cards ─────────────────────────────────────────────────────────────
function updateDashStats(data) {
    const set = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };

    set('dash-total-val',    data.total_transactions.toLocaleString());
    set('dash-fraud-val',    data.fraud_count.toLocaleString());
    set('dash-rate-val',     data.fraud_rate_percent.toFixed(2) + '%');

    const critical = data.risk_tier_breakdown ? data.risk_tier_breakdown.CRITICAL : 0;
    set('dash-critical-val', critical.toLocaleString());
}


// ── Fraud Rate Over Time Chart ─────────────────────────────────────────────
function renderFraudRateChart(data) {
    const canvas = document.getElementById('fraudRateChart');
    const empty  = document.getElementById('chart-empty-rate');

    if (!data.fraud_over_time || data.fraud_over_time.length === 0) {
        if (canvas) canvas.style.display = 'none';
        if (empty)  empty.style.display  = 'flex';
        return;
    }

    if (canvas) canvas.style.display = 'block';
    if (empty)  empty.style.display  = 'none';

    const labels = data.fraud_over_time.map(d => `TX ${d.index}`);
    const rates  = data.fraud_over_time.map(d => d.fraud_rate);

    if (fraudRateChart) fraudRateChart.destroy();
    fraudRateChart = new Chart(canvas, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Fraud Rate (%)',
                data: rates,
                borderColor: CHART_COLORS.fraud,
                backgroundColor: 'rgba(255,71,87,0.08)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: rates.length <= 20 ? 4 : 1,
                pointHoverRadius: 6,
                pointBackgroundColor: CHART_COLORS.fraud,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15,15,25,0.95)',
                    borderColor: 'rgba(108,99,255,0.3)',
                    borderWidth: 1,
                    callbacks: {
                        label: ctx => ` Fraud Rate: ${ctx.parsed.y.toFixed(2)}%`,
                    },
                },
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { maxTicksLimit: 10 },
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { callback: v => v + '%' },
                    beginAtZero: true,
                    max: Math.max(100, Math.ceil(Math.max(...rates) * 1.2)),
                },
            },
        },
    });
}


// ── Risk Tier Donut Chart ──────────────────────────────────────────────────
function renderTierChart(data) {
    const canvas = document.getElementById('tierChart');
    const empty  = document.getElementById('chart-empty-tier');
    const tiers  = data.risk_tier_breakdown || {};
    const total  = Object.values(tiers).reduce((s, v) => s + v, 0);

    if (total === 0) {
        if (canvas) canvas.style.display = 'none';
        if (empty)  empty.style.display  = 'flex';
        return;
    }

    if (canvas) canvas.style.display = 'block';
    if (empty)  empty.style.display  = 'none';

    const labels = Object.keys(tiers);
    const values = Object.values(tiers);
    const colors = labels.map(l => TIER_COLORS[l] || '#888');

    if (tierChart) tierChart.destroy();
    tierChart = new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderColor: 'rgba(15,15,25,0.8)',
                borderWidth: 3,
                hoverOffset: 8,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '68%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 16,
                        boxWidth: 14,
                        font: { size: 12 },
                    },
                },
                tooltip: {
                    backgroundColor: 'rgba(15,15,25,0.95)',
                    borderColor: 'rgba(108,99,255,0.3)',
                    borderWidth: 1,
                    callbacks: {
                        label: ctx => {
                            const pct = ((ctx.parsed / total) * 100).toFixed(1);
                            return ` ${ctx.label}: ${ctx.parsed} (${pct}%)`;
                        },
                    },
                },
            },
        },
    });
}


// ── Hourly Pattern Bar Chart ───────────────────────────────────────────────
function renderHourlyChart(data) {
    const canvas  = document.getElementById('hourlyChart');
    const empty   = document.getElementById('chart-empty-hourly');
    const hourly  = data.hourly_pattern || {};
    const total   = Object.values(hourly).reduce((s, v) => s + v, 0);

    if (total === 0) {
        if (canvas) canvas.style.display = 'none';
        if (empty)  empty.style.display  = 'flex';
        return;
    }

    if (canvas) canvas.style.display = 'block';
    if (empty)  empty.style.display  = 'none';

    const labels = Array.from({ length: 24 }, (_, i) => `${i}:00`);
    const values = labels.map((_, i) => hourly[String(i)] || 0);

    // Colour bars: high-risk hours (0-4) red, others blue-purple
    const barColors = values.map((_, i) => i < 5
        ? 'rgba(255,71,87,0.75)'
        : 'rgba(108,99,255,0.65)'
    );

    if (hourlyChart) hourlyChart.destroy();
    hourlyChart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Transactions',
                data: values,
                backgroundColor: barColors,
                borderRadius: 4,
                borderSkipped: false,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15,15,25,0.95)',
                    borderColor: 'rgba(108,99,255,0.3)',
                    borderWidth: 1,
                    callbacks: {
                        title: ([ctx]) => `Hour: ${ctx.label}`,
                        label: ctx => ` Transactions: ${ctx.parsed.y}`,
                        afterLabel: ctx => ctx.dataIndex < 5 ? ' ⚠️ High-risk window' : '',
                    },
                },
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 10 }, maxRotation: 0 },
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    beginAtZero: true,
                    ticks: { stepSize: 1 },
                },
            },
        },
    });
}


// ── Category Fraud Rate Chart ──────────────────────────────────────────────
function renderCategoryChart(data) {
    const canvas   = document.getElementById('categoryChart');
    const empty    = document.getElementById('chart-empty-cat');
    const catStats = data.category_stats || {};
    const entries  = Object.entries(catStats).filter(([, v]) => v.total > 0);

    if (entries.length === 0) {
        if (canvas) canvas.style.display = 'none';
        if (empty)  empty.style.display  = 'flex';
        return;
    }

    if (canvas) canvas.style.display = 'block';
    if (empty)  empty.style.display  = 'none';

    const labels = entries.map(([cat]) => cat.charAt(0).toUpperCase() + cat.slice(1));
    const rates  = entries.map(([, v]) =>
        v.total > 0 ? parseFloat(((v.fraud / v.total) * 100).toFixed(1)) : 0
    );
    const counts = entries.map(([, v]) => v.total);

    const barColors = rates.map(r =>
        r > 60 ? 'rgba(255,71,87,0.8)'
        : r > 30 ? 'rgba(255,121,63,0.8)'
        : r > 10 ? 'rgba(255,165,2,0.8)'
        : 'rgba(0,212,170,0.7)'
    );

    if (categoryChart) categoryChart.destroy();
    categoryChart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Fraud Rate (%)',
                data: rates,
                backgroundColor: barColors,
                borderRadius: 4,
                borderSkipped: false,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15,15,25,0.95)',
                    borderColor: 'rgba(108,99,255,0.3)',
                    borderWidth: 1,
                    callbacks: {
                        label: (ctx) => {
                            const idx = ctx.dataIndex;
                            return [
                                ` Fraud Rate: ${ctx.parsed.x}%`,
                                ` Total Analyzed: ${counts[idx]}`,
                                ` Fraud Cases: ${entries[idx][1].fraud}`,
                            ];
                        },
                    },
                },
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { callback: v => v + '%' },
                    max: 100,
                    beginAtZero: true,
                },
                y: {
                    grid: { display: false },
                },
            },
        },
    });
}


// ── Boot ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', initDashboard);
