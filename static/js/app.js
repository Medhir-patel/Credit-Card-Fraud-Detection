// ══════════════════════════════════════════════════════════
//  FraudGuard AI — Dashboard JavaScript
// ══════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    if (typeof FEATURE_NAMES !== "undefined" && FEATURE_NAMES.length > 0) {
        buildFeatureInputs();
    }
});

/** Build dynamic feature input fields */
function buildFeatureInputs() {
    const grid = document.getElementById("features-grid");
    if (!grid) return;

    grid.innerHTML = "";
    FEATURE_NAMES.forEach((name, i) => {
        const div = document.createElement("div");
        div.className = "feature-input";
        div.innerHTML = `
            <label for="f_${i}">${name}</label>
            <input type="number" step="any" id="f_${i}" placeholder="0.0" />
        `;
        grid.appendChild(div);
    });
}

/** Load sample transaction data */
async function loadSample(type) {
    try {
        const res = await fetch("/api/sample");
        const data = await res.json();

        const values = data[type];
        if (!values) return;

        FEATURE_NAMES.forEach((_, i) => {
            const input = document.getElementById(`f_${i}`);
            if (input && values[i] !== undefined) {
                input.value = parseFloat(values[i]).toFixed(4);
            }
        });

        // Visual feedback
        const btn = type === "legitimate"
            ? document.getElementById("btn-load-legit")
            : document.getElementById("btn-load-fraud");
        if (btn) {
            btn.style.background = "rgba(108,99,255,0.15)";
            setTimeout(() => { btn.style.background = ""; }, 400);
        }
    } catch (err) {
        console.error("Failed to load sample:", err);
    }
}

/** Send prediction request */
async function predict() {
    const features = [];
    let empty = true;

    FEATURE_NAMES.forEach((_, i) => {
        const input = document.getElementById(`f_${i}`);
        const val = parseFloat(input ? input.value : 0) || 0;
        features.push(val);
        if (val !== 0) empty = false;
    });

    if (empty) {
        alert("Please load a sample or enter feature values first.");
        return;
    }

    // Show loading state
    const btnText = document.querySelector(".btn-text");
    const btnLoader = document.querySelector(".btn-loader");
    if (btnText) btnText.style.display = "none";
    if (btnLoader) btnLoader.style.display = "inline";

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features }),
        });

        const data = await res.json();

        if (data.error) {
            alert("Error: " + data.error);
            return;
        }

        showResult(data);
    } catch (err) {
        alert("Request failed: " + err.message);
    } finally {
        if (btnText) btnText.style.display = "inline";
        if (btnLoader) btnLoader.style.display = "none";
    }
}

/** Display prediction result */
function showResult(data) {
    const placeholder = document.getElementById("result-placeholder");
    const content = document.getElementById("result-content");
    const badge = document.getElementById("result-badge");
    const probBar = document.getElementById("prob-bar");
    const probValue = document.getElementById("prob-value");
    const details = document.getElementById("result-details");

    if (placeholder) placeholder.style.display = "none";
    if (content) content.style.display = "block";

    // Badge
    if (badge) {
        badge.textContent = data.label;
        badge.className = "result-badge " + (data.is_fraud ? "fraud" : "legit");
    }

    // Probability bar
    const pct = (data.probability * 100).toFixed(2);
    if (probBar) {
        // Small delay for animation
        setTimeout(() => { probBar.style.width = pct + "%"; }, 100);
    }
    if (probValue) {
        probValue.textContent = pct + "% fraud probability";
    }

    // Details
    if (details) {
        let riskHtml = "";
        if (data.risk) {
            riskHtml = `
                <div class="risk-decision-box">
                    <div class="risk-header">
                        <span class="risk-badge-tier ${data.risk.badge_class}">${data.risk.label} (${data.risk.score} / 100)</span>
                        <span style="font-size:12px; font-weight:700; color: ${data.risk.color}">DECISION: ${data.risk.decision}</span>
                    </div>
                    <div class="risk-action-title">🛡️ Recommended Action: ${data.risk.action}</div>
                    <div class="risk-action-desc">${data.risk.action_detail}</div>
                </div>
            `;
        }

        let driversHtml = "";
        if (data.risk_drivers && data.risk_drivers.length > 0) {
            const chips = data.risk_drivers.map(d => `
                <span class="driver-chip" title="Anomaly Magnitude: ${d.magnitude} (${d.direction})">
                    <strong>${d.feature}</strong>: ${d.raw_scaled > 0 ? '+' : ''}${d.raw_scaled}
                    <span class="impact-tag impact-${d.impact}">${d.impact}</span>
                </span>
            `).join("");

            driversHtml = `
                <div class="risk-drivers-section">
                    <div class="risk-drivers-title">⚡ Top Anomaly Drivers (Feature Deviations)</div>
                    <div class="risk-driver-chips">${chips}</div>
                </div>
            `;
        }

        details.innerHTML = `
            <p><strong>Prediction:</strong> ${data.is_fraud ? "Fraudulent" : "Legitimate"}</p>
            <p><strong>Fraud Probability:</strong> ${(data.probability * 100).toFixed(4)}%</p>
            <p><strong>Confidence:</strong> ${data.confidence}%</p>
            ${riskHtml}
            ${driversHtml}
        `;
    }

    // Render Visual Charts
    const riskColor = data.risk ? data.risk.color : (data.is_fraud ? "#ff4757" : "#00d4aa");
    renderRiskGaugeChart(data.probability, riskColor);
    if (data.risk_drivers) {
        renderShapBarChart(data.risk_drivers);
    }

    // Trigger Toast Notification
    if (data.is_fraud || (data.risk && (data.risk.tier === "CRITICAL" || data.risk.tier === "HIGH"))) {
        showToast("High Fraud Risk Detected!", `Transaction scored ${(data.probability * 100).toFixed(1)}% fraud probability. Action: ${data.risk ? data.risk.action : 'Block'}`, "danger");
    } else {
        showToast("Transaction Verified", `Legitimate score: ${(data.probability * 100).toFixed(1)}% fraud probability.`, "success");
    }

    // Scroll to result on mobile
    if (window.innerWidth < 900 && content) {
        content.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

/* ══════════════════════════════════════════════════════════
   Chart.js Engine & Controller
   ══════════════════════════════════════════════════════════ */
let chartGaugeInstance = null;
let chartShapInstance = null;
let chartSimTimelineInstance = null;
let simTimelineData = {
    labels: [],
    datasets: [{
        label: 'Fraud Prob %',
        data: [],
        borderColor: '#6c63ff',
        backgroundColor: 'rgba(108,99,255,0.15)',
        fill: true,
        tension: 0.3
    }]
};

function renderRiskGaugeChart(probVal, riskColor) {
    const canvas = document.getElementById("chart-risk-gauge");
    if (!canvas || typeof Chart === "undefined") return;

    const probPct = parseFloat((probVal * 100).toFixed(2));
    const remainPct = Math.max(0.01, 100 - probPct);

    if (chartGaugeInstance) {
        chartGaugeInstance.destroy();
    }

    chartGaugeInstance = new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels: ['Risk Score', 'Safe Margin'],
            datasets: [{
                data: [probPct, remainPct],
                backgroundColor: [riskColor || '#00d4aa', 'rgba(255,255,255,0.06)'],
                borderWidth: 0
            }]
        },
        options: {
            rotation: 270,
            circumference: 180,
            cutout: '75%',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: true }
            }
        }
    });
}

function renderShapBarChart(riskDrivers) {
    const canvas = document.getElementById("chart-shap-bar");
    if (!canvas || typeof Chart === "undefined" || !riskDrivers || riskDrivers.length === 0) return;

    const labels = riskDrivers.map(d => d.feature);
    const dataValues = riskDrivers.map(d => d.shap_value !== undefined ? d.shap_value : (d.direction === 'elevated' ? d.magnitude : -d.magnitude));
    const bgColors = dataValues.map(v => v >= 0 ? '#ff4757' : '#00d4aa');

    if (chartShapInstance) {
        chartShapInstance.destroy();
    }

    chartShapInstance = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: dataValues,
                backgroundColor: bgColors,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#8892b0', font: { size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#e8eaf6', font: { weight: 'bold', size: 11 } }
                }
            }
        }
    });
}

function updateSimTimelineChart(timestamp, probVal, isFraud) {
    const canvas = document.getElementById("chart-sim-timeline");
    if (!canvas || typeof Chart === "undefined") return;

    const probPct = parseFloat((probVal * 100).toFixed(2));

    if (!chartSimTimelineInstance) {
        chartSimTimelineInstance = new Chart(canvas, {
            type: 'line',
            data: simTimelineData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false }, ticks: { color: '#8892b0', font: { size: 9 } } },
                    y: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8892b0', font: { size: 9 } } }
                }
            }
        });
    }

    simTimelineData.labels.push(timestamp);
    simTimelineData.datasets[0].data.push(probPct);

    if (simTimelineData.labels.length > 20) {
        simTimelineData.labels.shift();
        simTimelineData.datasets[0].data.shift();
    }

    chartSimTimelineInstance.update();
}

/** Mode switching */
function switchTab(mode) {
    const singleTab = document.getElementById("tab-single");
    const batchTab = document.getElementById("tab-batch");
    const simTab = document.getElementById("tab-sim");
    const predictCard = document.getElementById("predict-card");
    const batchCard = document.getElementById("batch-card");
    const simCard = document.getElementById("sim-card");
    const singleResult = document.getElementById("result-content");
    const batchResult = document.getElementById("batch-result-content");
    const placeholder = document.getElementById("result-placeholder");

    if (singleTab) singleTab.classList.remove("active");
    if (batchTab) batchTab.classList.remove("active");
    if (simTab) simTab.classList.remove("active");
    if (predictCard) predictCard.style.display = "none";
    if (batchCard) batchCard.style.display = "none";
    if (simCard) simCard.style.display = "none";

    if (mode === "single") {
        if (singleTab) singleTab.classList.add("active");
        if (predictCard) predictCard.style.display = "block";
        if (batchResult) batchResult.style.display = "none";
        if (singleResult && singleResult.innerHTML.trim() !== "") {
            singleResult.style.display = "block";
            if (placeholder) placeholder.style.display = "none";
        } else {
            if (placeholder) placeholder.style.display = "block";
        }
    } else if (mode === "batch") {
        if (batchTab) batchTab.classList.add("active");
        if (batchCard) batchCard.style.display = "block";
        if (singleResult) singleResult.style.display = "none";
        if (batchResult && batchResult.innerHTML.trim() !== "" && document.getElementById("batch-table-body").children.length > 0) {
            batchResult.style.display = "block";
            if (placeholder) placeholder.style.display = "none";
        } else {
            if (placeholder) placeholder.style.display = "block";
        }
    } else if (mode === "sim") {
        if (simTab) simTab.classList.add("active");
        if (simCard) simCard.style.display = "block";
        if (singleResult) singleResult.style.display = "none";
        if (batchResult) batchResult.style.display = "none";
        if (placeholder) placeholder.style.display = "none";
    }
}

/* ══════════════════════════════════════════════════════════
   Real-Time Live Simulator Engine
   ══════════════════════════════════════════════════════════ */
let simInterval = null;
let simStats = { total: 0, approved: 0, fraud: 0 };

function startSim() {
    if (simInterval) return;
    const btnStart = document.getElementById("btn-sim-start");
    const btnPause = document.getElementById("btn-sim-pause");
    const speedSelect = document.getElementById("sim-speed");

    if (btnStart) btnStart.disabled = true;
    if (btnPause) btnPause.disabled = false;

    const intervalMs = parseInt(speedSelect ? speedSelect.value : 2000, 10);
    fetchSimStep();
    simInterval = setInterval(fetchSimStep, intervalMs);
}

function pauseSim() {
    if (simInterval) {
        clearInterval(simInterval);
        simInterval = null;
    }
    const btnStart = document.getElementById("btn-sim-start");
    const btnPause = document.getElementById("btn-sim-pause");
    if (btnStart) btnStart.disabled = false;
    if (btnPause) btnPause.disabled = true;
}

async function fetchSimStep() {
    try {
        const res = await fetch("/api/simulate", { method: "POST" });
        const tx = await res.json();
        if (tx.error) return;

        simStats.total++;
        if (tx.is_fraud) {
            simStats.fraud++;
        } else {
            simStats.approved++;
        }

        // Update Counter DOM
        const elTotal = document.getElementById("sim-counter-total");
        const elApp = document.getElementById("sim-counter-approved");
        const elFraud = document.getElementById("sim-counter-fraud");
        const elRate = document.getElementById("sim-counter-rate");

        if (elTotal) elTotal.innerText = simStats.total;
        if (elApp) elApp.innerText = simStats.approved;
        if (elFraud) elFraud.innerText = simStats.fraud;
        if (elRate) elRate.innerText = `${((simStats.fraud / simStats.total) * 100).toFixed(1)}%`;

        // Update Timeline Chart
        updateSimTimelineChart(tx.timestamp, tx.probability, tx.is_fraud);

        // Render live card in feed
        // Trigger Toast for High/Critical Risk in Stream
        if (tx.is_fraud || (tx.risk && (tx.risk.tier === "CRITICAL" || tx.risk.tier === "HIGH"))) {
            showToast(`🚨 Stream Alert: ${tx.id}`, `Amount: $${tx.raw_amount_usd} | Risk: ${(tx.probability * 100).toFixed(1)}% (${tx.risk ? tx.risk.decision : 'DECLINE'})`, "danger", 3500);
        }

        const feed = document.getElementById("sim-feed");
        if (!feed) return;

        if (simStats.total === 1) {
            feed.innerHTML = "";
        }

        const card = document.createElement("div");
        const isFraud = tx.is_fraud;
        const tierColor = tx.risk ? tx.risk.color : (isFraud ? "#ff4757" : "#00d4aa");
        const shapDrivers = (tx.risk_drivers || []).map(d => `${d.feature} (${d.direction.toLowerCase()})`).join(", ");

        card.style.cssText = `
            background: rgba(255, 255, 255, 0.03);
            border-left: 4px solid ${tierColor};
            border-radius: 8px;
            padding: 12px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            animation: fadeIn 0.3s ease-out;
            cursor: pointer;
        `;
        card.onclick = () => openDetailDrawer(tx);

        card.innerHTML = `
            <div>
                <div style="font-weight: 700; color: #fff;">
                    ${tx.id} <span style="font-size:0.8rem; color:var(--text-secondary); margin-left:8px;">${tx.timestamp}</span>
                </div>
                <div style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;">
                    Amount: <strong>$${tx.raw_amount_usd}</strong> | Top Drivers: ${shapDrivers || 'N/A'}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-weight: 800; color: ${tierColor};">${(tx.probability * 100).toFixed(2)}%</div>
                <div style="font-size: 0.75rem; font-weight: 700; color: ${tierColor}; text-transform: uppercase;">${tx.risk ? tx.risk.decision : (isFraud ? 'DECLINE' : 'APPROVE')}</div>
            </div>
        `;

        feed.insertBefore(card, feed.firstChild);
        if (feed.children.length > 20) {
            feed.removeChild(feed.lastChild);
        }
    } catch (err) {
        console.error("Simulation error:", err);
    }
}

/** Load a 10-transaction mock sample batch */
async function loadBatchSample() {
    try {
        const res = await fetch("/api/sample");
        const data = await res.json();
        const legit = data.legitimate;
        const fraud = data.fraud;

        const sampleBatch = [
            { id: "TX-9001 (Legit)", features: legit },
            { id: "TX-9002 (Fraud)", features: fraud },
            { id: "TX-9003 (Legit)", features: legit.map((v, i) => i === 28 ? v + 0.1 : v) },
            { id: "TX-9004 (Fraud)", features: fraud.map((v, i) => i === 28 ? v - 0.2 : v) },
            { id: "TX-9005 (Legit)", features: legit.map((v, i) => i === 29 ? (v + 3) % 24 : v) },
            { id: "TX-9006 (Legit)", features: legit.map((v, i) => i === 0 ? v + 0.05 : v) },
            { id: "TX-9007 (Fraud)", features: fraud.map((v, i) => i === 1 ? v - 0.1 : v) },
            { id: "TX-9008 (Legit)", features: legit.map((v, i) => i === 2 ? v - 0.05 : v) },
            { id: "TX-9009 (Legit)", features: legit.map((v, i) => i === 28 ? v + 0.3 : v) },
            { id: "TX-9010 (Fraud)", features: fraud.map((v, i) => i === 29 ? (v + 1) % 24 : v) }
        ];

        document.getElementById("batch-input-area").value = JSON.stringify(sampleBatch, null, 2);
    } catch (err) {
        alert("Failed to generate batch sample: " + err.message);
    }
}

/** Handle CSV / JSON File Upload */
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        const content = e.target.result;
        if (file.name.endsWith(".json")) {
            try {
                const parsed = JSON.parse(content);
                document.getElementById("batch-input-area").value = JSON.stringify(parsed, null, 2);
            } catch (err) {
                alert("Invalid JSON file: " + err.message);
            }
        } else if (file.name.endsWith(".csv")) {
            // Parse CSV lines
            const lines = content.trim().split("\n");
            const batch = [];
            const hasHeader = isNaN(parseFloat(lines[0].split(",")[0]));
            const startIdx = hasHeader ? 1 : 0;

            for (let i = startIdx; i < lines.length; i++) {
                const row = lines[i].trim();
                if (!row) continue;
                const cols = row.split(",").map(val => parseFloat(val.trim()));
                if (cols.some(isNaN)) continue;
                batch.push({
                    id: `CSV-ROW-${i + 1}`,
                    features: cols.slice(0, FEATURE_NAMES.length)
                });
            }
            document.getElementById("batch-input-area").value = JSON.stringify(batch, null, 2);
        }
    };
    reader.readAsText(file);
}

/** Execute Batch Prediction */
async function predictBatch() {
    const rawText = document.getElementById("batch-input-area").value.trim();
    if (!rawText) {
        alert("Please load a batch sample or paste JSON / CSV data first.");
        return;
    }

    let parsedTransactions;
    try {
        parsedTransactions = JSON.parse(rawText);
        if (!Array.isArray(parsedTransactions)) {
            throw new Error("Payload must be a JSON array of transactions.");
        }
    } catch (err) {
        alert("JSON Parsing error: " + err.message);
        return;
    }

    const btnText = document.querySelector(".btn-batch-text");
    const btnLoader = document.querySelector(".btn-batch-loader");
    if (btnText) btnText.style.display = "none";
    if (btnLoader) btnLoader.style.display = "inline";

    try {
        const res = await fetch("/predict/batch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ transactions: parsedTransactions })
        });

        const data = await res.json();
        if (data.error) {
            alert("Batch Error: " + data.error);
            return;
        }

        showBatchResult(data);
    } catch (err) {
        alert("Batch scan failed: " + err.message);
    } finally {
        if (btnText) btnText.style.display = "inline";
        if (btnLoader) btnLoader.style.display = "none";
    }
}

/** Display Batch Scan Results */
function showBatchResult(data) {
    const placeholder = document.getElementById("result-placeholder");
    const singleContent = document.getElementById("result-content");
    const batchContent = document.getElementById("batch-result-content");
    const summaryCards = document.getElementById("batch-summary-cards");
    const tableBody = document.getElementById("batch-table-body");

    if (placeholder) placeholder.style.display = "none";
    if (singleContent) singleContent.style.display = "none";
    if (batchContent) batchContent.style.display = "block";

    // Summary Cards
    const summary = data.summary;
    summaryCards.innerHTML = `
        <div class="batch-mini-stat">
            <div class="batch-mini-val">${summary.total_analyzed}</div>
            <div class="batch-mini-lbl">Scanned</div>
        </div>
        <div class="batch-mini-stat" style="border-color: rgba(255, 71, 87, 0.4);">
            <div class="batch-mini-val" style="color: var(--danger);">${summary.fraud_count}</div>
            <div class="batch-mini-lbl">Frauds Flagged</div>
        </div>
        <div class="batch-mini-stat" style="border-color: rgba(0, 212, 170, 0.4);">
            <div class="batch-mini-val" style="color: var(--success);">${summary.fraud_rate_percent}%</div>
            <div class="batch-mini-lbl">Fraud Rate</div>
        </div>
    `;

    // Table rows
    tableBody.innerHTML = "";
    data.results.forEach((row) => {
        const tr = document.createElement("tr");
        tr.className = row.is_fraud ? "batch-row-fraud" : "batch-row-legit";
        const riskTier = row.risk ? row.risk.tier : (row.is_fraud ? "CRITICAL" : "LOW");
        const badgeClass = row.risk ? row.risk.badge_class : (row.is_fraud ? "risk-critical" : "risk-low");
        const action = row.risk ? row.risk.action : (row.is_fraud ? "Block" : "Approve");

        tr.innerHTML = `
            <td><strong>${row.id}</strong></td>
            <td><span class="status-pill ${row.is_fraud ? 'fraud' : 'legit'}">${row.is_fraud ? '🚨 FRAUD' : '✅ LEGIT'}</span></td>
            <td><span class="risk-badge-tier ${badgeClass}">${riskTier}</span></td>
            <td>${(row.probability * 100).toFixed(2)}%</td>
            <td><small style="color: var(--text-secondary);">${action}</small></td>
        `;
        tableBody.appendChild(tr);
    });

    if (window.innerWidth < 900 && batchContent) {
        batchContent.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

/* ══════════════════════════════════════════════════════════
   Theme Switcher & UI Enhancements
   ══════════════════════════════════════════════════════════ */
function initTheme() {
    const savedTheme = localStorage.getItem("fg_theme") || "dark";
    setTheme(savedTheme);
}

function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("fg_theme", theme);

    const icon = document.getElementById("theme-icon");
    const text = document.getElementById("theme-text");
    if (icon && text) {
        if (theme === "light") {
            icon.textContent = "☀️";
            text.textContent = "Light";
        } else {
            icon.textContent = "🌙";
            text.textContent = "Dark";
        }
    }
}

function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    setTheme(next);
}

/* ══════════════════════════════════════════════════════════
   Floating Toast Notification System
   ══════════════════════════════════════════════════════════ */
function showToast(title, message, type = "info", duration = 4000) {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;

    const icons = {
        danger: "🚨",
        warning: "⚠️",
        success: "✅",
        info: "ℹ️"
    };

    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || "ℹ️"}</span>
        <div class="toast-content">
            <div class="toast-title">${title}</div>
            <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()">✕</button>
    `;

    container.appendChild(toast);

    setTimeout(() => toast.classList.add("show"), 20);

    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.remove(), 400);
    }, duration);
}

/* ══════════════════════════════════════════════════════════
   Slide-Over Detail Drawer
   ══════════════════════════════════════════════════════════ */
function openDetailDrawer(tx) {
    const overlay = document.getElementById("drawer-overlay");
    const drawer = document.getElementById("slide-drawer");
    const title = document.getElementById("drawer-title");
    const body = document.getElementById("drawer-body");

    if (!overlay || !drawer || !body) return;

    if (title) title.textContent = `🔍 Telemetry: ${tx.id || 'Transaction'}`;

    const probPct = ((tx.probability || 0) * 100).toFixed(2);
    const tier = tx.risk_tier || (tx.risk ? tx.risk.tier : (tx.is_fraud ? "CRITICAL" : "LOW"));
    const tierColor = tier === "CRITICAL" ? "#ff4757" : (tier === "HIGH" ? "#ff793f" : (tier === "MODERATE" ? "#ffa502" : "#00d4aa"));
    const status = tx.status || (tx.is_fraud ? "FLAGGED_FRAUD" : "AUTO_APPROVED");

    let driversHtml = "";
    if (tx.risk_drivers && tx.risk_drivers.length > 0) {
        driversHtml = tx.risk_drivers.map(d => `
            <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:12px; padding:8px; background:rgba(255,255,255,0.03); border-radius:6px; border:1px solid rgba(255,255,255,0.05);">
                <span><strong>${d.feature}</strong> (${d.direction || 'anomalous'})</span>
                <span class="impact-tag impact-${d.impact}">${d.impact} (${d.raw_scaled})</span>
            </div>
        `).join("");
    } else {
        driversHtml = `<p style="font-size:12px; color:var(--text-muted);">No anomalous features detected.</p>`;
    }

    let rulesHtml = "";
    const triggeredRules = (tx.hybrid_eval && tx.hybrid_eval.triggered_rules) ? tx.hybrid_eval.triggered_rules : (tx.triggered_rules || []);
    if (triggeredRules.length > 0) {
        rulesHtml = triggeredRules.map(r => `
            <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:12px; padding:8px; background:rgba(255,165,2,0.1); border-radius:6px; border:1px solid rgba(255,165,2,0.3);">
                <span><strong>${r.code}</strong> (${r.name})</span>
                <span style="font-weight:700; color:#ffa502;">+${(r.score_impact * 100).toFixed(0)}% (${r.action})</span>
            </div>
        `).join("");
    } else {
        rulesHtml = `<p style="font-size:12px; color:var(--text-muted);">No business rules triggered.</p>`;
    }

    body.innerHTML = `
        <div style="margin-bottom:20px; padding:16px; background:rgba(255,255,255,0.03); border-radius:12px; border:1px solid var(--border-card);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:12px; text-transform:uppercase; color:var(--text-muted);">Risk Assessment</span>
                <span style="background:${tierColor}22; color:${tierColor}; border:1px solid ${tierColor}44; padding:3px 10px; border-radius:12px; font-weight:700; font-size:11px;">${tier} TIER</span>
            </div>
            <div style="font-size:24px; font-weight:800; color:${tierColor};">${probPct}% Fraud Probability</div>
            <div style="font-size:12px; color:var(--text-secondary); margin-top:4px;">Status: <strong>${status}</strong></div>
        </div>

        <div style="margin-bottom:20px;">
            <h4 style="font-size:13px; font-weight:700; color:var(--text-secondary); text-transform:uppercase; margin-bottom:8px;">🛡️ Triggered Business Rules</h4>
            ${rulesHtml}
        </div>

        <div style="margin-bottom:20px;">
            <h4 style="font-size:13px; font-weight:700; color:var(--text-secondary); text-transform:uppercase; margin-bottom:8px;">⚡ Feature Anomaly Drivers</h4>
            ${driversHtml}
        </div>

        <div style="margin-bottom:20px;">
            <h4 style="font-size:13px; font-weight:700; color:var(--text-secondary); text-transform:uppercase; margin-bottom:8px;">🕒 Timestamp & Metadata</h4>
            <div style="font-size:12px; color:var(--text-secondary); line-height:1.6;">
                <p>Created: <strong>${tx.timestamp ? new Date(tx.timestamp).toLocaleString() : 'N/A'}</strong></p>
                <p>Analyst Decision: <strong>${tx.analyst_decision || 'Pending Review'}</strong></p>
                <p>Analyst Notes: <strong>${tx.notes || 'None'}</strong></p>
            </div>
        </div>
    `;

    overlay.classList.add("open");
    drawer.classList.add("open");
}

function closeDetailDrawer() {
    const overlay = document.getElementById("drawer-overlay");
    const drawer = document.getElementById("slide-drawer");
    if (overlay) overlay.classList.remove("open");
    if (drawer) drawer.classList.remove("open");
}

