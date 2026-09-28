/**
 * OmniSales AI - Universal Sales Prediction & Decision Engine Frontend
 * ====================================================================
 * Reactive state management, Chart.js visualizations, real-time What-If sandbox,
 * CSV drag-and-drop parsing, and executive decision intelligence.
 */

// Global Application State
const state = {
  currentDatasetId: 'ecommerce_retail',
  inspection: null,
  featureStats: {},
  trainResults: null,
  charts: {
    forecast: null,
    bell: null,
    featureImp: null,
    correlation: null
  },
  sandboxInputs: {},
  sentimentPct: 15.0,
  goalTarget: 20000.0,
  isTraining: false,
  debounceTimer: null
};

// DOM Content Loaded Handler
document.addEventListener('DOMContentLoaded', () => {
  initIcons();
  setupNavigationTabs();
  setupPresetButtons();
  setupControlDeck();
  setupDropZone();
  setupSandboxListeners();
  setupModalListeners();

  // Load default dataset on launch
  loadSampleDataset('ecommerce_retail');
});

function initIcons() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

/* ==========================================================================
   Navigation Tabs
   ========================================================================== */
function setupNavigationTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const targetPaneId = `pane-${tab.dataset.tab}`;
      document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
      });

      const activePane = document.getElementById(targetPaneId);
      if (activePane) {
        activePane.classList.add('active');
      }

      // Resize charts if switching to dashboard
      if (tab.dataset.tab === 'dashboard') {
        Object.values(state.charts).forEach(c => c && c.resize());
      }
    });
  });
}

/* ==========================================================================
   Control Deck (Goal, Sentiment & Presets)
   ========================================================================== */
function setupPresetButtons() {
  const presetBtns = document.querySelectorAll('.preset-btn');
  presetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      presetBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const presetId = btn.dataset.preset;
      loadSampleDataset(presetId);
    });
  });
}

function setupControlDeck() {
  // Goal Input
  const goalInput = document.getElementById('goalInput');
  goalInput.addEventListener('change', () => {
    state.goalTarget = parseFloat(goalInput.value) || 0;
    triggerTraining();
  });

  // Quick Step Buttons (+10%, +25%)
  document.querySelectorAll('.quick-step-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const mult = parseFloat(btn.dataset.step);
      const cur = parseFloat(goalInput.value) || 10000;
      const nextVal = Math.round(cur * mult);
      goalInput.value = nextVal;
      state.goalTarget = nextVal;
      triggerTraining();
    });
  });

  // Sentiment Slider
  const sentimentSlider = document.getElementById('sentimentSlider');
  const sentimentBadge = document.getElementById('sentimentBadge');
  const sentimentImpactNote = document.getElementById('sentimentImpactNote');

  sentimentSlider.addEventListener('input', () => {
    const val = parseFloat(sentimentSlider.value);
    state.sentimentPct = val;

    let badgeText = `Neutral Market (0%)`;
    let badgeClass = 'badge-info';

    if (val > 25) {
      badgeText = `Strongly Bullish (+${val}%)`;
      badgeClass = 'badge-success';
    } else if (val > 5) {
      badgeText = `Moderately Bullish (+${val}%)`;
      badgeClass = 'badge-success';
    } else if (val < -25) {
      badgeText = `Strongly Bearish (${val}%)`;
      badgeClass = 'badge-danger';
    } else if (val < -5) {
      badgeText = `Moderately Bearish (${val}%)`;
      badgeClass = 'badge-warning';
    }

    sentimentBadge.textContent = badgeText;
    sentimentBadge.className = `sentiment-indicator-badge ${badgeClass}`;

    const liftPct = ((val / 100.0) * 0.4 * 100).toFixed(1);
    sentimentImpactNote.innerHTML = `
      <i data-lucide="info"></i> Demand sensitivity: ${val >= 0 ? '+' : ''}${liftPct}% expected buyer response
    `;
    initIcons();

    // Debounced retrain
    clearTimeout(state.debounceTimer);
    state.debounceTimer = setTimeout(() => {
      triggerTraining();
    }, 400);
  });

  // Run AI Predict Button
  document.getElementById('runAiPredictBtn').addEventListener('click', () => {
    triggerTraining();
  });
}

/* ==========================================================================
   Dataset Loading & Schema Management
   ========================================================================== */
async function loadSampleDataset(sampleId) {
  try {
    showLoadingStatus(true, `Loading ${sampleId} dataset...`);
    const res = await fetch(`/api/load-sample/${sampleId}`, { method: 'POST' });
    if (!res.ok) throw new Error('Failed to load sample dataset');

    const data = await res.json();
    state.currentDatasetId = sampleId;
    state.inspection = data.inspection;
    state.featureStats = data.feature_stats;

    // Update dataset status pills
    const nameMap = {
      ecommerce_retail: 'E-Commerce Retail',
      saas_b2b: 'SaaS / B2B Subscriptions',
      restaurant_hospitality: 'Restaurant Hospitality',
      electronics_tech: 'Electronics & Tech'
    };
    document.getElementById('currentDatasetText').textContent = nameMap[sampleId] || sampleId;

    // Set smart default goals
    const defaultGoals = {
      ecommerce_retail: 20000,
      saas_b2b: 135000,
      restaurant_hospitality: 11500,
      electronics_tech: 125000
    };
    if (defaultGoals[sampleId]) {
      state.goalTarget = defaultGoals[sampleId];
      document.getElementById('goalInput').value = defaultGoals[sampleId];
    }

    populateSchemaTab();
    await triggerTraining();
  } catch (err) {
    alert(`Error loading sample: ${err.message}`);
  } finally {
    showLoadingStatus(false);
  }
}

function populateSchemaTab() {
  if (!state.inspection) return;

  const targetSelect = document.getElementById('targetColSelect');
  const dateSelect = document.getElementById('dateColSelect');
  const chipsContainer = document.getElementById('featureChipsContainer');

  // Populate Target Column dropdown
  targetSelect.innerHTML = '';
  state.inspection.numeric_columns.forEach(col => {
    const opt = document.createElement('option');
    opt.value = col;
    opt.textContent = col;
    if (col === state.inspection.detected_target) {
      opt.selected = true;
    }
    targetSelect.appendChild(opt);
  });

  // Populate Date Column dropdown
  dateSelect.innerHTML = '<option value="">No date column (Use row sequence)</option>';
  state.inspection.all_columns.forEach(col => {
    const opt = document.createElement('option');
    opt.value = col;
    opt.textContent = col;
    if (col === state.inspection.detected_date) {
      opt.selected = true;
    }
    dateSelect.appendChild(opt);
  });

  // Populate Feature Checkbox chips
  chipsContainer.innerHTML = '';
  state.inspection.suggested_features.forEach(feat => {
    const label = document.createElement('label');
    label.className = 'feature-chip-label checked';
    label.innerHTML = `
      <input type="checkbox" value="${feat}" checked>
      <span>${feat}</span>
    `;
    label.querySelector('input').addEventListener('change', (e) => {
      label.classList.toggle('checked', e.target.checked);
    });
    chipsContainer.appendChild(label);
  });

  // Select All / Deselect All buttons
  document.getElementById('selectAllFeaturesBtn').onclick = () => {
    chipsContainer.querySelectorAll('input').forEach(chk => {
      chk.checked = true;
      chk.parentElement.classList.add('checked');
    });
  };
  document.getElementById('deselectAllFeaturesBtn').onclick = () => {
    chipsContainer.querySelectorAll('input').forEach(chk => {
      chk.checked = false;
      chk.parentElement.classList.remove('checked');
    });
  };

  // Populate Preview Table
  populatePreviewTable();

  // Apply Schema & Train Button
  document.getElementById('applySchemaAndTrainBtn').onclick = () => {
    triggerTraining();
    // Switch to dashboard
    document.querySelector('.tab-btn[data-tab="dashboard"]').click();
  };
}

function populatePreviewTable() {
  const thead = document.getElementById('previewTableHead');
  const tbody = document.getElementById('previewTableBody');
  thead.innerHTML = '';
  tbody.innerHTML = '';

  if (!state.inspection || !state.inspection.preview || !state.inspection.preview.length) return;

  const cols = Object.keys(state.inspection.preview[0]);
  cols.forEach(c => {
    const th = document.createElement('th');
    th.textContent = c;
    thead.appendChild(th);
  });

  state.inspection.preview.forEach(row => {
    const tr = document.createElement('tr');
    cols.forEach(c => {
      const td = document.createElement('td');
      td.textContent = row[c] !== null && row[c] !== undefined ? row[c] : '-';
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  document.getElementById('tablePreviewCount').textContent =
    `Showing first 5 rows of ${state.inspection.row_count} total records`;
}

/* ==========================================================================
   CSV Drag & Drop Upload
   ========================================================================== */
function setupDropZone() {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('csvFileInput');
  const statusMsg = document.getElementById('uploadStatusMessage');

  dropZone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
      handleFileUpload(fileInput.files[0]);
    }
  });

  async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
      statusMsg.innerHTML = '<span class="status-pill"><i data-lucide="loader-2" class="spin"></i> Uploading & parsing CSV...</span>';
      initIcons();

      const res = await fetch('/api/upload-csv', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await res.json();
      state.currentDatasetId = data.filename;
      state.inspection = data.inspection;
      state.featureStats = data.feature_stats;

      document.getElementById('currentDatasetText').textContent = data.filename;
      statusMsg.innerHTML = `<span class="badge-status badge-success">✓ Uploaded ${data.filename} (${data.inspection.row_count} rows)</span>`;

      // Remove active from sample presets
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));

      populateSchemaTab();
      await triggerTraining();

      // Switch to dashboard
      document.querySelector('.tab-btn[data-tab="dashboard"]').click();
    } catch (err) {
      statusMsg.innerHTML = `<span class="badge-status badge-danger">Upload error: ${err.message}</span>`;
    }
  }
}

/* ==========================================================================
   AI Training & Prediction Engine Call
   ========================================================================== */
async function triggerTraining() {
  if (!state.inspection || state.isTraining) return;

  const targetCol = document.getElementById('targetColSelect')?.value || state.inspection.detected_target;
  const dateCol = document.getElementById('dateColSelect')?.value || state.inspection.detected_date || null;
  
  // Selected features
  const selectedFeatures = [];
  document.querySelectorAll('#featureChipsContainer input:checked').forEach(c => {
    selectedFeatures.push(c.value);
  });

  const features = selectedFeatures.length > 0 ? selectedFeatures : state.inspection.suggested_features;

  const payload = {
    target_column: targetCol,
    feature_columns: features,
    date_column: dateCol,
    sentiment_pct: state.sentimentPct,
    goal_target: state.goalTarget
  };

  try {
    state.isTraining = true;
    showLoadingStatus(true, 'Training multiple ML models...');

    const res = await fetch('/api/train-and-predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Training failed');
    }

    const data = await res.json();
    state.trainResults = data;

    renderDashboardMetrics(data);
    renderForecastChart(data.forecast, data.goal_analysis.goal_target);
    renderGoalBellCurveChart(data.goal_analysis);
    renderFeatureImportanceChart(data.feature_importance);
    renderCorrelationChart(data.correlations);
    renderHiddenPatterns(data.hidden_patterns);
    renderStrategicRecommendations(data.strategic_recommendations);
    renderSandboxSliders(features);

    // Update champion badge
    document.getElementById('championModelText').textContent =
      `${data.champion_model.name} (${data.champion_model.accuracy_percent}%)`;

  } catch (err) {
    console.error(err);
    alert(`Model Training Error: ${err.message}`);
  } finally {
    state.isTraining = false;
    showLoadingStatus(false);
  }
}

function showLoadingStatus(isLoading, text = '') {
  const pill = document.getElementById('engineStatusPill');
  const dot = pill.querySelector('.status-dot');
  const statusText = document.getElementById('engineStatusText');

  if (isLoading) {
    dot.className = 'status-dot';
    dot.style.backgroundColor = '#f59e0b';
    statusText.textContent = text || 'Computing...';
  } else {
    dot.className = 'status-dot pulse';
    dot.style.backgroundColor = '#10b981';
    statusText.textContent = 'ML Engine Ready';
  }
}

/* ==========================================================================
   Dashboard Metrics Rendering
   ========================================================================== */
function renderDashboardMetrics(data) {
  const summary = data.forecast.summary;
  const goal = data.goal_analysis;
  const champion = data.champion_model;

  // KPI 1: Projected Sales
  document.getElementById('kpiProjectedSales').textContent = formatCurrency(summary.expected_average_sales);
  const growthElem = document.getElementById('kpiGrowthDelta');
  const growthPct = summary.growth_vs_baseline_pct;
  growthElem.textContent = `${growthPct >= 0 ? '+' : ''}${growthPct.toFixed(1)}% vs History`;
  growthElem.className = `kpi-delta ${growthPct >= 0 ? 'positive' : 'negative'}`;

  const firstFuture = data.forecast.future_projections[0] || {};
  document.getElementById('kpiConfidenceInterval').textContent =
    `P10 - P90: ${formatCurrency(firstFuture.lower_bound_p10 || 0)} - ${formatCurrency(firstFuture.upper_bound_p90 || 0)}`;

  // KPI 2: Goal Probability
  const prob = goal.probability_percent;
  document.getElementById('kpiGoalProbability').textContent = `${prob}%`;
  
  const goalBadge = document.getElementById('kpiGoalBadge');
  goalBadge.textContent = goal.status;
  goalBadge.className = `badge-status badge-${goal.status_badge}`;

  document.getElementById('goalProgressBar').style.width = `${Math.min(100, Math.max(5, prob))}%`;
  document.getElementById('kpiGoalGapText').textContent =
    goal.gap_amount > 0 ? `Gap: ${formatCurrency(goal.gap_amount)} to target` : 'Exceeding target trajectory!';

  // KPI 3: Champion Accuracy
  document.getElementById('kpiModelAccuracy').textContent = `${champion.accuracy_percent}% R²`;
  document.getElementById('kpiChampionName').textContent = champion.name;
  document.getElementById('kpiMae').textContent = formatCurrency(champion.metrics.mae);
  document.getElementById('kpiRmse').textContent = formatCurrency(champion.metrics.rmse);
  document.getElementById('kpiCvScore').textContent = `${(champion.metrics.cv_r2 * 100).toFixed(1)}%`;

  // KPI 4: Primary Revenue Driver
  const topFeatKey = Object.keys(data.feature_importance)[0];
  if (topFeatKey) {
    const topFeat = data.feature_importance[topFeatKey];
    document.getElementById('kpiTopDriver').textContent = topFeatKey.replace(/_/g, ' ');
    document.getElementById('kpiTopDriverImpact').textContent = `${topFeat.importance_pct}% Influence`;
    document.getElementById('kpiTopDriverDirection').textContent =
      `${topFeat.direction === 'positive' ? 'Positive' : 'Negative'} (Corr: ${topFeat.correlation > 0 ? '+' : ''}${topFeat.correlation})`;
  }

  // Goal Verdict
  document.getElementById('goalVerdictText').textContent = goal.projected_vs_goal_verdict;
}

/* ==========================================================================
   Chart 1: Probabilistic Forecast Line Chart
   ========================================================================== */
function renderForecastChart(forecastData, goalTarget) {
  const ctx = document.getElementById('forecastChart').getContext('2d');

  if (state.charts.forecast) {
    state.charts.forecast.destroy();
  }

  const histPoints = forecastData.historical_points;
  const futurePoints = forecastData.future_projections;

  const labels = [
    ...histPoints.map(p => p.label),
    ...futurePoints.map(p => p.period)
  ];

  // Actual Historical series
  const actualData = [
    ...histPoints.map(p => p.actual_sales),
    ...futurePoints.map(() => null)
  ];

  // Expected Forecast P50 (connects to the last historical point)
  const lastHistVal = histPoints[histPoints.length - 1]?.actual_sales || null;
  const p50Data = [
    ...histPoints.slice(0, -1).map(() => null),
    lastHistVal,
    ...futurePoints.map(p => p.expected_sales)
  ];

  // P90 (Upper bound)
  const p90Data = [
    ...histPoints.slice(0, -1).map(() => null),
    lastHistVal,
    ...futurePoints.map(p => p.upper_bound_p90)
  ];

  // P10 (Lower bound)
  const p10Data = [
    ...histPoints.slice(0, -1).map(() => null),
    lastHistVal,
    ...futurePoints.map(p => p.lower_bound_p10)
  ];

  // Goal line series
  const goalLine = labels.map(() => goalTarget);

  state.charts.forecast = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Historical Sales',
          data: actualData,
          borderColor: '#94a3b8',
          backgroundColor: 'rgba(148, 163, 184, 0.1)',
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: '#94a3b8',
          tension: 0.25
        },
        {
          label: 'Expected Forecast (P50)',
          data: p50Data,
          borderColor: '#6366f1',
          backgroundColor: '#6366f1',
          borderWidth: 3,
          pointRadius: 4,
          pointBackgroundColor: '#ffffff',
          tension: 0.3
        },
        {
          label: 'Upper Confidence (P90)',
          data: p90Data,
          borderColor: 'rgba(99, 102, 241, 0.4)',
          borderDash: [5, 5],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: '+1',
          backgroundColor: 'rgba(99, 102, 241, 0.12)',
          tension: 0.3
        },
        {
          label: 'Lower Confidence (P10)',
          data: p10Data,
          borderColor: 'rgba(99, 102, 241, 0.4)',
          borderDash: [5, 5],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
          tension: 0.3
        },
        {
          label: 'Goal Target',
          data: goalLine,
          borderColor: '#f59e0b',
          borderDash: [6, 4],
          borderWidth: 2,
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(17, 24, 39, 0.95)',
          titleFont: { family: 'Outfit', size: 13 },
          bodyFont: { family: 'Inter', size: 12 },
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          padding: 12,
          callbacks: {
            label: (ctx) => {
              if (ctx.raw === null || ctx.raw === undefined) return null;
              return `${ctx.dataset.label}: ${formatCurrency(ctx.raw)}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#64748b', font: { family: 'Inter', size: 10 } }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#64748b',
            font: { family: 'Inter', size: 10 },
            callback: (val) => formatShortNumber(val)
          }
        }
      }
    }
  });
}

/* ==========================================================================
   Chart 2: Goal Probability Bell Curve
   ========================================================================== */
function renderGoalBellCurveChart(goalAnalysis) {
  const ctx = document.getElementById('goalBellChart').getContext('2d');

  if (state.charts.bell) {
    state.charts.bell.destroy();
  }

  const points = goalAnalysis.bell_curve_points;
  const labels = points.map(p => formatShortNumber(p.x));
  const densities = points.map(p => p.density);

  state.charts.bell = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Forecast Outcome Density',
          data: densities,
          borderColor: '#06b6d4',
          borderWidth: 2.5,
          fill: true,
          backgroundColor: 'rgba(6, 182, 212, 0.15)',
          pointRadius: 0,
          tension: 0.4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(17, 24, 39, 0.95)',
          padding: 10,
          callbacks: {
            title: (items) => `Projected Value: $${items[0].label}`,
            label: () => `Goal Target: $${goalAnalysis.goal_target.toLocaleString()}`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#64748b', font: { size: 9 }, maxTicksLimit: 7 }
        },
        y: {
          display: false
        }
      }
    }
  });
}

/* ==========================================================================
   Chart 3: Key Parameter Importance Ranking
   ========================================================================== */
function renderFeatureImportanceChart(importances) {
  const ctx = document.getElementById('featureImportanceChart').getContext('2d');

  if (state.charts.featureImp) {
    state.charts.featureImp.destroy();
  }

  const labels = Object.keys(importances).map(k => k.replace(/_/g, ' '));
  const values = Object.values(importances).map(v => v.importance_pct);
  const colors = Object.values(importances).map(v =>
    v.direction === 'positive' ? 'rgba(16, 185, 129, 0.85)' : 'rgba(244, 63, 94, 0.85)'
  );

  state.charts.featureImp = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          data: values,
          backgroundColor: colors,
          borderRadius: 6,
          borderSkipped: false
        }
      ]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Predictive Power: ${ctx.raw}%`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#64748b', callback: (v) => `${v}%` }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#cbd5e1', font: { size: 11, family: 'Inter' } }
        }
      }
    }
  });
}

/* ==========================================================================
   Chart 4: Parameter Correlations
   ========================================================================== */
function renderCorrelationChart(correlations) {
  const ctx = document.getElementById('correlationChart').getContext('2d');

  if (state.charts.correlation) {
    state.charts.correlation.destroy();
  }

  const labels = Object.keys(correlations).map(k => k.replace(/_/g, ' '));
  const values = Object.values(correlations);
  const colors = values.map(v =>
    v >= 0 ? 'rgba(99, 102, 241, 0.85)' : 'rgba(245, 158, 11, 0.85)'
  );

  state.charts.correlation = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          data: values,
          backgroundColor: colors,
          borderRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Correlation: ${ctx.raw > 0 ? '+' : ''}${ctx.raw}`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#cbd5e1', font: { size: 10 } }
        },
        y: {
          min: -1.0,
          max: 1.0,
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#64748b' }
        }
      }
    }
  });
}

/* ==========================================================================
   Tab 3: Hidden Patterns & Insights Rendering
   ========================================================================== */
function renderHiddenPatterns(hiddenPatterns) {
  const grid = document.getElementById('patternsGrid');
  grid.innerHTML = '';

  if (!hiddenPatterns.patterns || !hiddenPatterns.patterns.length) {
    grid.innerHTML = '<div class="empty-state"><p>No non-linear patterns detected in this dataset sample.</p></div>';
    return;
  }

  hiddenPatterns.patterns.forEach(pat => {
    const card = document.createElement('div');
    card.className = 'pattern-card glass-card';
    card.innerHTML = `
      <div class="pattern-card-header">
        <span class="pattern-type-badge">${pat.type}</span>
      </div>
      <div class="pattern-title">
        <i data-lucide="${pat.icon || 'sparkles'}"></i>
        <span>${pat.title}</span>
      </div>
      <p class="pattern-desc">${pat.description}</p>
    `;
    grid.appendChild(card);
  });

  initIcons();
}

/* ==========================================================================
   Tab 4: Strategic Recommendations Rendering
   ========================================================================== */
function renderStrategicRecommendations(recs) {
  const list = document.getElementById('recommendationsList');
  list.innerHTML = '';

  if (!recs || !recs.length) {
    list.innerHTML = '<div class="empty-state"><p>No recommendations generated.</p></div>';
    return;
  }

  recs.forEach((rec, idx) => {
    const card = document.createElement('div');
    card.className = 'recommendation-card glass-card';
    card.innerHTML = `
      <div class="rec-priority-num">${idx + 1}</div>
      <div class="rec-body">
        <div class="rec-header-row">
          <span class="rec-badge rec-badge-${rec.badge_color}">${rec.badge}</span>
          <h4 class="rec-title">${rec.title}</h4>
        </div>
        <p class="rec-action">${rec.action}</p>
        <span class="rec-impact-tag">
          <i data-lucide="trending-up"></i> ${rec.expected_impact}
        </span>
      </div>
    `;
    list.appendChild(card);
  });

  initIcons();
}

/* ==========================================================================
   Tab 2: What-If Sandbox Sliders & Real-Time Simulation
   ========================================================================== */
function renderSandboxSliders(features) {
  const container = document.getElementById('sandboxSlidersList');
  container.innerHTML = '';
  state.sandboxInputs = {};

  features.forEach(feat => {
    const stats = state.featureStats[feat] || { min: 0, max: 100, mean: 50, step: 1 };
    state.sandboxInputs[feat] = stats.mean;

    const group = document.createElement('div');
    group.className = 'slider-group';
    group.innerHTML = `
      <div class="slider-top-row">
        <span class="slider-label">${feat.replace(/_/g, ' ')}</span>
        <span class="slider-value-badge" id="val-${feat}">${stats.mean.toLocaleString()}</span>
      </div>
      <div class="slider-range-row">
        <span class="range-min">${stats.min.toLocaleString()}</span>
        <input type="range" class="slider-input" id="range-${feat}"
               min="${stats.min}" max="${stats.max}" step="${stats.step}" value="${stats.mean}">
        <span class="range-max">${stats.max.toLocaleString()}</span>
      </div>
    `;

    const slider = group.querySelector('.slider-input');
    const badge = group.querySelector(`#val-${feat}`);

    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      state.sandboxInputs[feat] = val;
      badge.textContent = val.toLocaleString();
      debounceScenarioSimulation();
    });

    container.appendChild(group);
  });

  // Run initial simulation
  runScenarioSimulation();
}

function setupSandboxListeners() {
  document.getElementById('resetSandboxBtn').addEventListener('click', () => {
    Object.keys(state.sandboxInputs).forEach(feat => {
      const stats = state.featureStats[feat];
      if (stats) {
        state.sandboxInputs[feat] = stats.mean;
        const slider = document.getElementById(`range-${feat}`);
        const badge = document.getElementById(`val-${feat}`);
        if (slider) slider.value = stats.mean;
        if (badge) badge.textContent = stats.mean.toLocaleString();
      }
    });
    runScenarioSimulation();
  });
}

function debounceScenarioSimulation() {
  clearTimeout(state.debounceTimer);
  state.debounceTimer = setTimeout(() => {
    runScenarioSimulation();
  }, 120);
}

async function runScenarioSimulation() {
  if (!state.trainResults) return;

  const payload = {
    custom_inputs: state.sandboxInputs,
    sentiment_pct: state.sentimentPct,
    goal_target: state.goalTarget
  };

  try {
    const res = await fetch('/api/simulate-scenario', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error('Simulation failed');

    const result = await res.json();
    renderSimulationResults(result);
  } catch (err) {
    console.error('Scenario error:', err);
  }
}

function renderSimulationResults(res) {
  document.getElementById('simPredictedSales').textContent = formatCurrency(res.predicted_sales);

  const deltaElem = document.getElementById('simDeltaBadge');
  deltaElem.textContent = `${res.delta_percent >= 0 ? '+' : ''}${res.delta_percent.toFixed(1)}% vs Baseline`;
  deltaElem.className = `kpi-delta ${res.delta_percent >= 0 ? 'positive' : 'negative'}`;

  document.getElementById('simDeltaAmount').textContent =
    `(${res.delta_amount >= 0 ? '+' : ''}${formatCurrency(res.delta_amount)} difference)`;

  const prob = res.goal_probability_percent;
  document.getElementById('simGoalProbPercent').textContent = `${prob}%`;
  document.getElementById('simGoalProgressBar').style.width = `${Math.min(100, Math.max(5, prob))}%`;
  document.getElementById('simGoalVerdict').textContent =
    `Target Goal: ${formatCurrency(res.goal_target || state.goalTarget)} • Status: ${
      prob >= 75 ? 'Goal Achieved' : prob >= 50 ? 'Close to Goal' : 'Under Target'
    }`;

  // Snapshot chips
  const chipsWrap = document.getElementById('simSnapshotChips');
  chipsWrap.innerHTML = '';
  Object.entries(res.inputs_used).forEach(([k, v]) => {
    const chip = document.createElement('span');
    chip.className = 'snapshot-chip';
    chip.innerHTML = `${k.replace(/_/g, ' ')}: <strong>${v.toLocaleString()}</strong>`;
    chipsWrap.appendChild(chip);
  });
}

/* ==========================================================================
   Executive Report Modal
   ========================================================================== */
function setupModalListeners() {
  const modal = document.getElementById('reportModal');
  const openBtn = document.getElementById('exportReportBtn');
  const closeBtn = document.getElementById('closeReportModalBtn');
  const dismissBtn = document.getElementById('dismissReportModalBtn');

  openBtn.addEventListener('click', async () => {
    if (!state.trainResults) {
      alert('Please train a model first!');
      return;
    }
    await populateExecutiveReport();
    modal.classList.add('active');
  });

  [closeBtn, dismissBtn].forEach(b => {
    b.addEventListener('click', () => modal.classList.remove('active'));
  });

  modal.addEventListener('click', (e) => {
    if (e.target === modal) modal.classList.remove('active');
  });
}

async function populateExecutiveReport() {
  const data = state.trainResults;
  const content = document.getElementById('reportBodyContent');

  content.innerHTML = `
    <div style="margin-bottom: 20px;">
      <h2 style="color: #ffffff; font-family: 'Outfit'; font-size: 22px; margin-bottom: 6px;">
        Sales Prediction & Growth Strategy Briefing
      </h2>
      <p style="color: #94a3b8; font-size: 13px;">
        Dataset: <strong>${state.currentDatasetId}</strong> • Date Generated: ${new Date().toLocaleDateString()}
      </p>
    </div>

    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 24px;">
      <div style="background: rgba(255,255,255,0.03); padding: 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">
        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">Expected Average Sales</div>
        <div style="font-size: 20px; font-weight: 700; color: #6366f1; margin-top: 4px;">
          ${formatCurrency(data.forecast.summary.expected_average_sales)}
        </div>
      </div>
      <div style="background: rgba(255,255,255,0.03); padding: 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">
        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">Goal Probability</div>
        <div style="font-size: 20px; font-weight: 700; color: #10b981; margin-top: 4px;">
          ${data.goal_analysis.probability_percent}%
        </div>
      </div>
      <div style="background: rgba(255,255,255,0.03); padding: 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">
        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">Champion Model</div>
        <div style="font-size: 20px; font-weight: 700; color: #38bdf8; margin-top: 4px;">
          ${data.champion_model.name} (${data.champion_model.accuracy_percent}%)
        </div>
      </div>
    </div>

    <h3 style="color: #f8fafc; font-size: 15px; margin-bottom: 10px;">Executive Summary & Goal Verdict</h3>
    <p style="color: #cbd5e1; margin-bottom: 18px; line-height: 1.6;">
      ${data.goal_analysis.projected_vs_goal_verdict}
      Our machine learning evaluation tested multiple models (Gradient Boosting, Random Forest, Ridge, ElasticNet, and Voting Ensemble).
      The champion architecture <strong>${data.champion_model.name}</strong> achieved an R² accuracy of 
      <strong>${data.champion_model.accuracy_percent}%</strong> with a Mean Absolute Error of 
      <strong>${formatCurrency(data.champion_model.metrics.mae)}</strong>.
    </p>

    <h3 style="color: #f8fafc; font-size: 15px; margin-bottom: 10px;">Top Strategic Next Steps</h3>
    <div style="display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px;">
      ${data.strategic_recommendations.slice(0, 3).map((rec, i) => `
        <div style="padding: 12px; background: rgba(255,255,255,0.02); border-left: 3px solid #6366f1; border-radius: 4px;">
          <div style="font-weight: 600; color: #ffffff; margin-bottom: 4px;">${i + 1}. ${rec.title}</div>
          <div style="font-size: 12px; color: #94a3b8;">${rec.action}</div>
        </div>
      `).join('')}
    </div>
  `;
}

/* ==========================================================================
   Utilities & Formatting
   ========================================================================== */
function formatCurrency(val) {
  if (val === null || val === undefined) return '$0.00';
  return '$' + Number(val).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });
}

function formatShortNumber(val) {
  if (val >= 1000000) return (val / 1000000).toFixed(1) + 'M';
  if (val >= 1000) return (val / 1000).toFixed(0) + 'K';
  return val.toFixed(0);
}
