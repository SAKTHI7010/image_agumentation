const $ = (id) => document.getElementById(id);

let selectedTileEl = null;

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $("tab-" + btn.dataset.tab).classList.add("active");
    });
  });
}

function populateCategories() {
  const sel = $("categoryFilter");
  for (let i = 1; i <= 45; i++) {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `Category ${i}`;
    sel.appendChild(opt);
  }
}

async function loadSummary() {
  const res = await fetch("/api/summary");
  const data = await res.json();
  const insp = data.inspection || {};
  const split = data.split_validation || {};
  const cc = data.classifier_comparison || {};
  const reg = data.regression_report || {};
  const drift = data.drift_evaluation || {};
  const p3 = data.project3_status || {};

  $("headerBadges").innerHTML = `
    <span class="badge ok">${insp.valid_images ?? "?"} Valid Images · ${insp.categories_total ?? 45} Categories</span>
    <span class="badge">${split.num_batches ?? 5} Physical Batches</span>
    <span class="badge warn">Project 3: ${p3.integration_status || "pending"}</span>
  `;

  $("benchDataset").innerHTML = `
    <p><strong>Original Images:</strong> ${insp.total_files_scanned} (${Object.keys(insp.extensions || {}).join(", ")}, ${Object.keys(insp.dimensions || {}).join(", ")})</p>
    <p><strong>Anomalies:</strong> ${insp.corrupt_or_unsupported} corrupt, ${insp.exact_duplicates} exact duplicates, ${insp.near_duplicate_clusters_dhash_le_3} near-dup clusters (${insp.unique_leakage_groups} leakage-safe groups)</p>
    <p><strong>Physical Split:</strong> ${split.num_batches} batches (${Object.entries(split.batch_counts || {}).map(([k,v]) => `${k}: ${v}`).join(", ")}) · 1-to-1 match: <strong>${split.exact_one_to_one_match}</strong></p>
  `;

  const rf = cc.random_forest_45class || {};
  const cnn = cc.mobilenet_v2_45class || {};
  const stab = cc.random_forest_stability_within_split || {};
  $("benchClassifiers").innerHTML = `
    <table>
      <thead><tr><th>Model (45-Class)</th><th>Accuracy</th><th>Macro-F1</th><th>Weighted-F1</th><th>Train Time (s)</th></tr></thead>
      <tbody>
        <tr><td>Random Forest (Visual Features)</td><td>${((rf.accuracy || 0) * 100).toFixed(2)}%</td><td>${rf.macro_f1 ?? "-"}</td><td>${rf.weighted_f1 ?? "-"}</td><td>${rf.train_time_sec ?? "-"}</td></tr>
        <tr><td>MobileNetV2 CNN (224x224)</td><td>${((cnn.accuracy || 0) * 100).toFixed(2)}%</td><td>${cnn.macro_f1 ?? "-"}</td><td>${cnn.weighted_f1 ?? "-"}</td><td>${cnn.train_time_sec ?? "-"}</td></tr>
      </tbody>
    </table>
    <p class="muted" style="margin-top:8px;">Within-split Binary Stability RF Accuracy: <strong>${((stab.accuracy || 0) * 100).toFixed(2)}%</strong> (Macro-F1: ${stab.macro_f1 ?? "-"})</p>
  `;

  const audit = reg.metadata_audit || {};
  const wt = reg.within_condition_test_metrics || {};
  const gcv = reg.category_heldout_5fold_cv_metrics || {};
  $("benchRegression").innerHTML = `
    <p><strong>Froth Depth Available?</strong> ${audit.froth_depth_available ? "Yes" : "No"} — <span class="muted">${audit.froth_depth_note || ""}</span></p>
    <p><strong>Air Flow [L/min] (${audit.air_flow_valid_images} valid images across ${audit.air_flow_valid_categories} categories; 3 'Unknown' excluded):</strong></p>
    <p>Within-condition Test: MAE = <strong>${wt.mae}</strong> L/min, RMSE = <strong>${wt.rmse}</strong>, R² = <strong>${wt.r2}</strong></p>
    <p>Category-held-out 5-Fold GroupCV: MAE = <strong>${gcv.mae}</strong> L/min, RMSE = <strong>${gcv.rmse}</strong>, R² = <strong>${gcv.r2}</strong></p>
  `;

  const exp1 = drift.experiment_1_closed_set_stable || {};
  const exp2 = drift.experiment_2_heldout_unsteady_novelty_detection || {};
  const exp3 = drift.experiment_3_cross_category_stability_5fold_groupcv || {};
  const comb = exp2.combined_novelty_score || {};
  const mahal = exp2.feature_mahalanobis_distance || {};
  $("benchDrift").innerHTML = `
    <p><strong>Closed-Set Stable Categories (28 cats):</strong> Test Acc = <strong>${((exp1.accuracy || 0) * 100).toFixed(2)}%</strong>, Macro-F1 = ${exp1.macro_f1}</p>
    <p><strong>Held-Out Unsteady Novelty Detection (17 unseen Unstable cats, 850 images):</strong></p>
    <ul>
      <li>Combined Novelty Score: AUROC = <strong>${comb.auroc}</strong>, AUPRC = <strong>${comb.auprc}</strong>, TPR@5%FPR = <strong>${((comb.tpr_at_5pct_fpr || 0) * 100).toFixed(1)}%</strong></li>
      <li>Feature Mahalanobis Distance: AUROC = <strong>${mahal.auroc}</strong>, TPR@5%FPR = <strong>${((mahal.tpr_at_5pct_fpr || 0) * 100).toFixed(1)}%</strong></li>
    </ul>
    <p><strong>Cross-Category Stability Generalization (5-Fold GroupKFold by Category):</strong> Acc = <strong>${((exp3.accuracy || 0) * 100).toFixed(2)}%</strong>, Macro-F1 = <strong>${exp3.macro_f1}</strong></p>
  `;

  $("benchProject3").innerHTML = `
    <p><strong>Status:</strong> <span class="badge warn">${p3.integration_status}</span></p>
    <p class="muted">Adapter interface <code>Project3KineticModelAdapter</code> is implemented in <code>backend/project3_adapter.py</code>. Missing artifacts required to connect live recovery prediction:</p>
    <ul>${(p3.missing_requirements || []).map((m) => `<li>${m}</li>`).join("")}</ul>
  `;

  const tbody = $("unsteadyTable").querySelector("tbody");
  tbody.innerHTML = (data.unsteady_categories_detail || []).map((r) => `
    <tr>
      <td>${r.category_id}</td>
      <td>${r.category}</td>
      <td>${r.mineral_used}</td>
      <td>${r.air_flow_raw}</td>
      <td>${r.quality}</td>
      <td>${r.sample_count}</td>
      <td>${r.mean_uncertainty}</td>
      <td>${r.mean_mahalanobis}</td>
      <td>${r.mean_novelty_score}</td>
      <td>${(r.drift_flag_rate * 100).toFixed(1)}%</td>
    </tr>
  `).join("");
}

async function loadSamples() {
  const batch = $("batchFilter").value;
  const stab = $("stabilityFilter").value;
  const cat = $("categoryFilter").value;
  const params = new URLSearchParams({ batch_id: batch, stability: stab, limit: "60" });
  if (cat) params.set("category_id", cat);

  const res = await fetch(`/api/samples?${params.toString()}`);
  const data = await res.json();
  const gal = $("sampleGallery");
  gal.innerHTML = "";
  for (const s of data.samples || []) {
    const tile = document.createElement("button");
    tile.type = "button";
    tile.className = "sample-tile";
    tile.innerHTML = `
      <img src="${s.thumbnail_url}" alt="${s.filename}" loading="lazy" />
      <div class="cap">#${s.image_id} · ${s.category}</div>
      <div class="subcap">${s.stability} · ${s.batch_id}</div>
    `;
    tile.addEventListener("click", () => {
      if (selectedTileEl) selectedTileEl.classList.remove("selected");
      tile.classList.add("selected");
      selectedTileEl = tile;
      predictLocalSample(s);
    });
    gal.appendChild(tile);
  }
}

function showError(msg) {
  $("errorBanner").style.display = "block";
  $("errorBanner").textContent = msg;
}
function clearError() {
  $("errorBanner").style.display = "none";
}

function renderPrediction(data, previewUrl) {
  clearError();
  $("predictionPlaceholder").style.display = "none";
  $("predictionResults").style.display = "block";

  if (previewUrl) {
    $("previewWrap").style.display = "block";
    $("previewImg").src = previewUrl;
  }

  const gt = data.catalog_ground_truth;
  if (gt) {
    $("groundTruthMeta").innerHTML = `
      <span><strong>Ground Truth:</strong> ${gt.category}</span>
      <span><strong>Stability:</strong> ${gt.stability}</span>
      <span><strong>Quality:</strong> ${gt.quality}</span>
      <span><strong>Mineral:</strong> ${gt.mineral_used}</span>
      <span><strong>Air Flow:</strong> ${gt.air_flow_raw} L/min</span>
    `;
  } else {
    $("groundTruthMeta").innerHTML = `<span>Uploaded image (no catalog ground truth)</span>`;
  }

  const p = data.predictions;
  const u = data.uncertainty_and_drift;
  const vf = data.visual_features;

  $("kpiRfCat").textContent = p.random_forest_category;
  $("kpiRfConf").textContent = `Conf: ${(p.random_forest_confidence * 100).toFixed(1)}%`;
  $("kpiCnnCat").textContent = p.cnn_category;
  $("kpiCnnConf").textContent = `Conf: ${(p.cnn_confidence * 100).toFixed(1)}%`;
  $("kpiStability").textContent = p.predicted_stability;
  $("kpiStabScore").textContent = `Stability score: ${(p.stability_score * 100).toFixed(1)}%`;
  $("kpiAirFlow").textContent = `${p.estimated_air_flow_l_min} L/min`;
  $("kpiProxy").textContent = `Visual proxy: ${p.visual_stability_proxy}`;

  const driftEl = $("driftBanner");
  driftEl.style.display = "block";
  if (u.drift_warning || u.outside_training_distribution || (u.reasons && u.reasons.length)) {
    driftEl.className = "alert warn";
    const metaWarn = (u.metadata_warnings || []).length ? `<br/><em>Metadata note: ${u.metadata_warnings.join(" ")}</em>` : "";
    driftEl.innerHTML = `<strong>${u.drift_warning ? "DRIFT / UNCERTAINTY WARNING" : "Advisory Notice"}:</strong> ${(u.reasons || ["Elevated uncertainty detected."]).join(" ")}${metaWarn}`;
  } else {
    driftEl.className = "alert ok";
    const metaWarn = (u.metadata_warnings || []).length ? ` (${u.metadata_warnings.join(" ")})` : "";
    driftEl.innerHTML = `<strong>In-Distribution &amp; Steady Froth:</strong> Novelty score ${u.novelty_score} within normal operating envelope.${metaWarn}`;
  }

  $("top3List").innerHTML = `
    <p class="muted" style="margin:0 0 6px;">Random Forest Top-3:</p>
    ${p.random_forest_top3.map((t) => `<div>${t.category}: <strong>${(t.probability * 100).toFixed(1)}%</strong></div>`).join("")}
    <p class="muted" style="margin:8px 0 6px;">MobileNetV2 Top-3:</p>
    ${p.cnn_top3.map((t) => `<div>${t.category}: <strong>${(t.probability * 100).toFixed(1)}%</strong></div>`).join("")}
  `;

  $("uncertaintyStats").innerHTML = `
    <div>RF Uncertainty (1 - max P): <strong>${u.rf_uncertainty}</strong></div>
    <div>CNN Uncertainty (1 - max P): <strong>${u.cnn_uncertainty}</strong></div>
    <div>Mahalanobis Distance: <strong>${u.mahalanobis_distance}</strong> (p95=${u.mahalanobis_p95_threshold}, p99=${u.mahalanobis_p99_threshold})</div>
    <div>Combined Novelty Score: <strong>${u.novelty_score}</strong></div>
    <div>Outside Training Dist: <strong>${u.outside_training_distribution ? "YES" : "No"}</strong></div>
  `;

  const featPairs = [
    ["Dimensions", `${vf.orig_width}x${vf.orig_height}`],
    ["Seg. Status", `${vf.segmentation_status} (${vf.segmentation_confidence})`],
    ["Bubble Count", vf.bubble_count],
    ["Bubble D10 / D50 / D90", `${vf.bubble_d10 ?? "-"} / ${vf.bubble_d50 ?? "-"} / ${vf.bubble_d90 ?? "-"} px`],
    ["Bubble Size Span", vf.bubble_size_span ?? "-"],
    ["Circularity", vf.bubble_circularity_mean ?? "-"],
    ["Texture Entropy", vf.texture_entropy],
    ["GLCM Homogeneity", vf.glcm_homogeneity],
    ["Laplacian Focus", vf.laplacian_var],
    ["RGB Mean", `${vf.r_mean}, ${vf.g_mean}, ${vf.b_mean}`],
    ["HSV Mean", `${vf.h_mean}, ${vf.s_mean}, ${vf.v_mean}`],
    ["Dark / Bright Frac", `${vf.dark_fraction} / ${vf.bright_fraction}`],
    ["Froth Velocity", vf.velocity_available ? vf.froth_velocity_px_s : "Unavailable (no frame timestamps)"],
  ];
  $("visualFeaturesGrid").innerHTML = featPairs
    .map(([k, v]) => `<div class="feat-item"><span>${k}</span><strong>${v}</strong></div>`)
    .join("");

  const p3 = data.project3_integration;
  const opOnly = p3.operating_only_payload || {};
  const opPlus = p3.operating_plus_image_payload || {};
  const fmtVal = (v, unit = "") => (v === null || v === undefined || v === "" ? '<span class="muted">Not provided</span>' : `${v}${unit}`);
  $("project3Panel").innerHTML = `
    <p><strong>Integration Status:</strong> <span class="badge warn">Pending Project 3 Model</span> — ${p3.message}</p>
    <div class="p3-compare">
      <div class="p3-col">
        <h5>Baseline: Operating Variables Only</h5>
        <div class="p3-row"><span>Air Flow</span><strong>${fmtVal(opOnly.air_flow_l_min, " L/min")}</strong></div>
        <div class="p3-row"><span>Collector Dosage</span><strong>${fmtVal(opOnly.collector_dosage_gt, " g/t")}</strong></div>
        <div class="p3-row"><span>Frother Condition</span><strong>${fmtVal(opOnly.frother_condition_gt, " g/t")}</strong></div>
        <div class="p3-row"><span>Water Type</span><strong>${fmtVal(opOnly.kind_of_water)}</strong></div>
        <div class="p3-row"><span>Mineral Origin</span><strong>${fmtVal(opOnly.mineral_used)}</strong></div>
        <div class="p3-row"><span>Residence Time</span><strong>${fmtVal(opOnly.residence_time_min, " min")}</strong></div>
        <div class="p3-row"><span>Impeller Speed</span><strong>${fmtVal(opOnly.impeller_rpm, " RPM")}</strong></div>
        <div class="p3-row"><span>Predicted Recovery</span><strong>${p3.recovery_operating_only ?? '<span class="muted">Pending Project 3</span>'}</strong></div>
      </div>
      <div class="p3-col">
        <h5>Augmented: Operating + Froth Vision</h5>
        <div class="p3-row"><span>Froth Stability Score</span><strong>${((opPlus.froth_stability_score ?? 0) * 100).toFixed(1)}% (${opPlus.predicted_stability_label})</strong></div>
        <div class="p3-row"><span>Visual Stability Proxy</span><strong>${fmtVal(opPlus.visual_stability_proxy)}</strong></div>
        <div class="p3-row"><span>Bubble D50</span><strong>${fmtVal(opPlus.bubble_d50_px, " px")}</strong></div>
        <div class="p3-row"><span>Bubble Size Span</span><strong>${fmtVal(opPlus.bubble_size_span)}</strong></div>
        <div class="p3-row"><span>Texture Entropy</span><strong>${fmtVal(opPlus.texture_entropy)}</strong></div>
        <div class="p3-row"><span>Dark Load Fraction</span><strong>${fmtVal(opPlus.dark_fraction)}</strong></div>
        <div class="p3-row"><span>Novelty / Drift</span><strong>${opPlus.novelty_score} (${opPlus.drift_warning ? "Warning" : "Normal"})</strong></div>
        <div class="p3-row"><span>Predicted Recovery</span><strong>${p3.recovery_operating_plus_image ?? '<span class="muted">Pending Project 3</span>'}</strong></div>
      </div>
    </div>
    <details class="p3-details">
      <summary>Developer API Payload &amp; Integration Requirements</summary>
      <ul>${(p3.missing_requirements || []).map((m) => `<li>${m}</li>`).join("")}</ul>
      <pre class="json-box">${JSON.stringify({ operating_only_payload: opOnly, operating_plus_image_payload: opPlus }, null, 2)}</pre>
    </details>
  `;
}

async function predictLocalSample(sample) {
  try {
    const res = await fetch("/api/predict-local", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: sample.image_id }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Prediction failed");
    renderPrediction(data, sample.image_url);
  } catch (err) {
    showError(err.message);
  }
}

async function uploadAndPredict() {
  const input = $("fileInput");
  if (!input.files || !input.files[0]) {
    showError("Please choose an image file first.");
    return;
  }
  const file = input.files[0];
  const form = new FormData();
  form.append("file", file);
  const localUrl = URL.createObjectURL(file);
  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Prediction failed");
    renderPrediction(data, localUrl);
  } catch (err) {
    showError(err.message);
  }
}

async function runBatchAnalyze() {
  const batchId = $("runBatchSelect").value;
  const limit = Number($("runBatchLimit").value || 45);
  $("batchStatus").textContent = `Processing ${limit} images sequentially from ${batchId} and writing CSV to disk...`;
  $("runBatchBtn").disabled = true;
  try {
    const res = await fetch("/api/batch-analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ batch_id: batchId, limit }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Batch run failed");

    $("batchStatus").innerHTML = `Completed <strong>${data.processed_count}</strong> images in ${data.elapsed_sec}s. Saved to <code>${data.output_csv_relative_path}</code>.`;
    $("batchSummaryKpis").style.display = "grid";
    $("batchSummaryKpis").innerHTML = `
      <div class="kpi"><span class="kpi-label">RF Accuracy</span><strong>${(data.rf_accuracy_on_run * 100).toFixed(1)}%</strong></div>
      <div class="kpi"><span class="kpi-label">CNN Accuracy</span><strong>${(data.cnn_accuracy_on_run * 100).toFixed(1)}%</strong></div>
      <div class="kpi"><span class="kpi-label">Predicted Unstable</span><strong>${data.predicted_unstable_count} / ${data.processed_count}</strong></div>
      <div class="kpi"><span class="kpi-label">Drift Warnings</span><strong>${data.drift_warning_count}</strong></div>
    `;
    const table = $("batchResultsTable");
    table.style.display = "table";
    table.querySelector("tbody").innerHTML = (data.preview_rows || []).map((r) => `
      <tr>
        <td>${r.image_id}</td>
        <td>${r.true_category}</td>
        <td>${r.true_stability}</td>
        <td>${r.rf_category} (${(r.rf_confidence * 100).toFixed(0)}%)</td>
        <td>${r.cnn_category} (${(r.cnn_confidence * 100).toFixed(0)}%)</td>
        <td>${r.predicted_stability} (${(r.stability_score * 100).toFixed(0)}%)</td>
        <td>${r.estimated_air_flow_l_min}</td>
        <td>${r.bubble_d50 ?? "-"}</td>
        <td>${r.novelty_score}</td>
        <td>${r.drift_warning ? "WARNING" : "OK"}</td>
      </tr>
    `).join("");
  } catch (err) {
    $("batchStatus").textContent = "Error: " + err.message;
  } finally {
    $("runBatchBtn").disabled = false;
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  populateCategories();
  $("refreshSamplesBtn").addEventListener("click", loadSamples);
  $("uploadBtn").addEventListener("click", uploadAndPredict);
  $("runBatchBtn").addEventListener("click", runBatchAnalyze);
  await Promise.all([loadSummary(), loadSamples()]);
});
