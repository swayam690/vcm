const $ = (selector) => document.querySelector(selector);

const videoSelect = $("#videoSelect");
const queryInput = $("#queryInput");
const searchButton = $("#searchButton");
const progressBox = $("#progressBox");
const errorBox = $("#errorBox");
const resultsSection = $("#resultsSection");
const frameGrid = $("#frameGrid");
const frameBudget = $("#selectedBudget");
const sortSelect = $("#sortSelect");
const cancelButton = $("#cancelButton");

let currentResult = null;
let currentPage = 0;
let currentJobId = null;
const PAGE_SIZE = 50;

function formatDuration(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return "duration unavailable";
  const total = Math.max(0, Math.round(Number(seconds)));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

function money(value) {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "$0.000000";
  if (value < 0.0001) return `$${value.toFixed(8)}`;
  if (value < 0.01) return `$${value.toFixed(5)}`;
  return `$${value.toFixed(4)}`;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

async function loadApp() {
  try {
    const [configResponse, videosResponse] = await Promise.all([
      fetch("api/config"),
      fetch("api/videos"),
    ]);
    if (!configResponse.ok) throw new Error("Could not read backend configuration.");
    if (!videosResponse.ok) throw new Error("Could not scan the video folder.");

    const config = await configResponse.json();
    const data = await videosResponse.json();
    $("#runtimeText").textContent = `${config.device_label || config.device} · ready`;
    $(".status-dot").classList.add("online");
    const modelId = String(config.siglip_model || "SigLIP2");
    const modelSuffix = modelId.split(/[\\/]/).pop();
    const modelLabel = modelId.toLowerCase().endsWith(".safetensors")
      ? "SigLIP2 · fine-tuned"
      : modelId.toLowerCase().includes("siglip2-base-patch16-224")
        ? "SigLIP2 · Base/224"
        : `SigLIP2 · ${modelSuffix}`;
    $("#modelTag").textContent = modelLabel;
    $("#modelTag").title = modelId;
    const geminiStatus = $("#geminiStatus");
    if (config.gemini_configured) {
      geminiStatus.textContent = `Gemini connected · ${config.gemini_model}`;
      geminiStatus.classList.remove("offline");
    } else {
      geminiStatus.textContent = "Gemini not configured · original query used";
      geminiStatus.classList.add("offline");
    }

    videoSelect.replaceChildren();
    if (!data.videos.length) {
      const option = document.createElement("option");
      option.textContent = "No videos found — set VIDEO_DIR in .env";
      option.value = "";
      videoSelect.append(option);
      $("#videoMeta").textContent = "Set VIDEO_DIR to your downloaded video folder, then refresh this page.";
      return;
    }

    const placeholder = document.createElement("option");
      placeholder.textContent = `Choose from ${data.videos.length} videos`;
    placeholder.value = "";
    placeholder.disabled = true;
    placeholder.selected = true;
    videoSelect.append(placeholder);

    const nameCounts = new Map();
    for (const video of data.videos) nameCounts.set(video.name, (nameCounts.get(video.name) || 0) + 1);
    for (const video of data.videos) {
      const option = document.createElement("option");
      option.value = video.id;
      const displayName = nameCounts.get(video.name) > 1 ? video.label : video.name;
      option.textContent = `${displayName} · ${formatDuration(video.duration_seconds)}`;
      option.dataset.name = video.name;
      option.dataset.duration = video.duration_seconds ?? "";
      videoSelect.append(option);
    }
    $("#videoMeta").textContent = `${data.videos.length} videos available.`;

    const resumeJobId = new URLSearchParams(window.location.search).get("job_id");
    let jobToResume = null;
    if (resumeJobId) {
      const response = await fetch(`api/jobs/${encodeURIComponent(resumeJobId)}`);
      if (response.ok) jobToResume = await response.json();
      else history.replaceState({}, "", window.location.pathname);
    }
    if (!jobToResume) {
      const activeResponse = await fetch("api/jobs/active");
      if (activeResponse.ok) {
        const active = await activeResponse.json();
        jobToResume = active.job;
      }
    }
    if (jobToResume) {
      currentJobId = jobToResume.job_id;
      populateSearchFromJob(jobToResume);
      searchButton.disabled = true;
      progressBox.classList.remove("hidden");
      setJobMeta(jobToResume);
      updateProgress(jobToResume.phase || "Reconnecting to search", jobToResume.progress, jobToResume.progress_label || "Restoring this search's live progress.");
      setCancelState(jobToResume.status);
      try {
        await pollJob(jobToResume.job_id);
      } catch (error) {
        showError(error.message);
        progressBox.classList.add("hidden");
      } finally {
        searchButton.disabled = false;
      }
    }
  } catch (error) {
    $("#runtimeText").textContent = "backend unavailable";
    $(".status-dot").classList.remove("online");
    showError(`${error.message}\nStart the Python backend, then refresh this page.`);
  }
}

function setJobMeta(job) {
  const parts = [job.video_name, job.query].filter(Boolean);
  $("#progressJobMeta").textContent = parts.join(" · ");
}

function populateSearchFromJob(job) {
  if (job.video_id) {
    videoSelect.value = job.video_id;
    videoSelect.dispatchEvent(new Event("change"));
  }
  if (job.query) queryInput.value = job.query;
}

function setCancelState(status) {
  const active = ["queued", "running", "cancelling"].includes(status);
  const cancelling = status === "cancelling";
  cancelButton.classList.toggle("hidden", !active);
  cancelButton.disabled = cancelling;
  cancelButton.textContent = cancelling ? "Cancelling…" : "Cancel search";
}

videoSelect.addEventListener("change", () => {
  const option = videoSelect.selectedOptions[0];
  if (option && option.value) {
    const duration = option.dataset.duration ? formatDuration(option.dataset.duration) : "duration unavailable";
    $("#videoMeta").textContent = `${option.dataset.name} · ${duration}`;
  }
});

async function startSearch() {
  errorBox.classList.add("hidden");
  const query = queryInput.value.trim();
  const videoId = videoSelect.value;
  const selectedFrameCount = Math.max(1, Math.min(200, Number(frameBudget.value) || 10));
  frameBudget.value = selectedFrameCount;

  if (!videoId) return showError("Select a video first.");
  if (!query) return showError("Type a query describing what you want to find.");

  currentResult = null;
  currentJobId = null;
  resultsSection.classList.add("hidden");
  $("#exportStatus").classList.add("hidden");
  searchButton.disabled = true;
  progressBox.classList.remove("hidden");
  updateProgress("Starting search", 0, "Sending query to backend");
  setJobMeta({
    video_name: videoSelect.selectedOptions[0]?.dataset.name,
    query,
  });
  cancelButton.classList.add("hidden");

  try {
    const response = await fetch("api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId, query, selected_frame_count: selectedFrameCount }),
    });
    const body = await response.json();
    if (!response.ok && response.status === 409) {
      const activeResponse = await fetch("api/jobs/active");
      const active = activeResponse.ok ? await activeResponse.json() : null;
      if (active?.job) {
        currentJobId = active.job.job_id;
        populateSearchFromJob(active.job);
        setJobMeta(active.job);
        updateProgress(active.job.phase, active.job.progress, active.job.progress_label || "Reconnected to the active search.");
        setCancelState(active.job.status);
        await pollJob(currentJobId);
        return;
      }
    }
    if (!response.ok) throw new Error(body.detail || "Search could not be started.");
    currentJobId = body.job_id;
    setCancelState("queued");
    await pollJob(currentJobId);
  } catch (error) {
    showError(error.message);
    progressBox.classList.add("hidden");
    cancelButton.classList.add("hidden");
  } finally {
    searchButton.disabled = false;
  }
}

function updateProgress(phase, fraction, label) {
  const pct = Math.round(Math.max(0, Math.min(1, fraction || 0)) * 100);
  $("#progressPhase").textContent = phase || "Working";
  $("#progressPercent").textContent = `${pct}%`;
  $("#progressBar").style.width = `${pct}%`;
  const displayLabel = (label || "Working…").replace(
    /^Encoded through ([\d,]+) \/ ([\d,]+) sampled seconds$/,
    "Scanned $1 / $2 frames; SigLIP2 embeds them in batches."
  );
  $("#progressLabel").textContent = displayLabel;
}

async function pollJob(jobId) {
  while (true) {
    const response = await fetch(`api/jobs/${encodeURIComponent(jobId)}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || "Could not read search progress.");
    setJobMeta(job);
    updateProgress(job.phase, job.progress, job.progress_label || job.status);
    setCancelState(job.status);

    if (job.status === "done") {
      progressBox.classList.add("hidden");
      currentResult = job.result;
      currentPage = 0;
      renderResults();
      return;
    }
    if (job.status === "error") {
      progressBox.classList.add("hidden");
      throw new Error(job.error || "Search failed.");
    }
    if (job.status === "cancelled") {
      cancelButton.classList.add("hidden");
      updateProgress("Cancelled", job.progress, job.progress_label || "Search cancelled. You can start another search.");
      history.replaceState({}, "", window.location.pathname);
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 900));
  }
}

async function cancelSearch() {
  if (!currentJobId) return;
  cancelButton.disabled = true;
  cancelButton.textContent = "Cancelling…";
  try {
    const response = await fetch(`api/jobs/${encodeURIComponent(currentJobId)}`, { method: "DELETE" });
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || "Could not cancel the search.");
    if (job.status === "done") {
      cancelButton.classList.add("hidden");
      return;
    }
    setCancelState(job.status);
    updateProgress(job.phase, job.progress, job.progress_label || "Cancellation requested.");
  } catch (error) {
    showError(error.message);
    cancelButton.disabled = false;
    cancelButton.textContent = "Cancel search";
  }
}

function renderResults() {
  const result = currentResult;
  if (!result) return;
  resultsSection.classList.remove("hidden");
  $("#resultsTitle").textContent = result.video.name;
  updateResultCount();

  const phraseList = $("#searchPhrases");
  phraseList.replaceChildren();
  for (const phrase of result.queries) {
    const chip = document.createElement("span");
    chip.className = "phrase-chip";
    chip.textContent = phrase;
    phraseList.append(chip);
  }

  const note = $("#searchNote");
  if (result.note) {
    note.textContent = result.note;
    note.classList.remove("hidden");
  } else {
    note.classList.add("hidden");
  }

  const best = result.top_frames[0];
  $("#bestFrameBadge").textContent = best
    ? `Top match · ${formatDuration(best.second)} · ${best.score.toFixed(4)}`
    : "No frame match";

  drawChart();
  renderGallery();
  renderCost();
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function updateResultCount() {
  if (!currentResult) return;
  const selected = Math.min(currentResult.frame_count, Math.max(1, Number(frameBudget.value) || 10));
  $("#resultCount").textContent = `${formatDuration(currentResult.duration_seconds)} · ${currentResult.frame_count.toLocaleString()} sampled · ${selected} selected`;
}

function drawChart() {
  if (!currentResult) return;
  const canvas = $("#scoreChart");
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const width = rect.width;
  const height = rect.height;
  const left = 52, right = 17, top = 15, bottom = 34;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const frames = currentResult.frames;
  const scores = frames.map((frame) => frame.score);
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  const padding = Math.max((maxScore - minScore) * 0.08, 0.002);
  const yMin = minScore - padding;
  const yMax = maxScore + padding;
  const maxSecond = Math.max(1, ...frames.map((frame) => frame.second));
  const x = (second) => left + (second / maxSecond) * plotW;
  const y = (score) => top + (1 - (score - yMin) / (yMax - yMin)) * plotH;

  ctx.clearRect(0, 0, width, height);
  ctx.font = "10px 'DM Mono', monospace";
  ctx.textBaseline = "middle";
  ctx.strokeStyle = "#e2e7ed";
  ctx.fillStyle = "#8291a1";
  ctx.lineWidth = 1;

  for (let i = 0; i <= 4; i++) {
    const yy = top + (plotH * i) / 4;
    const score = yMax - ((yMax - yMin) * i) / 4;
    ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(width - right, yy); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(score.toFixed(3), left - 8, yy);
  }
  const tickCount = Math.min(6, Math.max(2, Math.floor(maxSecond / 60) + 1));
  for (let i = 0; i <= tickCount; i++) {
    const second = (maxSecond * i) / tickCount;
    const xx = x(second);
    ctx.beginPath(); ctx.moveTo(xx, top); ctx.lineTo(xx, top + plotH); ctx.stroke();
    ctx.textAlign = "center"; ctx.fillText(formatDuration(second), xx, top + plotH + 17);
  }

  ctx.strokeStyle = "#315e83";
  ctx.lineWidth = 1.55;
  ctx.beginPath();
  frames.forEach((frame, index) => {
    const xx = x(frame.second), yy = y(frame.score);
    if (index === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
  });
  ctx.stroke();

  const selected = Math.max(1, Math.min(200, Number(frameBudget.value) || 10));
  $("#downloadSelected").textContent = `Export ${selected} selected frames`;
  ctx.fillStyle = "rgba(40, 122, 120, .76)";
  for (const frame of frames) {
    if (frame.rank <= selected) {
      ctx.beginPath(); ctx.arc(x(frame.second), y(frame.score), 2.8, 0, Math.PI * 2); ctx.fill();
    }
  }
  const best = frames.reduce((current, frame) => frame.score > current.score ? frame : current, frames[0]);
  if (best) {
    ctx.fillStyle = "#ae674f";
    ctx.beginPath(); ctx.arc(x(best.second), y(best.score), 5, 0, Math.PI * 2); ctx.fill();
  }

  canvas._chartScale = { left, right, plotW, maxSecond, frames };
}

function renderGallery() {
  if (!currentResult) return;
  const sortMode = sortSelect.value;
  const ordered = [...currentResult.frames];
  if (sortMode === "score") ordered.sort((a, b) => b.score - a.score);
  else ordered.sort((a, b) => a.second - b.second);

  const pageCount = Math.max(1, Math.ceil(ordered.length / PAGE_SIZE));
  currentPage = Math.max(0, Math.min(currentPage, pageCount - 1));
  const start = currentPage * PAGE_SIZE;
  const pageItems = ordered.slice(start, start + PAGE_SIZE);
  const selected = Math.max(1, Math.min(200, Number(frameBudget.value) || 10));
  const selectedCount = Math.min(selected, currentResult.frame_count);
  $("#selectedFrameCount").textContent = `${selectedCount} of ${currentResult.frame_count.toLocaleString()} selected`;

  frameGrid.replaceChildren();
  for (const frame of pageItems) {
    const card = document.createElement("article");
    card.className = `frame-card${frame.rank <= selected ? " is-selected" : ""}`;
    const wrap = document.createElement("div");
    wrap.className = "frame-image-wrap";
    const image = document.createElement("img");
    image.loading = "lazy";
    image.decoding = "async";
    image.alt = `Frame at ${frame.second.toFixed(0)} seconds`;
    image.src = `api/frame/${encodeURIComponent(currentResult.video.id)}/${Math.floor(frame.second)}`;
    const loading = document.createElement("span");
    loading.className = "img-loading";
    loading.textContent = "Loading frame…";
    image.addEventListener("load", () => loading.remove(), { once: true });
    image.addEventListener("error", () => { loading.textContent = "Preview unavailable"; }, { once: true });
    wrap.append(image, loading);
    if (frame.rank <= selected) {
      const mark = document.createElement("span");
      mark.className = "selected-mark";
      mark.textContent = `SELECTED · #${frame.rank}`;
      wrap.append(mark);
    }
    const data = document.createElement("div");
    data.className = "frame-data";
    const time = document.createElement("span");
    time.textContent = formatDuration(frame.second);
    const score = document.createElement("span");
    score.className = "frame-rank";
    score.textContent = `#${frame.rank} · ${frame.score.toFixed(4)}`;
    data.append(time, score);
    card.append(wrap, data);
    frameGrid.append(card);
  }

  $("#pageInfo").textContent = `Page ${currentPage + 1} of ${pageCount} · ${ordered.length.toLocaleString()} frames`;
  $("#prevPage").disabled = currentPage === 0;
  $("#nextPage").disabled = currentPage >= pageCount - 1;
}

function renderCost() {
  if (!currentResult) return;
  const totalFrames = currentResult.frame_count;
  const keepFrames = Math.min(totalFrames, Math.max(1, Number(frameBudget.value) || 10));
  const imageTokens = Math.max(1, Number($("#imageTokens").value) || 258);
  const inputRate = Math.max(0, Number($("#inputRate").value) || 0);
  const outputRate = Math.max(0, Number($("#outputRate").value) || 0);
  const framesPerRequest = Math.max(1, Number($("#framesPerRequest").value) || 20);
  const promptTokens = Math.max(0, Number($("#promptTokens").value) || 0);
  const answerTokens = Math.max(0, Number($("#answerTokens").value) || 0);
  const requestCost = (n) => {
    const calls = Math.ceil(n / framesPerRequest);
    return {
      calls,
      cost: ((n * imageTokens + calls * promptTokens) * inputRate + calls * answerTokens * outputRate) / 1_000_000,
    };
  };

  const fullEstimate = requestCost(totalFrames);
  const selectedEstimate = requestCost(keepFrames);
  const full = fullEstimate.cost;
  const selected = selectedEstimate.cost;
  const savings = full - selected;
  const percent = full > 0 ? (savings / full) * 100 : 0;

  $("#fullCost").textContent = money(full);
  $("#selectedCost").textContent = money(selected);
  $("#savingPercent").textContent = `${percent.toFixed(1)}%`;
  $("#savingAmount").textContent = `${money(savings)} saved per query`;
  $("#fullCostMeta").textContent = `${totalFrames.toLocaleString()} frames · ${fullEstimate.calls} request(s) · ${(totalFrames * imageTokens).toLocaleString()} image tokens`;
  $("#selectedCostMeta").textContent = `${keepFrames.toLocaleString()} frames · ${selectedEstimate.calls} request(s) · ${(keepFrames * imageTokens).toLocaleString()} image tokens`;
  const vlmName = $("#vlmName").value.trim() || "VLM";
  $(".estimate-tag").textContent = `${vlmName.toUpperCase()} ESTIMATE`;
}

async function downloadSelectedFrames() {
  if (!currentResult) return;
  const keep = Math.min(currentResult.frame_count, Math.max(1, Number(frameBudget.value) || 10));
  const frames = currentResult.frames
    .filter((frame) => frame.rank <= keep)
    .map(({ second, rank, score }) => ({ second: Math.floor(second), rank, score }));
  const button = $("#downloadSelected");
  const status = $("#exportStatus");
  button.disabled = true;
  status.classList.remove("hidden", "error");
  status.textContent = `Preparing ${frames.length} frames and their timestamp manifest…`;

  try {
    const response = await fetch("api/export-frames", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_id: currentResult.video.id,
        query: currentResult.original_query,
        frames,
      }),
    });
    if (!response.ok) {
      const body = await response.json();
      throw new Error(body.detail || "Could not prepare the selected frames.");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = response.headers.get("content-disposition")?.match(/filename="?([^";]+)"?/i)?.[1] || "selected_frames.zip";
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    status.textContent = `Downloaded ${frames.length} frames plus manifest.csv.`;
  } catch (error) {
    status.classList.add("error");
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

searchButton.addEventListener("click", startSearch);
cancelButton.addEventListener("click", cancelSearch);
queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") startSearch();
});
$("#prevPage").addEventListener("click", () => { currentPage--; renderGallery(); });
$("#nextPage").addEventListener("click", () => { currentPage++; renderGallery(); });
sortSelect.addEventListener("change", () => { currentPage = 0; renderGallery(); });
$("#downloadSelected").addEventListener("click", downloadSelectedFrames);
frameBudget.addEventListener("input", () => {
  if (currentResult) { updateResultCount(); drawChart(); renderGallery(); renderCost(); }
});
for (const id of ["vlmName", "imageTokens", "framesPerRequest", "inputRate", "outputRate", "promptTokens", "answerTokens"]) {
  $(`#${id}`).addEventListener("input", renderCost);
}
window.addEventListener("resize", () => { if (currentResult) drawChart(); });

$("#scoreChart").addEventListener("mousemove", (event) => {
  const canvas = event.currentTarget;
  const scale = canvas._chartScale;
  if (!scale) return;
  const rect = canvas.getBoundingClientRect();
  const px = event.clientX - rect.left - scale.left;
  const second = Math.max(0, Math.min(scale.maxSecond, (px / scale.plotW) * scale.maxSecond));
  let nearest = scale.frames[0];
  for (const frame of scale.frames) {
    if (Math.abs(frame.second - second) < Math.abs(nearest.second - second)) nearest = frame;
  }
  $("#chartTooltip").textContent = `Second ${nearest.second.toFixed(0)} · score ${nearest.score.toFixed(4)} · rank #${nearest.rank}`;
});

$("#scoreChart").addEventListener("click", (event) => {
  const canvas = event.currentTarget;
  const scale = canvas._chartScale;
  if (!scale) return;
  const rect = canvas.getBoundingClientRect();
  const px = event.clientX - rect.left - scale.left;
  const second = Math.max(0, Math.min(scale.maxSecond, (px / scale.plotW) * scale.maxSecond));
  let nearestIndex = 0;
  for (let i = 1; i < scale.frames.length; i++) {
    if (Math.abs(scale.frames[i].second - second) < Math.abs(scale.frames[nearestIndex].second - second)) nearestIndex = i;
  }
  sortSelect.value = "time";
  currentPage = Math.floor(nearestIndex / PAGE_SIZE);
  renderGallery();
  $("#frameGrid").scrollIntoView({ behavior: "smooth", block: "center" });
});

loadApp();
