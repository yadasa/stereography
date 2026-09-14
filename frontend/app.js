const WORKER_URL = "http://127.0.0.1:8765";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const els = {
  workerPill: $("#workerPill"),
  workerLabel: $("#workerLabel"),
  reconnect: $("#reconnectButton"),
  input: $("#mediaInput"),
  dropzone: $("#dropzone"),
  dropTitle: $("#dropTitle"),
  dropMeta: $("#dropMeta"),
  sourcePreview: $("#sourcePreview"),
  sourceVideo: $("#sourceVideo"),
  sourceImage: $("#sourceImage"),
  fileName: $("#fileName"),
  fileDetails: $("#fileDetails"),
  replace: $("#replaceButton"),
  compare: $("#compareButton"),
  reset: $("#resetButton"),
  process: $("#processButton"),
  processLabel: $("#processLabel"),
  resolution: $("#resolution"),
  emptyPreview: $("#emptyPreview"),
  renderPreview: $("#renderPreview"),
  outputVideo: $("#outputVideo"),
  outputImage: $("#outputImage"),
  download: $("#downloadButton"),
  progress: $("#jobProgress"),
  progressLabel: $("#progressLabel"),
  progressValue: $("#progressValue"),
  progressBar: $("#progressBar"),
  progressDetail: $("#progressDetail"),
  comparison: $("#comparison"),
  daImage: $("#depthAnythingImage"),
  midasImage: $("#midasImage"),
  daTime: $("#depthAnythingTime"),
  midasTime: $("#midasTime"),
  comfortBar: $("#comfortBar"),
  comfortLabel: $("#comfortLabel"),
  smoothingControl: $("#smoothingControl"),
  gaussianControl: $("#gaussianControl"),
  toast: $("#toast"),
};

const controls = {
  eyeSeparation: $("#eyeSeparation"),
  depthStrength: $("#depthStrength"),
  convergence: $("#convergence"),
  smoothing: $("#smoothing"),
  gaussianScale: $("#gaussianScale"),
};

const defaults = {
  eyeSeparation: 28,
  depthStrength: 1,
  convergence: 50,
  smoothing: 70,
  gaussianScale: 1.35,
  resolution: "720p",
  model: "depth-anything-v2-small",
  renderMethod: "depth-warp",
};

let state = {
  file: null,
  sourceUrl: null,
  outputUrl: null,
  workerOnline: false,
  processing: false,
  mediaKind: null,
  model: defaults.model,
  renderMethod: defaults.renderMethod,
};

function toast(message, error = false) {
  els.toast.textContent = message;
  els.toast.classList.toggle("error", error);
  els.toast.classList.add("show");
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => els.toast.classList.remove("show"), 4200);
}

function readableBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function readableTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return minutes ? `${minutes}:${String(remainder).padStart(2, "0")}` : `${remainder}s`;
}

function updateActions() {
  const ready = Boolean(state.file && state.workerOnline && !state.processing);
  els.process.disabled = !ready;
  els.compare.disabled = !ready;
}

async function checkWorker(announce = false) {
  els.workerPill.dataset.state = "checking";
  els.workerLabel.textContent = "Checking local worker…";
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 2500);
  try {
    const response = await fetch(`${WORKER_URL}/api/health`, { cache: "no-store", signal: controller.signal });
    if (!response.ok) throw new Error(`Worker returned ${response.status}`);
    const health = await response.json();
    state.workerOnline = true;
    els.workerPill.dataset.state = "online";
    els.workerLabel.textContent = health.cuda ? `GPU ready · ${health.device}` : "Worker ready · CPU mode";
    if (announce) toast(health.cuda ? `Connected to ${health.device}` : "Connected. CUDA was not found, so processing will use CPU.");
  } catch {
    state.workerOnline = false;
    els.workerPill.dataset.state = "offline";
    els.workerLabel.textContent = "Local worker offline";
    if (announce) toast("Start the local worker, then reconnect. Your video has not left this browser.", true);
  } finally {
    window.clearTimeout(timeout);
    updateActions();
  }
}

function updateRange(input) {
  const min = Number(input.min || 0);
  const max = Number(input.max || 100);
  const value = Number(input.value);
  input.style.setProperty("--fill", `${((value - min) / (max - min)) * 100}%`);
  const output = $(`#${input.id}Out`);
  if (input.id === "eyeSeparation") output.value = `${value} px`;
  if (input.id === "depthStrength") output.value = `${value.toFixed(2)}×`;
  if (input.id === "gaussianScale") output.value = `${value.toFixed(2)} px`;
  if (input.id === "convergence" || input.id === "smoothing") output.value = `${value}%`;
  updateComfort();
}

function updateComfort() {
  const parallax = Number(controls.eyeSeparation.value) * Number(controls.depthStrength.value);
  const score = Math.max(4, Math.min(100, 100 - Math.max(0, parallax - 22) * 1.5));
  els.comfortBar.style.width = `${score}%`;
  if (score > 70) {
    els.comfortLabel.value = "Comfortable";
    els.comfortLabel.style.color = "#bff6dc";
  } else if (score > 42) {
    els.comfortLabel.value = "Review edges";
    els.comfortLabel.style.color = "#f4dc91";
  } else {
    els.comfortLabel.value = "High parallax";
    els.comfortLabel.style.color = "#ff9da6";
  }
}

function resetControls() {
  Object.entries(controls).forEach(([key, input]) => {
    input.value = defaults[key];
    updateRange(input);
  });
  els.resolution.value = defaults.resolution;
  state.model = defaults.model;
  state.renderMethod = defaults.renderMethod;
  $$(".segmented").forEach((group) => {
    const key = group.dataset.control;
    group.querySelectorAll("button").forEach((button) => button.classList.toggle("active", button.dataset.value === state[key]));
  });
  updateMethodUI();
}

function updateMethodUI() {
  const gaussianActive = state.renderMethod === "gaussian-4d";
  controls.gaussianScale.disabled = !gaussianActive;
  els.gaussianControl.classList.toggle("inactive", !gaussianActive);
  els.gaussianControl.title = gaussianActive ? "Projected Gaussian footprint in output pixels." : "Available in 4D Gaussian Lite mode.";
}

function loadFile(file) {
  if (!file) return;
  const mediaKind = file.type.startsWith("video/") ? "video" : file.type.startsWith("image/") ? "image" : null;
  if (!mediaKind) return toast("Choose an MP4, MOV, WebM, JPG, PNG, or WebP file.", true);
  if (file.size > 512 * 1024 * 1024) return toast("Keep this first workflow under 512 MB.", true);
  if (state.sourceUrl) URL.revokeObjectURL(state.sourceUrl);
  state.file = file;
  state.mediaKind = mediaKind;
  state.sourceUrl = URL.createObjectURL(file);
  els.dropzone.classList.add("hidden");
  els.sourcePreview.classList.remove("hidden");
  els.fileName.textContent = file.name;
  els.fileDetails.textContent = readableBytes(file.size);
  els.sourceVideo.classList.toggle("hidden", mediaKind !== "video");
  els.sourceImage.classList.toggle("hidden", mediaKind !== "image");
  els.processLabel.textContent = mediaKind === "image" ? "Generate stereo image" : "Generate stereo preview";
  controls.smoothing.disabled = mediaKind === "image";
  els.smoothingControl.classList.toggle("inactive", mediaKind === "image");
  els.smoothingControl.title = mediaKind === "image" ? "Temporal smoothing applies only to video." : "";

  if (mediaKind === "video") {
    els.sourceImage.removeAttribute("src");
    els.sourceVideo.src = state.sourceUrl;
    els.sourceVideo.load();
    els.sourceVideo.onloadedmetadata = () => {
      if (els.sourceVideo.duration > 90) {
        toast("This local-first version accepts clips up to 90 seconds. Trim the source and try again.", true);
        clearFile();
        return;
      }
      els.fileDetails.textContent = `${readableBytes(file.size)} · ${readableTime(els.sourceVideo.duration)} · ${els.sourceVideo.videoWidth}×${els.sourceVideo.videoHeight}`;
    };
  } else {
    els.sourceVideo.pause();
    els.sourceVideo.removeAttribute("src");
    els.sourceImage.src = state.sourceUrl;
    els.sourceImage.onload = () => {
      const pixels = els.sourceImage.naturalWidth * els.sourceImage.naturalHeight;
      if (pixels > 50_000_000) {
        toast("Keep still images under 50 megapixels for this workflow.", true);
        clearFile();
        return;
      }
      els.fileDetails.textContent = `${readableBytes(file.size)} · STILL · ${els.sourceImage.naturalWidth}×${els.sourceImage.naturalHeight}`;
    };
  }
  updateActions();
}

function clearFile() {
  state.file = null;
  state.mediaKind = null;
  if (state.sourceUrl) URL.revokeObjectURL(state.sourceUrl);
  state.sourceUrl = null;
  els.sourceVideo.removeAttribute("src");
  els.sourceImage.removeAttribute("src");
  els.sourceVideo.classList.remove("hidden");
  els.sourceImage.classList.add("hidden");
  els.sourcePreview.classList.add("hidden");
  els.dropzone.classList.remove("hidden");
  els.input.value = "";
  els.processLabel.textContent = "Generate stereo preview";
  controls.smoothing.disabled = false;
  els.smoothingControl.classList.remove("inactive");
  updateActions();
}

function settingsForm() {
  const data = new FormData();
  data.append("media", state.file, state.file.name);
  data.append("model", state.model);
  data.append("render_method", state.renderMethod);
  data.append("eye_separation", controls.eyeSeparation.value);
  data.append("depth_strength", controls.depthStrength.value);
  data.append("convergence", String(Number(controls.convergence.value) / 100));
  data.append("temporal_smoothing", String(Number(controls.smoothing.value) / 100));
  data.append("gaussian_scale", controls.gaussianScale.value);
  data.append("resolution", els.resolution.value);
  return data;
}

async function pollJob(jobId) {
  while (state.processing) {
    const response = await fetch(`${WORKER_URL}/api/jobs/${jobId}`, { cache: "no-store" });
    if (!response.ok) throw new Error("The local worker lost the render job.");
    const job = await response.json();
    const progress = Math.round(job.progress * 100);
    els.progressValue.textContent = `${progress}%`;
    els.progressBar.style.width = `${progress}%`;
    els.progressLabel.textContent = job.stage || "Processing…";
    els.progressDetail.textContent = job.detail || "Depth inference and stereo synthesis are running locally.";
    if (job.status === "failed") throw new Error(job.error || "Render failed.");
    if (job.status === "complete") return job;
    await new Promise((resolve) => setTimeout(resolve, 900));
  }
  throw new Error("Render cancelled.");
}

async function processVideo() {
  if (!state.file || !state.workerOnline || state.processing) return;
  state.processing = true;
  updateActions();
  els.emptyPreview.classList.add("hidden");
  els.renderPreview.classList.add("hidden");
  els.progress.classList.remove("hidden");
  els.progressBar.style.width = "2%";
  els.progressValue.textContent = "2%";
  els.progressLabel.textContent = "Sending to your local GPU…";
  els.progressDetail.textContent = "The browser is talking only to 127.0.0.1.";
  document.querySelector("#previewSection").scrollIntoView({ behavior: "smooth", block: "start" });

  try {
    const response = await fetch(`${WORKER_URL}/api/jobs`, { method: "POST", body: settingsForm() });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Worker returned ${response.status}`);
    }
    const { id } = await response.json();
    const result = await pollJob(id);
    if (state.outputUrl) URL.revokeObjectURL(state.outputUrl);
    const output = await fetch(`${WORKER_URL}/api/jobs/${id}/output`);
    if (!output.ok) throw new Error("The render finished, but the output could not be loaded.");
    state.outputUrl = URL.createObjectURL(await output.blob());
    const outputKind = result.output_kind || state.mediaKind;
    els.outputVideo.classList.toggle("hidden", outputKind !== "video");
    els.outputImage.classList.toggle("hidden", outputKind !== "image");
    if (outputKind === "image") {
      els.outputVideo.pause();
      els.outputVideo.removeAttribute("src");
      els.outputImage.src = state.outputUrl;
      els.download.download = "stereo-sbs.png";
    } else {
      els.outputImage.removeAttribute("src");
      els.outputVideo.src = state.outputUrl;
      els.download.download = "stereo-sbs.mp4";
    }
    els.download.href = state.outputUrl;
    els.download.classList.remove("disabled");
    els.progress.classList.add("hidden");
    els.renderPreview.classList.remove("hidden");
    if (outputKind === "video") els.outputVideo.play().catch(() => {});
    const gaussianDetail = result.gaussian_mode && state.mediaKind === "video"
      ? ` · ${result.average_temporal_reuse_pct.toFixed(0)}% temporal reuse`
      : "";
    toast(`Stereo render complete in ${result.elapsed_seconds.toFixed(1)}s${gaussianDetail}.`);
  } catch (error) {
    els.progress.classList.add("hidden");
    els.emptyPreview.classList.remove("hidden");
    toast(error.message || "Render failed.", true);
    await checkWorker();
  } finally {
    state.processing = false;
    updateActions();
  }
}

async function captureFrame() {
  if (state.mediaKind === "image") {
    const image = els.sourceImage;
    if (!image.naturalWidth) throw new Error("Wait for the image preview to load.");
    const canvas = document.createElement("canvas");
    const scale = Math.min(1, 960 / image.naturalWidth);
    canvas.width = Math.round(image.naturalWidth * scale);
    canvas.height = Math.round(image.naturalHeight * scale);
    canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Could not sample this image.")), "image/jpeg", 0.9));
  }
  const video = els.sourceVideo;
  if (!video.videoWidth) throw new Error("Wait for the video preview to load.");
  if (video.readyState < 2) await new Promise((resolve) => video.addEventListener("loadeddata", resolve, { once: true }));
  const canvas = document.createElement("canvas");
  const scale = Math.min(1, 960 / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Could not sample this frame.")), "image/jpeg", 0.9));
}

async function compareModels() {
  if (!state.file || !state.workerOnline || state.processing) return;
  const originalText = els.compare.firstElementChild.textContent;
  els.compare.disabled = true;
  els.compare.firstElementChild.textContent = "Comparing on local GPU…";
  try {
    const frame = await captureFrame();
    const data = new FormData();
    data.append("frame", frame, "comparison-frame.jpg");
    const response = await fetch(`${WORKER_URL}/api/compare`, { method: "POST", body: data });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || "Model comparison failed.");
    }
    const result = await response.json();
    els.daImage.src = `data:image/png;base64,${result["depth-anything-v2-small"].image}`;
    els.midasImage.src = `data:image/png;base64,${result["midas-hybrid"].image}`;
    els.daTime.textContent = `${result["depth-anything-v2-small"].milliseconds.toFixed(0)} ms · ${result["depth-anything-v2-small"].device}`;
    els.midasTime.textContent = `${result["midas-hybrid"].milliseconds.toFixed(0)} ms · ${result["midas-hybrid"].device}`;
    els.comparison.classList.remove("hidden");
    els.comparison.scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (error) {
    toast(error.message || "Model comparison failed.", true);
  } finally {
    els.compare.disabled = false;
    els.compare.firstElementChild.textContent = originalText;
  }
}

Object.values(controls).forEach((input) => {
  input.addEventListener("input", () => updateRange(input));
  updateRange(input);
});

$$('.segmented').forEach((group) => {
  group.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    group.querySelectorAll("button").forEach((item) => item.classList.toggle("active", item === button));
    state[group.dataset.control] = button.dataset.value;
    updateMethodUI();
    if (["point-cloud", "gaussian-4d"].includes(state.renderMethod) && els.resolution.value === "1080p") toast("This renderer is substantially heavier at 1080p. Start with 720p for calibration.");
  });
});

els.input.addEventListener("change", () => loadFile(els.input.files[0]));
els.replace.addEventListener("click", () => els.input.click());
els.reconnect.addEventListener("click", () => checkWorker(true));
els.reset.addEventListener("click", resetControls);
els.process.addEventListener("click", processVideo);
els.compare.addEventListener("click", compareModels);
els.resolution.addEventListener("change", () => {
  if (["point-cloud", "gaussian-4d"].includes(state.renderMethod) && els.resolution.value === "1080p") toast("Use 720p to calibrate this renderer before a final 1080p render.");
});

["dragenter", "dragover"].forEach((name) => els.dropzone.addEventListener(name, (event) => {
  event.preventDefault();
  els.dropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => els.dropzone.addEventListener(name, (event) => {
  event.preventDefault();
  els.dropzone.classList.remove("dragging");
}));
els.dropzone.addEventListener("drop", (event) => loadFile(event.dataTransfer.files[0]));

window.addEventListener("beforeunload", () => {
  if (state.sourceUrl) URL.revokeObjectURL(state.sourceUrl);
  if (state.outputUrl) URL.revokeObjectURL(state.outputUrl);
});

resetControls();
if (["localhost", "127.0.0.1"].includes(window.location.hostname)) {
  $("#connectionNote").textContent = "Fully local mode: both the interface and GPU worker are running on this machine.";
}
checkWorker();
