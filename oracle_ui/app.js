const $ = (id) => document.getElementById(id);
const form = $("source-form");
const input = $("source-id");
const modelSelect = $("model-select");
const modelPicker = $("model-picker");
const modelPickerToggle = $("model-picker-toggle");
const modelOptions = [...$("model-options").querySelectorAll(".model-option")];
const button = $("analyze-button");
const chart = $("lightcurve-chart");
const tooltip = $("chart-tooltip");
const rollingChart = $("rolling-chart");
const rollingTooltip = $("rolling-tooltip");
const modelNames = { "BTSv2-pro": "ORACLE-2 Omni", BTSv2: "ORACLE-2", "BTSv2-lite": "ORACLE-2 Lite" };
const modelDescriptions = { "BTSv2-pro": "Light curve + source context + ZTF reference image", BTSv2: "Light curve + source context", "BTSv2-lite": "Light curve only" };
const bandColors = { g: "#59d39a", r: "#ff8477", i: "#e9b66f" };
const branches = { Persistent: ["AGN", "CV", "Varstar"], Transient: ["SN-Ia", "SN-II", "SN-Ib/c", "SLSN"] };
const OOD_MAG_LIMIT = 18.5;
function extractObjectId(raw) {
  const match = String(raw).match(/ZTF\d{2}[a-z]+/i);
  return match ? match[0] : String(raw).trim();
}
async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  const glyph = button.querySelector("span");
  const original = glyph.textContent;
  button.disabled = true;
  glyph.textContent = "✓";
  setTimeout(() => { glyph.textContent = original; button.disabled = false; }, 1200);
}
let source = null;
let busy = false;
let plotted = [];
let xDomain = null;
let plotBox = null;
let drag = null;
let rolling = null;
let visibleClasses = new Set();
let rollingPlot = null;
let linkedIndex = null;
let historyDb = null;
let historyItems = [];
let activeHistoryId = null;
let currentMessage = "";
let currentMessageError = false;

function renderMessage() {
  const target = $("app-message");
  $("message-dots").hidden = !(busy && currentMessage);
  $("message-text").textContent = currentMessage;
  target.classList.toggle("error", currentMessageError);
  target.hidden = !currentMessage;
}
function message(value, error = false) {
  currentMessage = value;
  currentMessageError = error;
  renderMessage();
}
function setBusy(value) {
  busy = value;
  document.querySelector(".app-shell").classList.toggle("is-busy", value);
  button.disabled = value;
  modelSelect.disabled = value;
  modelPickerToggle.disabled = value;
  if (value) closeModelPicker();
  $("new-source").disabled = value;
  button.setAttribute("aria-label", value ? "Fetching and classifying source" : "Classify source");
  button.title = value ? "Fetching and classifying source" : "Classify source";
  button.setAttribute("aria-busy", String(value));
  button.classList.toggle("is-loading", value);
  button.firstElementChild.textContent = value ? "" : "↑";
  renderMessage();
}
function updateModelDescription() {
  const name = modelNames[modelSelect.value] || modelSelect.value;
  const isDefault = modelSelect.value === "BTSv2-pro";
  $("model-description").textContent = modelDescriptions[modelSelect.value] || "";
  $("model-picker-label").textContent = name;
  $("model-picker-default").hidden = !isDefault;
  modelPickerToggle.setAttribute("aria-label", `Model: ${name}${isDefault ? " (default)" : ""}`);
  modelOptions.forEach((option) => option.setAttribute("aria-selected", String(option.dataset.value === modelSelect.value)));
}
function closeModelPicker(restoreFocus = false) {
  $("model-options").hidden = true;
  modelPickerToggle.setAttribute("aria-expanded", "false");
  if (restoreFocus) modelPickerToggle.focus();
}
function openModelPicker() {
  if (busy) return;
  $("model-options").hidden = false;
  modelPickerToggle.setAttribute("aria-expanded", "true");
  (modelOptions.find((option) => option.dataset.value === modelSelect.value) || modelOptions[0]).focus();
}
modelPickerToggle.addEventListener("click", () => {
  if ($("model-options").hidden) openModelPicker();
  else closeModelPicker();
});
modelPickerToggle.addEventListener("keydown", (event) => {
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    event.stopPropagation();
    openModelPicker();
  }
});
modelOptions.forEach((option) => option.addEventListener("click", () => {
  if (busy) return;
  modelSelect.value = option.dataset.value;
  modelSelect.dispatchEvent(new Event("change"));
  closeModelPicker(true);
}));
modelPicker.addEventListener("keydown", (event) => {
  if ($("model-options").hidden) return;
  if (event.key === "Escape") {
    event.preventDefault();
    event.stopPropagation();
    closeModelPicker(true);
  } else if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key) && modelOptions.includes(document.activeElement)) {
    event.preventDefault();
    const current = modelOptions.indexOf(document.activeElement);
    const next = event.key === "Home" ? 0 : event.key === "End" ? modelOptions.length - 1
      : (current + (event.key === "ArrowDown" ? 1 : -1) + modelOptions.length) % modelOptions.length;
    modelOptions[next].focus();
  }
});
document.addEventListener("pointerdown", (event) => { if (!modelPicker.contains(event.target)) closeModelPicker(); });
modelPicker.addEventListener("focusout", (event) => { if (!modelPicker.contains(event.relatedTarget)) closeModelPicker(); });
modelSelect.addEventListener("change", updateModelDescription);
updateModelDescription();
fetch("/api/config")
  .then((response) => response.json())
  .then((config) => { $("babamul-warning").hidden = config.babamul_configured !== false; })
  .catch(() => {});
async function postJson(path, body) {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  let data;
  try { data = await response.json(); } catch { throw new Error("The local server returned an unreadable response."); }
  if (!response.ok) throw new Error(data.error || "The request failed.");
  return data;
}
function number(value, digits = 3) { return value !== null && value !== undefined && Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—"; }
function percent(value) {
  const n = Number(value) * 100;
  if (!Number.isFinite(n)) return "—";
  return n > 0 && n < .01 ? "<0.01%" : `${n.toFixed(n >= 10 ? 1 : 2)}%`;
}
function metadataCell(item) {
  const cell = document.createElement("div");
  cell.className = `metadata-item${item.value === null ? " missing" : ""}`;
  const label = document.createElement("span"); label.textContent = item.name;
  const value = document.createElement("strong");
  value.textContent = item.value === null ? "Unavailable" : Number(item.value).toLocaleString(undefined, { maximumFractionDigits: 4 });
  cell.append(label, value);
  return cell;
}
function renderMetadata() {
  const items = source.metadata || [];
  const available = items.filter((item) => item.value !== null);
  $("metadata-rest").replaceChildren(...items.map(metadataCell));
  const unavailable = source.classification?.missing_context_features?.length;
  $("metadata-note").textContent = `${source.metadata_error || `${available.length} of ${items.length} values available.`}${unavailable ? ` ${unavailable} contextual features were unavailable and passed as −9.` : ""}`;
}
function renderSource() {
  $("source-title").textContent = source.source_id;
  $("broker-link").href = `https://babamul.caltech.edu/objects/ZTF/${encodeURIComponent(source.source_id)}`;
  $("fritz-link").href = `https://fritz.science/source/${encodeURIComponent(source.source_id)}`;
  $("metric-detections").textContent = Number(source.detections).toLocaleString();
  $("metric-span").textContent = `${number(source.last_jd - source.first_jd, 1)} d`;
  $("metric-position").textContent = `${number(source.ra)}° / ${number(source.dec)}°`;
  $("metric-band").textContent = source.latest_band;
  $("image-band").textContent = `${source.latest_band} band`;
  const image = $("reference-image");
  image.hidden = !source.image;
  $("image-crosshair").hidden = !source.image;
  $("image-fallback").hidden = Boolean(source.image);
  if (source.image) image.src = source.image;
  else $("image-fallback").textContent = source.image_error || "No reference image available.";
  const psImage = $("ps-image");
  psImage.hidden = !source.ps_image;
  $("ps-crosshair").hidden = !source.ps_image;
  $("ps-fallback").hidden = Boolean(source.ps_image);
  if (source.ps_image) psImage.src = source.ps_image;
  else $("ps-fallback").textContent = source.ps_image_error || "No Pan-STARRS image available.";
  const omni = modelSelect.value === "BTSv2-pro";
  $("ztf-image-entry").hidden = !omni;
  $("image-grid").classList.toggle("single", !omni);
  renderMetadata();
  $("empty-state").hidden = true;
  $("workspace").hidden = false;
  xDomain = null;
  requestAnimationFrame(drawLightCurve);
}
function fullXDomain() {
  const max = Math.max(1, ...source.photometry.map((point) => point.days));
  return [0, max];
}
function setXDomain(min, max) {
  const [fullMin, fullMax] = fullXDomain();
  const fullWidth = fullMax - fullMin;
  const width = Math.min(fullWidth, Math.max(fullWidth / 1000, max - min));
  let left = Math.max(fullMin, Math.min(fullMax - width, min));
  xDomain = [left, left + width];
  tooltip.hidden = true;
  drawLightCurve();
}
function zoom(factor, fraction = .5) {
  if (!source || !source.photometry.length) return;
  const [min, max] = xDomain || fullXDomain();
  const width = (max - min) * factor;
  const anchor = min + (max - min) * fraction;
  setXDomain(anchor - width * fraction, anchor + width * (1 - fraction));
}
function drawLightCurve() {
  if (!source || !source.photometry.length || $("workspace").hidden) return;
  tooltip.hidden = true;
  const rect = chart.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const ratio = window.devicePixelRatio || 1;
  chart.width = Math.round(rect.width * ratio); chart.height = Math.round(rect.height * ratio);
  const ctx = chart.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const plot = { left: 48, right: rect.width - 14, top: 12, bottom: rect.height - 32 };
  plotBox = plot;
  const [minDay, maxDay] = xDomain || fullXDomain();
  const visible = source.photometry
    .map((point, index) => ({ point, index }))
    .filter(({ point }) => point.days >= minDay && point.days <= maxDay);
  const forY = visible.length ? visible.map((entry) => entry.point) : source.photometry;
  const minMag = Math.min(...forY.map((p) => p.mag - p.error)) - .2;
  const maxMag = Math.max(...forY.map((p) => p.mag + p.error)) + .2;
  const magSpan = Math.max(.5, maxMag - minMag);
  const x = (day) => plot.left + (day - minDay) / (maxDay - minDay) * (plot.right - plot.left);
  const y = (mag) => plot.top + (mag - minMag) / magSpan * (plot.bottom - plot.top);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.font = "11px Inter, system-ui, sans-serif";
  const lightTheme = document.documentElement.dataset.theme === "light";
  ctx.strokeStyle = lightTheme ? "#edf1f6" : "#23262c";
  ctx.fillStyle = lightTheme ? "#7d8c9b" : "#a8adb6";
  ctx.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick++) {
    const yy = plot.top + tick / 4 * (plot.bottom - plot.top);
    const magnitude = minMag + tick / 4 * magSpan;
    if (tick % 2 === 0) {
      ctx.beginPath(); ctx.moveTo(plot.left, yy); ctx.lineTo(plot.right, yy); ctx.stroke();
    }
    ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(magnitude.toFixed(1), plot.left - 8, yy);
    const xx = plot.left + tick / 4 * (plot.right - plot.left);
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText((minDay + tick / 4 * (maxDay - minDay)).toFixed(maxDay - minDay < 10 ? 1 : 0), xx, plot.bottom + 9);
  }
  plotted = visible.map(({ point, index }) => ({ ...point, index, x: x(point.days), y: y(point.mag) }));
  ctx.save(); ctx.beginPath(); ctx.rect(plot.left, plot.top, plot.right - plot.left, plot.bottom - plot.top); ctx.clip();
  for (const band of ["g", "r", "i"]) {
    const series = plotted.filter((point) => point.band === band);
    const color = lightTheme ? { g: "#238468", r: "#c45d50", i: "#ac813f" }[band] : bandColors[band];
    ctx.strokeStyle = color + "55"; ctx.lineWidth = 1.1;
    if (series.length > 1) {
      ctx.beginPath(); series.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)); ctx.stroke();
    }
    ctx.strokeStyle = color + "99"; ctx.fillStyle = color;
    for (const p of series) {
      const error = Math.max(2, Math.abs(y(p.mag + p.error) - p.y));
      ctx.beginPath(); ctx.moveTo(p.x, p.y - error); ctx.lineTo(p.x, p.y + error); ctx.stroke();
      ctx.beginPath(); ctx.arc(p.x, p.y, 2.8, 0, Math.PI * 2); ctx.fill();
    }
  }
  ctx.restore();
  if (linkedIndex !== null) {
    const target = plotted.find((point) => point.index === linkedIndex);
    if (target) {
      ctx.strokeStyle = lightTheme ? "#007aff" : "#0a84ff";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(target.x, target.y, 6.5, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.arc(target.x, target.y, 2.8, 0, Math.PI * 2); ctx.fillStyle = ctx.strokeStyle; ctx.fill();
    }
  }
}
function renderTaxonomy(result) {
  const root = $("taxonomy"); root.replaceChildren();
  const levels = result.probabilities_by_level || {};
  const parents = levels["1"] || {}, leaves = levels["2"] || {};
  const topLeaf = Object.entries(leaves).sort((a, b) => b[1] - a[1])[0];
  const topBranch = Object.entries(parents).sort((a, b) => b[1] - a[1])[0]?.[0];
  const columns = document.createElement("div"); columns.className = "taxonomy-branches";
  for (const [parent, children] of Object.entries(branches)) {
    const branch = document.createElement("div"); branch.className = `taxonomy-branch${parent === topBranch ? " leading" : ""}`;
    const header = document.createElement("div"); header.className = "branch-header";
    const label = document.createElement("span"); label.textContent = parent;
    const score = document.createElement("span"); score.textContent = percent(parents[parent] || 0);
    header.append(label, score);
    const track = document.createElement("div"); track.className = "prob-track";
    const fill = document.createElement("span"); fill.className = "prob-fill";
    fill.style.width = `${Math.max(0, Math.min(100, (parents[parent] || 0) * 100))}%`; track.append(fill);
    const list = document.createElement("div"); list.className = "leaf-list";
    for (const name of [...children].sort((a, b) => (leaves[b] || 0) - (leaves[a] || 0))) {
      const row = document.createElement("div"); row.className = `leaf-row${name === topLeaf?.[0] ? " top-leaf" : ""}`;
      row.dataset.class = name;
      const childLabel = document.createElement("span"); childLabel.textContent = name;
      const childScore = document.createElement("strong");
      childScore.textContent = name === topLeaf?.[0] ? "Highest" : percent(leaves[name] || 0);
      if (name === topLeaf?.[0]) childScore.className = "leaf-top-label";
      row.append(childLabel, childScore); list.append(row);
    }
    branch.append(header, track, list); columns.append(branch);
  }
  root.append(columns);
  $("top-class").textContent = topLeaf?.[0] || "—";
  $("top-probability").textContent = topLeaf ? percent(topLeaf[1]) : "—";
  $("prediction").dataset.model = result.model;
  $("prediction").dataset.class = topLeaf?.[0] || "";
  $("completed-model").textContent = modelNames[result.model] || result.model;
  $("prediction").hidden = false;
  document.querySelector(".app-shell").classList.add("has-result");
}
const classColorVars = { AGN: "--class-agn", CV: "--class-cv", Varstar: "--class-varstar", "SN-Ia": "--class-sn-ia", "SN-II": "--class-sn-ii", "SN-Ib/c": "--class-sn-ibc", SLSN: "--class-slsn" };
const classColorCache = new Map();
function classColor(name) {
  const key = `${document.documentElement.dataset.theme}:${name}`;
  if (!classColorCache.has(key)) {
    const styles = getComputedStyle(document.documentElement);
    classColorCache.set(key, styles.getPropertyValue(classColorVars[name] || "--muted").trim());
  }
  return classColorCache.get(key);
}
function renderRolling(data) {
  rolling = data;
  linkedIndex = null;
  rollingPlot = null;
  $("rolling-section").hidden = !data?.points?.length;
  rollingTooltip.hidden = true;
  resetRollingDataPanel();
  if (!data?.points?.length) return;
  const leaves = Object.keys(data.points.at(-1).probabilities);
  visibleClasses = new Set([...leaves].sort((a, b) => data.points.at(-1).probabilities[b] - data.points.at(-1).probabilities[a]).slice(0, 3));
  const legend = $("rolling-legend"); legend.replaceChildren();
  for (const name of leaves) {
    const toggle = document.createElement("button"); toggle.type = "button"; toggle.className = "chart-button class-toggle";
    toggle.dataset.class = name;
    const syncToggle = () => {
      toggle.setAttribute("aria-pressed", String(visibleClasses.has(name)));
      toggle.setAttribute("aria-label", `${visibleClasses.has(name) ? "Hide" : "Show"} ${name}`);
    };
    syncToggle();
    toggle.addEventListener("click", () => {
      if (visibleClasses.has(name)) visibleClasses.delete(name); else visibleClasses.add(name);
      syncToggle(); drawRolling();
    });
    const dot = document.createElement("span"); dot.className = "rolling-legend-dot"; dot.setAttribute("aria-hidden", "true");
    const caption = document.createElement("span"); caption.textContent = name;
    toggle.append(dot, caption); legend.append(toggle);
  }
  $("rolling-note").textContent = data.note || "Class probabilities after each observation.";
  requestAnimationFrame(drawRolling);
}
function drawRolling() {
  if (!rolling?.points?.length || $("rolling-section").hidden) return;
  const rect = rollingChart.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const ratio = window.devicePixelRatio || 1;
  rollingChart.width = Math.round(rect.width * ratio); rollingChart.height = Math.round(rect.height * ratio);
  const ctx = rollingChart.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const plot = { left: 46, right: rect.width - 16, top: 15, bottom: rect.height - 34 };
  const points = rolling.points;
  const maxDay = Math.max(1, points.at(-1).days);
  const x = (point) => plot.left + (points.length === 1 ? .5 : point.days / maxDay) * (plot.right - plot.left);
  const y = (probability) => plot.bottom - probability * (plot.bottom - plot.top);
  rollingPlot = { plot, positions: points.map((point, index) => x(point, index)) };
  const light = document.documentElement.dataset.theme === "light";
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.font = "11px Inter, system-ui, sans-serif"; ctx.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick++) {
    const yy = y(tick / 4), xx = plot.left + tick / 4 * (plot.right - plot.left);
    ctx.strokeStyle = light ? "#edf1f6" : "#23262c";
    if (tick % 2 === 0) {
      ctx.beginPath(); ctx.moveTo(plot.left, yy); ctx.lineTo(plot.right, yy); ctx.stroke();
    }
    ctx.fillStyle = light ? "#7d8c9b" : "#a8adb6";
    ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(`${tick * 25}%`, plot.left - 8, yy);
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText((maxDay * tick / 4).toFixed(maxDay < 10 ? 1 : 0), xx, plot.bottom + 10);
  }
  ctx.save(); ctx.beginPath(); ctx.rect(plot.left, plot.top, plot.right - plot.left, plot.bottom - plot.top); ctx.clip();
  for (const name of visibleClasses) {
    ctx.strokeStyle = classColor(name); ctx.lineWidth = 2; ctx.beginPath();
    points.forEach((point, index) => index ? ctx.lineTo(x(point, index), y(point.probabilities[name] || 0)) : ctx.moveTo(x(point, index), y(point.probabilities[name] || 0)));
    ctx.stroke();
    if (points.length <= 50) for (let index = 0; index < points.length; index++) {
      ctx.beginPath(); ctx.arc(x(points[index], index), y(points[index].probabilities[name] || 0), 2.4, 0, Math.PI * 2); ctx.fillStyle = classColor(name); ctx.fill();
    }
  }
  ctx.restore();
  if (linkedIndex !== null && points[linkedIndex]) {
    const px = x(points[linkedIndex]);
    ctx.save();
    ctx.strokeStyle = light ? "#8ea4c0" : "#6f7c8f";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(px, plot.top); ctx.lineTo(px, plot.bottom); ctx.stroke();
    ctx.setLineDash([]);
    for (const name of visibleClasses) {
      const py = y(points[linkedIndex].probabilities[name] || 0);
      ctx.beginPath(); ctx.arc(px, py, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = light ? "#fafbfe" : "#0b0c0e"; ctx.fill();
      ctx.strokeStyle = classColor(name); ctx.lineWidth = 2; ctx.stroke();
    }
    ctx.restore();
  }
  syncRollingTooltip();
}
function rollingIndexFor(point) {
  if (!rolling?.points?.length) return null;
  const candidate = rolling.points[point.index];
  if (candidate && Math.abs(candidate.jd - point.jd) < 1e-6) return point.index;
  const found = rolling.points.findIndex((entry) => Math.abs(entry.jd - point.jd) < 1e-6);
  return found >= 0 ? found : null;
}
function setChartTooltip(element, heading, point = null, band = null) {
  const title = document.createElement("div"); title.className = "tooltip-heading";
  if (band) {
    const marker = document.createElement("span"); marker.className = `tooltip-band band-${band}`; marker.textContent = band;
    title.append(marker, " · ");
  }
  title.append(heading);
  element.replaceChildren(title);
  if (!point) return;
  const scores = document.createElement("div"); scores.className = "tooltip-scores";
  for (const name of [...visibleClasses].sort((a, b) => (point.probabilities[b] || 0) - (point.probabilities[a] || 0))) {
    const row = document.createElement("div"); row.className = "tooltip-score"; row.dataset.class = name;
    const label = document.createElement("span"); label.textContent = name;
    const value = document.createElement("strong"); value.textContent = percent(point.probabilities[name] || 0);
    row.append(label, value); scores.append(row);
  }
  if (scores.childElementCount) element.append(scores);
}
function syncRollingTooltip() {
  const usable = linkedIndex !== null && rollingPlot && rollingPlot.positions[linkedIndex] !== undefined
    && rolling?.points?.[linkedIndex] && !$("rolling-section").hidden;
  if (!usable) { rollingTooltip.hidden = true; return; }
  const point = rolling.points[linkedIndex];
  setChartTooltip(rollingTooltip, `Obs ${point.observation} · JD ${number(point.jd, 5)}`, point, point.band);
  rollingTooltip.hidden = false;
  const rect = rollingChart.getBoundingClientRect();
  const px = rollingPlot.positions[linkedIndex];
  rollingTooltip.style.left = `${Math.max(5, Math.min(px + 12, rect.width - rollingTooltip.offsetWidth - 5))}px`;
  rollingTooltip.style.top = "12px";
}
function setLinked(index) {
  const next = index === undefined ? null : index;
  if (next !== linkedIndex) {
    linkedIndex = next;
    drawLightCurve();
    drawRolling();
  }
  syncRollingTooltip();
}
rollingChart.addEventListener("pointermove", (event) => {
  if (!rollingPlot || !rolling?.points?.length) return;
  const rect = rollingChart.getBoundingClientRect(), px = event.clientX - rect.left;
  let index = 0;
  for (let i = 1; i < rollingPlot.positions.length; i++) if (Math.abs(rollingPlot.positions[i] - px) < Math.abs(rollingPlot.positions[index] - px)) index = i;
  setLinked(Math.abs(rollingPlot.positions[index] - px) > 24 ? null : index);
});
rollingChart.addEventListener("pointerleave", () => setLinked(null));
if ("ResizeObserver" in window) new ResizeObserver(drawRolling).observe(document.querySelector(".rolling-chart-wrap"));
else window.addEventListener("resize", drawRolling);

function rollingClassNames() {
  return rolling?.points?.length ? Object.keys(rolling.points.at(-1).probabilities) : [];
}
function resetRollingDataPanel() {
  $("rolling-data-panel").hidden = true;
  $("rolling-data-head").replaceChildren();
  $("rolling-data-body").replaceChildren();
  const toggle = $("rolling-data-toggle");
  toggle.setAttribute("aria-expanded", "false");
  toggle.textContent = "View data";
}
function buildRollingTable() {
  const classes = rollingClassNames();
  const header = document.createElement("tr");
  for (const label of ["Obs", "JD", "Days", "Band", ...classes]) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    header.append(cell);
  }
  $("rolling-data-head").replaceChildren(header);
  const rows = rolling.points.map((point) => {
    const row = document.createElement("tr");
    const cells = [point.observation, number(point.jd, 5), number(point.days, 3), point.band,
      ...classes.map((name) => percent(point.probabilities[name] ?? 0))];
    cells.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    return row;
  });
  $("rolling-data-body").replaceChildren(...rows);
}
function csvCell(value) {
  const text = String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}
$("rolling-data-toggle").addEventListener("click", () => {
  const panel = $("rolling-data-panel");
  const opening = panel.hidden;
  if (opening) buildRollingTable();
  panel.hidden = !opening;
  $("rolling-data-toggle").setAttribute("aria-expanded", String(opening));
  $("rolling-data-toggle").textContent = opening ? "Hide data" : "View data";
});
$("rolling-download").addEventListener("click", () => {
  if (!source || !rolling?.points?.length) return;
  const classes = rollingClassNames();
  const rows = [["observation", "jd", "days", "band", ...classes],
    ...rolling.points.map((point) => [point.observation, point.jd, point.days, point.band,
      ...classes.map((name) => point.probabilities[name] ?? "")])];
  const csv = rows.map((row) => row.map(csvCell).join(",")).join("\r\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${source.source_id}_evolution_probabilities.csv`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

function isOodSource(sourceData) {
  const mags = (sourceData?.photometry || []).map((p) => Number(p.mag)).filter(Number.isFinite);
  return Boolean(mags.length) && Math.min(...mags) > OOD_MAG_LIMIT;
}
function updateOodWarning() {
  const banner = $("ood-warning");
  if (!banner) return;
  banner.hidden = !isOodSource(source);
}
function renderResult(data, historyId = null) {
  source = data.source;
  source.classification = data.classification;
  activeHistoryId = historyId;
  renderSource();
  updateOodWarning();
  if (data.classification) { renderTaxonomy(data.classification); renderMetadata(); }
  else $("prediction").hidden = true;
  renderRolling(data.rolling);
  renderHistory();
}
function setSidebar(open) {
  document.body.classList.toggle("sidebar-open", open);
  $("history-sidebar").inert = !open;
  $("history-open").setAttribute("aria-expanded", String(open));
  $("history-open").setAttribute("aria-label", open ? "Hide history sidebar" : "Show history sidebar");
  $("history-open").title = open ? "Hide history sidebar" : "Show history sidebar";
  try { localStorage.setItem("oracle-history-open", String(open)); } catch {}
  requestAnimationFrame(() => { drawLightCurve(); drawRolling(); });
}
$("history-open").addEventListener("click", () => setSidebar(!document.body.classList.contains("sidebar-open")));
$("sidebar-backdrop").addEventListener("click", () => { setSidebar(false); $("history-open").focus(); });
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && document.body.classList.contains("sidebar-open")) {
    setSidebar(false);
    $("history-open").focus();
  }
});
$("new-source").addEventListener("click", () => {
  if (busy) return;
  source = null; rolling = null; rollingPlot = null; linkedIndex = null; xDomain = null; plotted = []; activeHistoryId = null;
  showClassifierPage();
  $("workspace").hidden = true;
  $("empty-state").hidden = false;
  $("prediction").hidden = true;
  const oodBanner = $("ood-warning");
  if (oodBanner) oodBanner.hidden = true;
  $("rolling-section").hidden = true;
  resetRollingDataPanel();
  document.querySelector(".app-shell").classList.remove("has-result");
  input.value = "";
  $("evolution-enabled").checked = true;
  message("");
  renderHistory();
  if (window.innerWidth <= 900) setSidebar(false);
  input.focus();
});
let sidebarPreference = null;
try { sidebarPreference = localStorage.getItem("oracle-history-open"); } catch {}
setSidebar(sidebarPreference === null ? window.innerWidth > 900 : sidebarPreference === "true");
function openHistoryDb() {
  return new Promise((resolve, reject) => {
    if (!("indexedDB" in window)) { reject(new Error("Browser storage unavailable")); return; }
    const request = indexedDB.open("oracle-classification-history", 1);
    request.onupgradeneeded = () => request.result.createObjectStore("runs", { keyPath: "id" });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
function historyTransaction(mode, action) {
  return new Promise((resolve, reject) => {
    const transaction = historyDb.transaction("runs", mode);
    const request = action(transaction.objectStore("runs"));
    let result;
    request.onsuccess = () => { result = request.result; };
    transaction.oncomplete = () => resolve(result);
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error || new Error("Browser storage was interrupted."));
  });
}
let historyFilter = "";
function historyMatches(item, query) {
  if (!query) return true;
  const haystack = `${item.source_id} ${item.top_class} ${modelNames[item.model] || item.model}`.toLowerCase();
  return query.split(/\s+/).every((term) => haystack.includes(term));
}
function renderHistory() {
  const list = $("history-list"); list.replaceChildren();
  if (!historyItems.length) { const empty = document.createElement("p"); empty.className = "history-empty"; empty.textContent = "Classified sources will appear here."; list.append(empty); renderTrends(); return; }
  const query = historyFilter.trim().toLowerCase();
  const visible = historyItems.filter((item) => historyMatches(item, query));
  if (!visible.length) { const empty = document.createElement("p"); empty.className = "history-empty"; empty.textContent = `No matches for "${historyFilter.trim()}".`; list.append(empty); renderTrends(); return; }
  for (const item of visible) {
    const entry = document.createElement("button"); entry.type = "button"; entry.className = `history-entry${item.id === activeHistoryId ? " active" : ""}`;
    entry.dataset.model = item.model;
    entry.dataset.class = item.top_class;
    const heading = document.createElement("span"); heading.className = "history-entry-heading";
    const title = document.createElement("strong"); title.textContent = item.source_id;
    const model = document.createElement("span"); model.className = "history-model"; model.textContent = modelNames[item.model] || item.model;
    if (item.rolling) {
      const star = document.createElement("span"); star.className = "history-star"; star.textContent = "★";
      star.title = "Evolution mode"; star.setAttribute("aria-label", "Evolution mode");
      model.append(star);
    }
    heading.append(title, model);
    const detail = document.createElement("span"); detail.className = "history-result";
    const topClass = document.createElement("strong"); topClass.textContent = item.top_class;
    const score = document.createElement("strong"); score.textContent = percent(item.top_probability);
    detail.append(topClass, score);
    const time = document.createElement("small"); time.textContent = new Date(item.created_at).toLocaleString();
    if (item.ood ?? isOodSource(item.data?.source)) {
      const warn = document.createElement("span"); warn.className = "history-ood"; warn.textContent = "⚠";
      warn.title = "Out of distribution: No detections brighter than 18.5 mag";
      warn.setAttribute("aria-label", "Out of distribution: No detections brighter than 18.5 mag");
      time.prepend(warn, " · ");
    }
    entry.append(heading, detail, time);
    entry.addEventListener("click", async () => {
      if (!historyDb || busy) return;
      try {
        const saved = await historyTransaction("readonly", (store) => store.get(item.id));
        if (!saved) return;
        input.value = saved.source_id; modelSelect.value = saved.model; updateModelDescription();
        xDomain = null; showClassifierPage(); renderResult(saved.data, saved.id);
        message("");
        if (window.innerWidth <= 900) setSidebar(false);
      } catch { message("Could not open this saved classification.", true); }
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "history-delete";
    remove.setAttribute("aria-label", `Delete saved classification for ${item.source_id}`);
    remove.title = "Delete saved classification";
    remove.textContent = "×";
    remove.addEventListener("click", () => deleteHistoryItem(item));
    const row = document.createElement("div");
    row.className = "history-item";
    row.append(remove, entry);
    list.append(row);
  }
  renderTrends();
}
async function deleteHistoryItem(item) {
  if (!historyDb || busy) return;
  try {
    await historyTransaction("readwrite", (store) => store.delete(item.id));
    historyItems = historyItems.filter((entry) => entry.id !== item.id);
    if (activeHistoryId === item.id) activeHistoryId = null;
    renderHistory();
  } catch { message("Could not delete this saved classification.", true); }
}
async function saveHistory(data) {
  await historyReady;
  if (!historyDb || !data.classification) return;
  const leaves = data.classification.probabilities_by_level?.["2"] || {};
  const [topClass, topProbability] = Object.entries(leaves).sort((a, b) => b[1] - a[1])[0] || ["—", 0];
  const item = { id: crypto.randomUUID(), created_at: new Date().toISOString(), source_id: data.source.source_id,
    model: data.classification.model, top_class: topClass, top_probability: topProbability, rolling: Boolean(data.rolling), ood: isOodSource(data.source), data };
  await historyTransaction("readwrite", (store) => store.put(item));
  activeHistoryId = item.id;
  historyItems.unshift(item);
  historyFilter = "";
  const searchInput = $("history-search");
  if (searchInput) searchInput.value = "";
  renderHistory();
}
$("history-search")?.addEventListener("input", (event) => {
  historyFilter = event.target.value;
  renderHistory();
});
function showTrendsPage() {
  $("classifier-view").hidden = true;
  $("trends-page").hidden = false;
  renderTrends();
  window.scrollTo(0, 0);
}
function showClassifierPage() {
  $("trends-page").hidden = true;
  $("classifier-view").hidden = false;
}
$("open-trends").addEventListener("click", showTrendsPage);
$("trends-back").addEventListener("click", () => { showClassifierPage(); input.focus(); });
$("history-download").addEventListener("click", () => {
  if (!historyItems.length) return;
  const rows = [["source_id", "model", "final_classification", "probability", "classified_at"],
    ...historyItems.map((item) => [item.source_id, modelNames[item.model] || item.model,
      item.top_class, item.top_probability, item.created_at])];
  const csv = rows.map((row) => row.map(csvCell).join(",")).join("\r\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "oracle_classification_history.csv";
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
function renderHistoryTable() {
  $("history-download").disabled = !historyItems.length;
  $("history-table-panel").hidden = !historyItems.length;
  $("history-table-empty").hidden = historyItems.length > 0;
  const rows = historyItems.map((item) => {
    const row = document.createElement("tr");
    row.dataset.class = item.top_class;
    const values = [item.source_id, modelNames[item.model] || item.model, item.top_class,
      percent(item.top_probability), new Date(item.created_at).toLocaleString()];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    return row;
  });
  $("history-table-body").replaceChildren(...rows);
}
$("copy-source-id").addEventListener("click", () => { if (source) copyText(source.source_id, $("copy-source-id")); });
$("copy-position").addEventListener("click", () => {
  if (source && Number.isFinite(Number(source.ra)) && Number.isFinite(Number(source.dec)))
    copyText(`${source.ra} ${source.dec}`, $("copy-position"));
});
let trendsPieSlices = [];
let trendsModelSlices = [];
let trendsSkyPoints = [];
const modelColors = { "BTSv2-pro": "#c19dff", BTSv2: "#76baff", "BTSv2-lite": "#74d6a5" };
function sizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return null;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}
function renderTrends() {
  if ($("trends-page").hidden) return;
  renderHistoryTable();
  trendsPieSlices = Object.entries(historyItems.reduce((counts, item) => {
    counts[item.top_class] = (counts[item.top_class] || 0) + 1;
    return counts;
  }, {})).sort((a, b) => b[1] - a[1]).map(([name, count]) => ({ name, count }));
  $("trends-class-total").textContent = `${trendsPieSlices.length.toLocaleString()} ${trendsPieSlices.length === 1 ? "class" : "classes"}`;
  $("trends-classes-empty").hidden = historyItems.length > 0;
  const legend = $("trends-legend");
  legend.replaceChildren();
  legend.hidden = !historyItems.length;
  $("trends-pie").closest(".trends-pie-wrap").hidden = !historyItems.length;
  for (const { name, count } of trendsPieSlices) {
    const row = document.createElement("div"); row.className = "trends-legend-item";
    const dot = document.createElement("span"); dot.className = "trends-legend-dot"; dot.style.background = classColor(name);
    const label = document.createElement("span"); label.textContent = name;
    const value = document.createElement("strong"); value.textContent = `${count} · ${percent(historyItems.length ? count / historyItems.length : 0)}`;
    row.append(dot, label, value); legend.append(row);
  }
  trendsModelSlices = Object.entries(historyItems.reduce((counts, item) => {
    counts[item.model] = (counts[item.model] || 0) + 1;
    return counts;
  }, {})).sort((a, b) => b[1] - a[1]).map(([name, count]) => ({ name, count }));
  $("trends-model-total").textContent = `${trendsModelSlices.length.toLocaleString()} ${trendsModelSlices.length === 1 ? "model" : "models"}`;
  $("trends-models-empty").hidden = historyItems.length > 0;
  const modelLegend = $("trends-model-legend");
  modelLegend.replaceChildren();
  modelLegend.hidden = !historyItems.length;
  $("trends-model-pie").closest(".trends-pie-wrap").hidden = !historyItems.length;
  for (const { name: key, count } of trendsModelSlices) {
    const row = document.createElement("div"); row.className = "trends-legend-item";
    const dot = document.createElement("span"); dot.className = "trends-legend-dot"; dot.style.background = modelColors[key] || "#aaa";
    const label = document.createElement("span"); label.textContent = modelNames[key] || key;
    const value = document.createElement("strong"); value.textContent = `${count} · ${percent(historyItems.length ? count / historyItems.length : 0)}`;
    row.append(dot, label, value); modelLegend.append(row);
  }
  trendsSkyPoints = [];
  for (const item of historyItems) {
    const ra = Number(item.data?.source?.ra), dec = Number(item.data?.source?.dec);
    if (Number.isFinite(ra) && Number.isFinite(dec) && ra >= 0 && ra < 360 && dec >= -90 && dec <= 90)
      trendsSkyPoints.push({ ra, dec, cls: item.top_class });
  }
  $("trends-sky-empty").hidden = trendsSkyPoints.length > 0;
  requestAnimationFrame(() => { drawTrendsPies(); drawTrendsSky(); });
}
function drawTrendsPies() {
  if ($("trends-page").hidden) return;
  drawDonut($("trends-pie"), trendsPieSlices, classColor);
  drawDonut($("trends-model-pie"), trendsModelSlices, (name) => modelColors[name] || "#aaa");
}
function drawDonut(canvas, slices, colorFor) {
  if (!slices.length) return;
  const sized = sizeCanvas(canvas);
  if (!sized) return;
  const { ctx, width, height } = sized;
  const total = slices.reduce((sum, slice) => sum + slice.count, 0);
  const cx = width / 2, cy = height / 2, radius = Math.min(width, height) / 2 - 4;
  ctx.clearRect(0, 0, width, height);
  let angle = -Math.PI / 2;
  for (const { name, count } of slices) {
    const sweep = count / total * Math.PI * 2;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, radius, angle, angle + sweep); ctx.closePath();
    ctx.fillStyle = colorFor(name); ctx.fill();
    angle += sweep;
  }
  ctx.globalCompositeOperation = "destination-out";
  ctx.beginPath(); ctx.arc(cx, cy, radius * 0.58, 0, Math.PI * 2); ctx.fill();
  ctx.globalCompositeOperation = "source-over";
}
function mollweideTheta(phi) {
  let theta = phi / 2;
  for (let i = 0; i < 12; i++) {
    theta -= (2 * theta + Math.sin(2 * theta) - Math.PI * Math.sin(phi)) / (2 + 2 * Math.cos(2 * theta));
  }
  return theta;
}
function drawTrendsSky() {
  if ($("trends-page").hidden) return;
  const sized = sizeCanvas($("trends-sky"));
  if (!sized) return;
  const { ctx, width, height } = sized;
  const light = document.documentElement.dataset.theme === "light";
  const project = (ra, dec) => {
    const lambda = ((ra % 360) + 540) % 360 - 180;
    const theta = mollweideTheta(dec * Math.PI / 180);
    const x = 2 * Math.SQRT2 / Math.PI * (lambda * Math.PI / 180) * Math.cos(theta);
    const y = Math.SQRT2 * Math.sin(theta);
    return [width / 2 - x / (2 * Math.SQRT2) * width / 2, height / 2 - y / Math.SQRT2 * height / 2];
  };
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = light ? "#c4cfdc" : "#3a3e45";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.ellipse(width / 2, height / 2, width / 2 - 1, height / 2 - 1, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = light ? "#dbe3ec" : "#2c2f35";
  ctx.lineWidth = 1;
  for (let lon = -150; lon <= 150; lon += 30) {
    ctx.beginPath();
    for (let lat = -90; lat <= 90; lat += 3) {
      const ra = ((lon + 360) % 360);
      const [px, py] = project(ra, lat);
      lat === -90 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  for (let lat = -60; lat <= 60; lat += 30) {
    ctx.beginPath();
    for (let lon = -180; lon <= 180; lon += 3) {
      const ra = ((lon + 360) % 360);
      const [px, py] = project(ra, lat);
      lon === -180 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  for (const point of trendsSkyPoints) {
    const [px, py] = project(point.ra, point.dec);
    ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fillStyle = classColor(point.cls); ctx.fill();
  }
}
const historyReady = openHistoryDb().then(async (db) => {
  historyDb = db;
  historyItems = await historyTransaction("readonly", (store) => store.getAll());
  historyItems.sort((a, b) => b.created_at.localeCompare(a.created_at));
  renderHistory();
}).catch(() => { $("history-list").textContent = "History is unavailable in this browser."; });
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const objectId = extractObjectId(input.value);
  input.value = objectId;
  if (!/^ZTF\d{2}[a-z]+$/i.test(objectId)) { message("Enter a ZTF object ID, such as ZTF18abmrfqv.", true); return; }
  source = null; rolling = null; rollingPlot = null; linkedIndex = null; xDomain = null; plotted = []; activeHistoryId = null;
  showClassifierPage();
  $("workspace").hidden = true; $("empty-state").hidden = false;
  const pendingOod = $("ood-warning");
  if (pendingOod) pendingOod.hidden = true;
  resetRollingDataPanel();
  const useRolling = $("evolution-enabled").checked;
  setBusy(true); message(`Fetching ${objectId}…`);
  try {
    const data = await postJson("/api/analyze", { object_id: objectId, model: modelSelect.value, rolling: useRolling });
    renderResult(data);
    if (data.classification) {
      message(data.rolling_error ? `Classification complete. Evolution plot unavailable: ${data.rolling_error}` : `${modelNames[data.classification.model]} classification complete.`, Boolean(data.rolling_error));
      try { await saveHistory(data); } catch { message("Classification complete, but browser history could not be saved.", true); }
    } else message(data.error || "Classification could not be completed.", true);
  } catch (error) { message(error.message, true); }
  finally { setBusy(false); }
});
$("zoom-in").addEventListener("click", () => zoom(.7));
$("zoom-out").addEventListener("click", () => zoom(1 / .7));
$("zoom-reset").addEventListener("click", () => { xDomain = null; drawLightCurve(); });
chart.addEventListener("dblclick", () => { xDomain = null; drawLightCurve(); });
chart.addEventListener("wheel", (event) => {
  if (!source || !plotBox || (!event.ctrlKey && !event.metaKey)) return;
  event.preventDefault();
  const rect = chart.getBoundingClientRect();
  const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left - plotBox.left) / (plotBox.right - plotBox.left)));
  zoom(event.deltaY < 0 ? .8 : 1.25, fraction);
}, { passive: false });
chart.addEventListener("pointerdown", (event) => {
  if (!source || event.button !== 0 || !plotBox) return;
  chart.setPointerCapture(event.pointerId);
  drag = { x: event.clientX, domain: [...(xDomain || fullXDomain())] };
  chart.classList.add("dragging"); tooltip.hidden = true;
});
chart.addEventListener("pointermove", (event) => {
  if (drag) {
    const width = drag.domain[1] - drag.domain[0];
    const delta = (event.clientX - drag.x) / (plotBox.right - plotBox.left) * width;
    setXDomain(drag.domain[0] - delta, drag.domain[1] - delta);
    return;
  }
  if (!plotted.length) return;
  const rect = chart.getBoundingClientRect();
  const x = event.clientX - rect.left, y = event.clientY - rect.top;
  let nearest = null, distance = Infinity;
  for (const point of plotted) {
    const candidate = Math.hypot(point.x - x, point.y - y);
    if (candidate < distance) { distance = candidate; nearest = point; }
  }
  if (!nearest || distance > 14) { tooltip.hidden = true; setLinked(null); return; }
  const linked = rollingIndexFor(nearest);
  setLinked(linked);
  setChartTooltip(tooltip, `JD ${nearest.jd.toFixed(5)} · ${nearest.mag.toFixed(2)} ± ${nearest.error.toFixed(2)} mag`,
    linked !== null ? rolling.points[linked] : null, nearest.band);
  tooltip.hidden = false;
  tooltip.style.left = `${Math.max(5, Math.min(x + 12, rect.width - tooltip.offsetWidth - 5))}px`;
  tooltip.style.top = `${Math.max(5, y - tooltip.offsetHeight - 6)}px`;
});
function endDrag() { drag = null; chart.classList.remove("dragging"); }
chart.addEventListener("pointerup", endDrag);
chart.addEventListener("pointercancel", endDrag);
chart.addEventListener("pointerleave", () => { tooltip.hidden = true; setLinked(null); });
if ("ResizeObserver" in window) new ResizeObserver(drawLightCurve).observe(document.querySelector(".chart-wrap"));
else window.addEventListener("resize", drawLightCurve);
const dataDetails = $("data-details");
function syncDataToggle() {
  if (!dataDetails) return;
  const open = dataDetails.open;
  const eyeOpen = $("eye-open"), eyeClosed = $("eye-closed");
  if (eyeOpen) eyeOpen.toggleAttribute("hidden", !open);
  if (eyeClosed) eyeClosed.toggleAttribute("hidden", open);
  $("data-summary")?.setAttribute("aria-label", open ? "Hide input data" : "Show input data");
}
if (dataDetails) {
  syncDataToggle();
  dataDetails.addEventListener("toggle", () => {
    syncDataToggle();
    if (dataDetails.open) requestAnimationFrame(() => { drawLightCurve(); });
  });
}

const themeToggle = $("theme-toggle");
function applyTheme(theme) {
  const light = theme === "light";
  document.documentElement.dataset.theme = light ? "light" : "dark";
  themeToggle.setAttribute("aria-pressed", String(light));
  themeToggle.setAttribute("aria-label", light ? "Switch to dark mode" : "Switch to light mode");
  themeToggle.title = light ? "Switch to dark mode" : "Switch to light mode";
  $("theme-icon").textContent = light ? "☾" : "☀";
  document.querySelector('meta[name="theme-color"]').content = light ? "#fafbfe" : "#000000";
  drawLightCurve();
  drawRolling();
  renderTrends();
}
window.addEventListener("resize", () => { if (!$("trends-page").hidden) renderTrends(); });
applyTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");
themeToggle.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  applyTheme(next);
  try { localStorage.setItem("oracle-theme", next); } catch {}
});
