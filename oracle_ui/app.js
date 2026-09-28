const $ = (id) => document.getElementById(id);
const form = $("source-form");
const input = $("source-id");
const modelSelect = $("model-select");
const button = $("analyze-button");
const chart = $("lightcurve-chart");
const tooltip = $("chart-tooltip");
const rollingChart = $("rolling-chart");
const rollingTooltip = $("rolling-tooltip");
const modelNames = { "BTSv2-pro": "ORACLE-2 Omni", BTSv2: "ORACLE-2", "BTSv2-lite": "ORACLE-2 Lite" };
const modelDescriptions = { "BTSv2-pro": "Light curve + source context + ZTF reference image", BTSv2: "Light curve + source context", "BTSv2-lite": "Light curve only" };
const bandColors = { g: "#59d39a", r: "#ff8477", i: "#e9b66f" };
const branches = { Persistent: ["AGN", "CV", "Varstar"], Transient: ["SN-Ia", "SN-II", "SN-Ib/c", "SLSN"] };
let source = null;
let busy = false;
let plotted = [];
let xDomain = null;
let plotBox = null;
let drag = null;
let rolling = null;
let visibleClasses = new Set();
let rollingPlot = null;
let historyDb = null;
let historyItems = [];
let activeHistoryId = null;

function message(value, error = false) {
  const target = $("app-message");
  target.textContent = value;
  target.classList.toggle("error", error);
  target.hidden = !value;
}
function setBusy(value) {
  busy = value;
  button.disabled = value;
  modelSelect.disabled = value;
  $("new-source").disabled = value;
  button.setAttribute("aria-label", value ? "Fetching and classifying source" : "Classify source");
  button.title = value ? "Fetching and classifying source" : "Classify source";
  button.firstElementChild.textContent = value ? "…" : "↑";
}
function updateModelDescription() { $("model-description").textContent = modelDescriptions[modelSelect.value] || ""; }
modelSelect.addEventListener("change", updateModelDescription);
updateModelDescription();
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
  $("metadata-details").open = false;
  $("metadata-summary").textContent = source.metadata_error ? "Metadata unavailable" : `Show metadata (${items.length} fields)`;
  const unavailable = source.classification?.missing_context_features?.length;
  $("metadata-note").textContent = `${source.metadata_error || `${available.length} of ${items.length} values available.`}${unavailable ? ` ${unavailable} contextual features were unavailable and passed as −9.` : ""}`;
}
function renderSource() {
  $("source-title").textContent = source.source_id;
  $("source-subtitle").textContent = `Babamul · Latest detection JD ${number(source.last_jd, 5)}`;
  $("broker-link").href = `https://babamul.caltech.edu/objects/ZTF/${encodeURIComponent(source.source_id)}`;
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
  const rect = chart.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const ratio = window.devicePixelRatio || 1;
  chart.width = Math.round(rect.width * ratio); chart.height = Math.round(rect.height * ratio);
  const ctx = chart.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const plot = { left: 48, right: rect.width - 14, top: 12, bottom: rect.height - 32 };
  plotBox = plot;
  const [minDay, maxDay] = xDomain || fullXDomain();
  const visible = source.photometry.filter((p) => p.days >= minDay && p.days <= maxDay);
  const forY = visible.length ? visible : source.photometry;
  const minMag = Math.min(...forY.map((p) => p.mag - p.error)) - .2;
  const maxMag = Math.max(...forY.map((p) => p.mag + p.error)) + .2;
  const magSpan = Math.max(.5, maxMag - minMag);
  const x = (day) => plot.left + (day - minDay) / (maxDay - minDay) * (plot.right - plot.left);
  const y = (mag) => plot.top + (mag - minMag) / magSpan * (plot.bottom - plot.top);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.font = "11px Inter, system-ui, sans-serif";
  const lightTheme = document.documentElement.dataset.theme === "light";
  ctx.strokeStyle = lightTheme ? "#e7edf5" : "#303238";
  ctx.fillStyle = lightTheme ? "#7d8c9b" : "#a8adb6";
  ctx.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick++) {
    const yy = plot.top + tick / 4 * (plot.bottom - plot.top);
    const magnitude = minMag + tick / 4 * magSpan;
    ctx.beginPath(); ctx.moveTo(plot.left, yy); ctx.lineTo(plot.right, yy); ctx.stroke();
    ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(magnitude.toFixed(1), plot.left - 8, yy);
    const xx = plot.left + tick / 4 * (plot.right - plot.left);
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText((minDay + tick / 4 * (maxDay - minDay)).toFixed(maxDay - minDay < 10 ? 1 : 0), xx, plot.bottom + 9);
  }
  plotted = visible.map((point) => ({ ...point, x: x(point.days), y: y(point.mag) }));
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
      const childLabel = document.createElement("span"); childLabel.textContent = name;
      const childScore = document.createElement("strong"); childScore.textContent = percent(leaves[name] || 0);
      row.append(childLabel, childScore); list.append(row);
    }
    branch.append(header, track, list); columns.append(branch);
  }
  root.append(columns);
  $("top-class").textContent = topLeaf?.[0] || "—";
  $("top-probability").textContent = topLeaf ? percent(topLeaf[1]) : "—";
  $("prediction-model").textContent = modelNames[result.model] || result.model;
  $("prediction-note").textContent = "Probabilities are model outputs.";
  $("prediction").hidden = false;
  document.querySelector(".app-shell").classList.add("has-result");
}
const classColors = { AGN: "#0a84ff", CV: "#59d39a", Varstar: "#e9b66f", "SN-Ia": "#b395ff", "SN-II": "#ff8477", "SN-Ib/c": "#f2a5d8", SLSN: "#63cee2" };
function renderRolling(data) {
  rolling = data;
  $("rolling-section").hidden = !data?.points?.length;
  rollingTooltip.hidden = true;
  if (!data?.points?.length) return;
  const leaves = Object.keys(data.points.at(-1).probabilities);
  visibleClasses = new Set([...leaves].sort((a, b) => data.points.at(-1).probabilities[b] - data.points.at(-1).probabilities[a]).slice(0, 3));
  const legend = $("rolling-legend"); legend.replaceChildren();
  for (const name of leaves) {
    const label = document.createElement("label"); label.className = "rolling-legend-item";
    const checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.checked = visibleClasses.has(name);
    checkbox.addEventListener("change", () => { if (checkbox.checked) visibleClasses.add(name); else visibleClasses.delete(name); drawRolling(); });
    const dot = document.createElement("span"); dot.className = "rolling-legend-dot"; dot.style.background = classColors[name] || "#aaa";
    const caption = document.createElement("span"); caption.textContent = name;
    label.append(checkbox, dot, caption); legend.append(label);
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
    ctx.strokeStyle = light ? "#e7edf5" : "#303238";
    ctx.beginPath(); ctx.moveTo(plot.left, yy); ctx.lineTo(plot.right, yy); ctx.stroke();
    ctx.fillStyle = light ? "#7d8c9b" : "#a8adb6";
    ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(`${tick * 25}%`, plot.left - 8, yy);
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText((maxDay * tick / 4).toFixed(maxDay < 10 ? 1 : 0), xx, plot.bottom + 10);
  }
  ctx.save(); ctx.beginPath(); ctx.rect(plot.left, plot.top, plot.right - plot.left, plot.bottom - plot.top); ctx.clip();
  for (const name of visibleClasses) {
    ctx.strokeStyle = classColors[name] || "#aaa"; ctx.lineWidth = 2; ctx.beginPath();
    points.forEach((point, index) => index ? ctx.lineTo(x(point, index), y(point.probabilities[name] || 0)) : ctx.moveTo(x(point, index), y(point.probabilities[name] || 0)));
    ctx.stroke();
    if (points.length <= 50) for (let index = 0; index < points.length; index++) {
      ctx.beginPath(); ctx.arc(x(points[index], index), y(points[index].probabilities[name] || 0), 2.4, 0, Math.PI * 2); ctx.fillStyle = classColors[name] || "#aaa"; ctx.fill();
    }
  }
  ctx.restore();
}
rollingChart.addEventListener("pointermove", (event) => {
  if (!rollingPlot || !rolling?.points?.length) return;
  const rect = rollingChart.getBoundingClientRect(), x = event.clientX - rect.left;
  let index = 0;
  for (let i = 1; i < rollingPlot.positions.length; i++) if (Math.abs(rollingPlot.positions[i] - x) < Math.abs(rollingPlot.positions[index] - x)) index = i;
  const point = rolling.points[index];
  if (Math.abs(rollingPlot.positions[index] - x) > 24) { rollingTooltip.hidden = true; return; }
  const scores = [...visibleClasses].map((name) => `${name} ${percent(point.probabilities[name] || 0)}`).join(" · ");
  rollingTooltip.textContent = `Obs ${point.observation} · JD ${number(point.jd, 5)}${scores ? ` · ${scores}` : ""}`;
  rollingTooltip.hidden = false;
  rollingTooltip.style.left = `${Math.max(5, Math.min(x + 12, rect.width - rollingTooltip.offsetWidth - 5))}px`;
  rollingTooltip.style.top = "12px";
});
rollingChart.addEventListener("pointerleave", () => { rollingTooltip.hidden = true; });
if ("ResizeObserver" in window) new ResizeObserver(drawRolling).observe(document.querySelector(".rolling-chart-wrap"));
else window.addEventListener("resize", drawRolling);

function renderResult(data, historyId = null) {
  source = data.source;
  source.classification = data.classification;
  activeHistoryId = historyId;
  renderSource();
  if (data.classification) { renderTaxonomy(data.classification); renderMetadata(); }
  else $("prediction").hidden = true;
  renderRolling(data.rolling);
  $("advanced-options").open = false;
  renderHistory();
}
function setSidebar(open) {
  document.body.classList.toggle("sidebar-open", open);
  $("history-sidebar").inert = !open;
  $("history-open").setAttribute("aria-expanded", String(open));
  try { localStorage.setItem("oracle-history-open", String(open)); } catch {}
  requestAnimationFrame(() => { drawLightCurve(); drawRolling(); });
}
$("history-open").addEventListener("click", () => setSidebar(true));
$("history-close").addEventListener("click", () => setSidebar(false));
$("new-source").addEventListener("click", () => {
  if (busy) return;
  source = null; rolling = null; rollingPlot = null; xDomain = null; plotted = []; activeHistoryId = null;
  $("workspace").hidden = true;
  $("empty-state").hidden = false;
  $("prediction").hidden = true;
  $("rolling-section").hidden = true;
  document.querySelector(".app-shell").classList.remove("has-result");
  input.value = "";
  $("rolling-enabled").checked = false;
  $("advanced-options").open = false;
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
function renderHistory() {
  const list = $("history-list"); list.replaceChildren();
  if (!historyItems.length) { const empty = document.createElement("p"); empty.className = "history-empty"; empty.textContent = "Classified sources will appear here."; list.append(empty); return; }
  for (const item of historyItems) {
    const entry = document.createElement("button"); entry.type = "button"; entry.className = `history-entry${item.id === activeHistoryId ? " active" : ""}`;
    const title = document.createElement("strong"); title.textContent = item.source_id;
    const model = document.createElement("span"); model.className = "history-model"; model.textContent = modelNames[item.model] || item.model;
    const detail = document.createElement("span"); detail.className = "history-result";
    const topClass = document.createElement("strong"); topClass.textContent = item.top_class;
    const score = document.createElement("strong"); score.textContent = percent(item.top_probability);
    detail.append(topClass, score);
    const time = document.createElement("small"); time.textContent = new Date(item.created_at).toLocaleString() + (item.rolling ? " · Rolling" : "");
    entry.append(title, model, detail, time);
    entry.addEventListener("click", async () => {
      if (!historyDb || busy) return;
      try {
        const saved = await historyTransaction("readonly", (store) => store.get(item.id));
        if (!saved) return;
        input.value = saved.source_id; modelSelect.value = saved.model; updateModelDescription();
        xDomain = null; renderResult(saved.data, saved.id);
        message(`Showing saved ${saved.source_id} classification.`);
        if (window.innerWidth <= 900) setSidebar(false);
      } catch { message("Could not open this saved classification.", true); }
    });
    list.append(entry);
  }
}
async function saveHistory(data) {
  await historyReady;
  if (!historyDb || !data.classification) return;
  const leaves = data.classification.probabilities_by_level?.["2"] || {};
  const [topClass, topProbability] = Object.entries(leaves).sort((a, b) => b[1] - a[1])[0] || ["—", 0];
  const item = { id: crypto.randomUUID(), created_at: new Date().toISOString(), source_id: data.source.source_id,
    model: data.classification.model, top_class: topClass, top_probability: topProbability, rolling: Boolean(data.rolling), data };
  await historyTransaction("readwrite", (store) => store.put(item));
  activeHistoryId = item.id;
  historyItems.unshift(item);
  renderHistory();
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
  const objectId = input.value.trim();
  if (!/^ZTF\d{2}[a-z]+$/i.test(objectId)) { message("Enter a ZTF object ID, such as ZTF18abmrfqv.", true); return; }
  source = null; rolling = null; xDomain = null; plotted = []; activeHistoryId = null;
  $("workspace").hidden = true; $("empty-state").hidden = false;
  const useRolling = $("rolling-enabled").checked;
  setBusy(true); message(`Fetching ${objectId} and running ${modelNames[modelSelect.value]}${useRolling ? " after each observation" : ""}…`);
  try {
    const data = await postJson("/api/analyze", { object_id: objectId, model: modelSelect.value, rolling: useRolling });
    renderResult(data);
    if (data.classification) {
      message(data.rolling_error ? `Classification complete. Rolling plot unavailable: ${data.rolling_error}` : `${modelNames[data.classification.model]} classification complete.`, Boolean(data.rolling_error));
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
  if (!source || !plotBox) return;
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
  if (!nearest || distance > 14) { tooltip.hidden = true; return; }
  tooltip.textContent = `${nearest.band} · JD ${nearest.jd.toFixed(5)} · ${nearest.mag.toFixed(2)} ± ${nearest.error.toFixed(2)} mag`;
  tooltip.hidden = false;
  tooltip.style.left = `${Math.max(5, Math.min(x + 12, rect.width - tooltip.offsetWidth - 5))}px`;
  tooltip.style.top = `${Math.max(5, y - 35)}px`;
});
function endDrag() { drag = null; chart.classList.remove("dragging"); }
chart.addEventListener("pointerup", endDrag);
chart.addEventListener("pointercancel", endDrag);
chart.addEventListener("pointerleave", () => { tooltip.hidden = true; });
if ("ResizeObserver" in window) new ResizeObserver(drawLightCurve).observe(document.querySelector(".chart-wrap"));
else window.addEventListener("resize", drawLightCurve);

const themeToggle = $("theme-toggle");
function applyTheme(theme) {
  const light = theme === "light";
  document.documentElement.dataset.theme = light ? "light" : "dark";
  themeToggle.setAttribute("aria-pressed", String(light));
  themeToggle.setAttribute("aria-label", light ? "Switch to dark mode" : "Switch to light mode");
  $("theme-icon").textContent = light ? "☾" : "☀";
  $("theme-label").textContent = light ? "Dark mode" : "Light mode";
  document.querySelector('meta[name="theme-color"]').content = light ? "#fafbfe" : "#000000";
  drawLightCurve();
  drawRolling();
}
applyTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");
themeToggle.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  applyTheme(next);
  try { localStorage.setItem("oracle-theme", next); } catch {}
});
