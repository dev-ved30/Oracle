const $ = (id) => document.getElementById(id);
const form = $("source-form");
const input = $("source-id");
const modelSelect = $("model-select");
const button = $("analyze-button");
const chart = $("lightcurve-chart");
const tooltip = $("chart-tooltip");
const modelNames = { "BTSv2-pro": "ORACLE-2 Omni", BTSv2: "ORACLE-2", "BTSv2-lite": "ORACLE-2 Lite" };
const bandColors = { g: "#59d39a", r: "#ff8477", i: "#e9b66f" };
const branches = { Persistent: ["AGN", "CV", "Varstar"], Transient: ["SN-Ia", "SN-II", "SN-Ib/c", "SLSN"] };
let source = null;
let busy = false;
let plotted = [];
let xDomain = null;
let plotBox = null;
let drag = null;

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
  button.innerHTML = value ? "Fetching & classifying…" : 'Classify <span aria-hidden="true">↗</span>';
}
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
  $("metadata-note").textContent = source.metadata_error || `${available.length} of ${items.length} values available. Missing values are passed to the context models as −9.`;
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
  const omni = modelSelect.value === "BTSv2-pro";
  $("image-section").hidden = !omni;
  document.querySelector(".visual-grid").classList.toggle("single", !omni);
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
  ctx.strokeStyle = "#303238"; ctx.fillStyle = "#a8adb6"; ctx.lineWidth = 1;
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
    ctx.strokeStyle = bandColors[band] + "55"; ctx.lineWidth = 1.1;
    if (series.length > 1) {
      ctx.beginPath(); series.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)); ctx.stroke();
    }
    ctx.strokeStyle = bandColors[band] + "99"; ctx.fillStyle = bandColors[band];
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
  const rootNode = document.createElement("div"); rootNode.className = "taxonomy-root"; rootNode.textContent = "Alert · 100%";
  const columns = document.createElement("div"); columns.className = "taxonomy-branches";
  for (const [parent, children] of Object.entries(branches)) {
    const branch = document.createElement("div"); branch.className = "taxonomy-branch";
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
  root.append(rootNode, columns);
  $("top-class").textContent = topLeaf?.[0] || "—";
  $("top-probability").textContent = topLeaf ? percent(topLeaf[1]) : "—";
  $("prediction-model").textContent = modelNames[result.model] || result.model;
  const missing = (result.missing_context_features || []).length;
  $("prediction-note").textContent = missing ? `${missing} contextual features were unavailable and passed as −9. Probabilities are model outputs.` : "Probabilities are model outputs.";
  $("prediction").hidden = false;
}
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const objectId = input.value.trim();
  if (!/^ZTF\d{2}[a-z]+$/i.test(objectId)) { message("Enter a ZTF object ID, such as ZTF18abmrfqv.", true); return; }
  source = null; xDomain = null; plotted = [];
  $("workspace").hidden = true; $("empty-state").hidden = false;
  setBusy(true); message(`Fetching ${objectId} and running ${modelNames[modelSelect.value]}…`);
  try {
    const data = await postJson("/api/analyze", { object_id: objectId, model: modelSelect.value });
    source = data.source;
    renderSource();
    if (data.classification) { renderTaxonomy(data.classification); message(`${modelNames[data.classification.model]} classification complete.`); }
    else { $("prediction").hidden = true; message(data.error || "Classification could not be completed.", true); }
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
