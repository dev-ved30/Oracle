function buildClassificationReport(data, { classifiedAt } = {}) {
  const source = data.source;
  const result = data.classification;
  const escape = (value) => String(value ?? "—").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
  const numeric = (value, digits = 3) => value !== null && value !== undefined && Number.isFinite(Number(value))
    ? Number(value).toFixed(digits) : "—";
  const probability = (value) => {
    const score = Number(value) * 100;
    if (!Number.isFinite(score)) return "—";
    return score > 0 && score < .01 ? "<0.01%" : `${score.toFixed(score >= 10 ? 1 : 2)}%`;
  };
  const models = { "BTSv2-pro": "ORACLE-2 Omni", BTSv2: "ORACLE-2", "BTSv2-lite": "ORACLE-2 Lite" };
  const colors = { AGN: "#1769c2", CV: "#21845d", Varstar: "#976d19", "SN-Ia": "#7254ae",
    "SN-II": "#b95148", "SN-Ib/c": "#a14e84", SLSN: "#187e92" };
  const levels = result.probabilities_by_level || {};
  const leaves = Object.entries(levels["2"] || {}).sort((a, b) => b[1] - a[1]);
  const [topClass, topProbability] = leaves[0] || ["—", NaN];
  const table = (headers, rows) => `<div class="table-scroll"><table><thead><tr>${headers.map((label) => `<th scope="col">${escape(label)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((value) => `<td>${escape(value)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  const probabilities = (entries) => `<table><thead><tr><th scope="col">Class</th><th scope="col">Probability</th></tr></thead><tbody>${entries.map(([name, score]) => `<tr><td style="color:${colors[name] || "inherit"}">${escape(name)}</td><td>${escape(probability(score))}</td></tr>`).join("")}</tbody></table>`;
  const photometry = source.photometry || [];
  const magnitudes = photometry.map((point) => Number(point.mag)).filter(Number.isFinite);
  const outOfDistribution = magnitudes.length > 0 && Math.min(...magnitudes) > 18.5;
  const missing = result.missing_context_features || [];
  const stamp = classifiedAt ? new Date(classifiedAt) : null;
  const timestamp = stamp && Number.isFinite(stamp.getTime()) ? stamp.toISOString() : "Unavailable";
  const picture = (image, label) => typeof image === "string" && /^data:image\/(?:png|jpeg|jpg|webp);base64,[A-Za-z0-9+/=\s]+$/.test(image)
    ? `<figure><img src="${escape(image)}" alt="${escape(label)}"><figcaption>${escape(label)}</figcaption></figure>` : "";
  const pictures = picture(source.image, "ZTF reference image") + picture(source.ps_image, "Pan-STARRS1 color image");
  const points = data.rolling?.points || [];
  const evolutionClasses = [...new Set(points.flatMap((point) => Object.keys(point.probabilities || {})))];
  const evolution = points.length ? `<section><h2>Evolution over time</h2>${table(["Observation", "JD", "Days", "Band", ...evolutionClasses], points.map((point) => [point.observation, numeric(point.jd, 5), numeric(point.days), point.band, ...evolutionClasses.map((name) => probability(point.probabilities?.[name] || 0))]))}</section>` : "";
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${escape(source.source_id)} · ORACLE classification report</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#1c2633;font:14px/1.5 system-ui,sans-serif}main{max-width:960px;margin:32px auto;padding:40px;background:#fff;border:1px solid #dce3ec;border-radius:12px}header p{color:#617184}h1{font-size:30px;margin:8px 0}h2{font-size:18px;margin:0 0 16px}section{margin-top:32px}.brand{font-size:12px;font-weight:800;letter-spacing:.16em;color:#1769c2}.lead{display:flex;flex-wrap:wrap;gap:12px 28px;padding:24px 0}.lead strong{font-size:30px}.summary{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}.summary dt{color:#617184;font-size:12px}.summary dd{margin:4px 0 0;font-weight:650}.notice{padding:12px 16px;border:1px solid #eeb8b3;border-radius:6px;background:#fef0ef;color:#b3261e}.branches,.images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums;text-align:left}th,td{padding:8px 10px;border-bottom:1px solid #e4e9f0;vertical-align:top;overflow-wrap:anywhere}th{color:#617184;font-weight:650}.table-scroll{overflow-x:auto}figure{margin:0}img{display:block;max-width:100%;height:auto;max-height:320px;margin:auto}figcaption{margin-top:8px;color:#617184;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;background:#f5f7fa;padding:16px}footer{margin-top:32px;color:#617184;font-size:12px}a{color:#1769c2}@media(max-width:640px){main{margin:0;padding:24px;border:0;border-radius:0}.summary,.branches,.images{grid-template-columns:1fr 1fr}.branches{grid-template-columns:1fr}}@media print{body{background:#fff}main{max-width:none;margin:0;padding:0;border:0}h2{break-after:avoid}tr,figure,.notice{break-inside:avoid}.table-scroll{overflow:visible}th,td{padding:5px;font-size:10px}a{color:inherit}}
</style></head><body><main>
<header><span class="brand">ORACLE² · CLASSIFICATION REPORT</span><h1>${escape(source.source_id)}</h1><p>Completed using ${escape(models[result.model] || result.model)} · ${escape(timestamp)}</p></header>
${outOfDistribution ? '<p class="notice"><strong>Out of distribution:</strong> No detections brighter than 18.5 mag. Classification may be less reliable.</p>' : ""}
<div class="lead"><strong style="color:${colors[topClass] || "inherit"}">${escape(topClass)}</strong><strong>${escape(probability(topProbability))}</strong></div>
<dl class="summary"><div><dt>Detections</dt><dd>${escape(source.detections)}</dd></div><div><dt>Time span</dt><dd>${escape(numeric(source.last_jd - source.first_jd, 1))} d</dd></div><div><dt>RA / Dec</dt><dd>${escape(numeric(source.ra))}° / ${escape(numeric(source.dec))}°</dd></div><div><dt>Latest band</dt><dd>${escape(source.latest_band)}</dd></div></dl>
<section><h2>Classification probabilities</h2><div class="branches"><div><h3>Final classes</h3>${probabilities(leaves)}</div><div><h3>Branches</h3>${probabilities(Object.entries(levels["1"] || {}).sort((a, b) => b[1] - a[1]))}</div></div></section>
${missing.length ? `<p class="notice">Unavailable model context: ${escape(missing.join(", "))}. These features were passed as −9.</p>` : ""}
${data.rolling_error ? `<p class="notice">Evolution unavailable: ${escape(data.rolling_error)}</p>` : ""}
${evolution}
${pictures ? `<section><h2>Sky context</h2><div class="images">${pictures}</div></section>` : ""}
<section><h2>Photometry</h2>${table(["JD", "Days since first detection", "Band", "Magnitude", "Magnitude error"], photometry.map((point) => [numeric(point.jd, 5), numeric(point.days), point.band, numeric(point.mag), numeric(point.error)]))}</section>
<section><h2>Source context</h2>${source.metadata_error ? `<p>${escape(source.metadata_error)}</p>` : ""}${table(["Feature", "Value"], (source.metadata || []).map((item) => [item.name, item.value === null ? "Unavailable" : item.value]))}</section>
<section><h2>ORACLE output</h2><pre>${escape(JSON.stringify(result, null, 2))}</pre></section>
<footer>Source data: Babamul · <a href="https://babamul.caltech.edu/objects/ZTF/${encodeURIComponent(source.source_id)}">Open in Babamul</a> · <a href="https://fritz.science/source/${encodeURIComponent(source.source_id)}">Open in Fritz</a></footer>
</main></body></html>`;
}
