const toggle = document.getElementById("theme-toggle");
function applyTheme(theme) {
  const light = theme === "light";
  document.documentElement.dataset.theme = light ? "light" : "dark";
  toggle.setAttribute("aria-pressed", String(light));
  toggle.setAttribute("aria-label", light ? "Switch to dark mode" : "Switch to light mode");
  toggle.title = light ? "Switch to dark mode" : "Switch to light mode";
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = light ? "#fafbfe" : "#000000";
}
applyTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");
toggle.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  applyTheme(next);
  try { localStorage.setItem("oracle-theme", next); } catch {}
});
