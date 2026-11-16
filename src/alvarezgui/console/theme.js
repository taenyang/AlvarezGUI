(() => {
  const STORAGE_KEY = "alvarezgui-theme";
  const systemTheme = () => window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
  const savedTheme = () => {
    try { return localStorage.getItem(STORAGE_KEY); } catch { return null; }
  };
  const applyTheme = (theme) => {
    document.documentElement.dataset.theme = theme;
    document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
      const next = theme === "light" ? "dark" : "light";
      button.setAttribute("aria-label", next === "light" ? "切换到白天模式" : "切换到夜间模式");
      button.title = next === "light" ? "切换到白天模式" : "切换到夜间模式";
      const label = button.querySelector("[data-theme-label]");
      if (label) label.textContent = next === "light" ? "白天" : "夜间";
      const icon = button.querySelector("[data-theme-icon]");
      if (icon) icon.textContent = next === "light" ? "☼" : "☾";
    });
  };

  applyTheme(savedTheme() || systemTheme());
  document.addEventListener("click", (event) => {
    const button = event.target.closest?.("[data-theme-toggle]");
    if (!button) return;
    const theme = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    try { localStorage.setItem(STORAGE_KEY, theme); } catch { /* private browsing */ }
    applyTheme(theme);
  });
})();
