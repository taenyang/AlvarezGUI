// AlvarezGUI console. All dynamic text goes through textContent (h()), never innerHTML.
// Exception: the guide page parses the repo README (raw HTML escaped) with DOMParser.

const view = document.getElementById("view");
let pollTimer = null;
let routeToken = 0;

// ---------------------------------------------------------------- helpers

function h(tag, attrs = {}, ...children) {
  const isSvg = ["svg", "defs", "marker", "line", "path", "circle", "rect", "text", "g", "polyline", "polygon"].includes(tag);
  const node = isSvg ? document.createElementNS("http://www.w3.org/2000/svg", tag) : document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "class") node.setAttribute("class", value);
    else if (key in node && !isSvg && key !== "list") node[key] = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    cache: "no-store",
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (response.status === 401) { location.href = "/console/login"; throw new Error("未登录"); }
  const body = response.headers.get("content-type")?.includes("json") ? await response.json() : null;
  if (!response.ok) {
    const detail = body?.detail;
    const error = new Error(typeof detail === "string" ? detail : JSON.stringify(detail ?? response.statusText));
    error.detail = detail;
    error.code = detail && typeof detail === "object" ? detail.code : undefined;
    throw error;
  }
  return body;
}

const fmt = {
  pct: (v) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(0)}%`),
  num: (v, digits = 2) => (v === null || v === undefined ? "–" : Number(v).toLocaleString(undefined, { maximumFractionDigits: digits })),
  ms: (v) => (v === null || v === undefined ? "–" : `${Math.round(v).toLocaleString()} ms`),
  bytes: (v) => {
    if (v === null || v === undefined) return "–";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let i = 0;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
    return `${v.toFixed(i ? 1 : 0)} ${units[i]}`;
  },
  time: (v) => (v ? new Date(v).toLocaleString() : "–"),
  short: (id) => (id ? (id.length > 22 ? `${id.slice(0, 14)}…${id.slice(-6)}` : id) : "–"),
};

// Status always ships as dot + text label, never color alone.
const STATUS = {
  SUCCEEDED: ["good", "成功"], success: ["good", "任务完成"], READY: ["good", "在线"], FINISHED: ["good", "已结束"],
  RUNNING: ["running", "运行中"], running: ["running", "运行中"], RUNNING_ENV: ["running", "执行动作"],
  WAITING_MODEL: ["running", "等待模型"], ALLOCATING: ["running", "分配中"], EVALUATING: ["running", "评测中"],
  QUEUED: ["", "排队"], queued: ["", "排队"],
  unsuccessful: ["warning", "未完成任务"], max_steps: ["warning", "超过步数"], invalid_output: ["serious", "非法输出"],
  FAILED: ["critical", "失败"], error: ["critical", "运行错误"], UNREACHABLE: ["critical", "失联"],
  CANCELED: ["", "已取消"], canceled: ["", "已取消"], UNKNOWN: ["", "未知"],
};
const MODEL_STATUS = {
  REGISTERED: ["", "已登记"], READY: ["good", "已注册"], RETIRED: ["warning", "待清理"],
};
function badge(status) {
  const [kind, label] = STATUS[status] || ["", status];
  return h("span", { class: `badge ${kind}`, title: status }, label);
}
function modelBadge(status) {
  const [kind, label] = MODEL_STATUS[status] || ["", status];
  return h("span", { class: `badge ${kind}`, title: status }, label);
}
const ADAPTER_HINT = "只含 LoRA adapter 权重，加载时依赖基础模型；只能继续 LoRA SFT，在线策略算法 和全参数训练需先合并为完整权重";
// <option>s grouped as the resources page shows them: models, LoRA adapters, checkpoints.
function modelOptions(models, option) {
  const groups = [["模型", (m) => m.kind !== "checkpoint" && !m.adapter], ["LoRA", (m) => m.kind !== "checkpoint" && m.adapter],
    ["Checkpoint", (m) => m.kind === "checkpoint"]];
  return groups.map(([label, test]) => {
    const rows = models.filter(test);
    return rows.length ? h("optgroup", { label }, rows.map(option)) : null;
  });
}

function modelLabel(m) {
  return m.adapter ? `${m.version}（LoRA adapter）` : m.version;
}

function table(columns, rows, onClick) {
  return h("table", {},
    h("thead", {}, h("tr", {}, columns.map((c) => h("th", { class: c.num ? "num" : "" }, c.label)))),
    h("tbody", {}, rows.length
      ? rows.map((row) => h("tr", { class: onClick ? "clickable" : "", onclick: onClick ? () => onClick(row) : null },
        columns.map((c) => h("td", { class: c.num ? "num" : "" }, c.render ? c.render(row) : row[c.key] ?? "–"))))
      : h("tr", {}, h("td", { colSpan: columns.length, class: "muted" }, "暂无数据"))));
}

function tile(label, value, sub) {
  return h("div", { class: "card tile" }, h("div", { class: "label" }, label), h("div", { class: "value" }, value), sub ? h("div", { class: "sub" }, sub) : null);
}

function poll(fn, ms) {
  clearInterval(pollTimer);
  const token = routeToken;
  const run = async () => {
    if (token !== routeToken) return;
    try { await fn(); } catch (error) { console.error(error); }
  };
  pollTimer = setInterval(run, ms);
  return run();
}

// Confirmed destructive action; then re-render (or go to `next`) so the result is visible at once.
function actionButton(label, question, request, next, className = "danger") {
  return h("button", { class: className, onclick: async (event) => {
    event.stopPropagation();
    if (!confirm(question)) return;
    try {
      await request();
      if (next && location.hash !== next) location.hash = next; else route();
    } catch (error) { alert(`${label}失败：${error.message}`); }
  } }, label);
}

const CASCADEABLE = new Set(["training_runs", "evaluations", "tasks"]);

// MODEL_IN_USE only by ended runs, evaluations and trials, which ?cascade_training_runs=true deletes too.
function cascadeable(error) {
  const references = error.detail?.references;
  return error.code === "MODEL_IN_USE" && references && Object.keys(references).every((key) => CASCADEABLE.has(key));
}

function referenceText({ references, training_run_ids: ids = [], evaluation_ids: evaluationIds = [] }) {
  const runText = references.training_runs ? `${references.training_runs} 个训练实验` : "";
  const evalText = references.evaluations ? `${references.evaluations} 个评测` : "";
  const taskText = references.tasks ? `${references.tasks} 个 trial` : "";
  const related = [runText, evalText, taskText].filter(Boolean).join("、");
  const listedRuns = ids.length ? `训练实验：${ids.map(fmt.short).join("、")}` : "";
  const listedEvaluations = evaluationIds.length ? `评测：${evaluationIds.map(fmt.short).join("、")}` : "";
  const listed = [listedRuns, listedEvaluations].filter(Boolean).join("；");
  return `被 ${related} 引用${listed ? `（${listed}）` : ""}`;
}

// Deletes one version at a time (cascades lock shared runs): one confirmation up front, one more listing
// versions whose ended runs / evaluations / trials go with them. A version still loaded in the model
// service is taken offline first, and loaded again if it ends up kept.
async function deleteModels(versions, { what, runId = null, loaded = [] }) {
  if (modelUi.busy) return;
  try {
    const run = runId && (await api("/api/v1/training/runs")).find((r) => r.run_id === runId);
    if (run && ["QUEUED", "RUNNING"].includes(run.status)) {
      alert(`实验 ${fmt.short(runId)} 仍在运行，删除它的版本会中断训练；请先在训练实验页取消。`);
      return;
    }
    // Taking a loaded version offline would fail an evaluation that is still sampling from it.
    const busy = loaded.length ? (await api("/api/v1/evaluations")).filter((ev) => ev.status === "RUNNING"
      && (ev.model_versions || []).some((version) => loaded.includes(version))) : [];
    if (busy.length) {
      alert(`评测 ${busy.map((ev) => fmt.short(ev.eval_id)).join("、")} 正在使用这些版本；请等评测结束或先取消。`);
      return;
    }
  } catch (error) { alert(`删除失败：${error.message}`); return; }
  const unload = loaded.length ? `\n其中 ${loaded.join("、")} 已加载在模型服务中，会先下线。` : "";
  if (!confirm(`删除${what}并清理受控 checkpoint？${unload}`)) return;
  modelUi.busy = true;
  const inUse = [], failed = [], offlined = new Set(), deleted = new Set();
  const remove = async (version, query = "") => {
    const path = `/api/v1/model-versions/${encodeURIComponent(version)}`;
    try {
      await api(`${path}${query}`, { method: "DELETE" });
    } catch (error) {
      if (error.code !== "MODEL_LOADED") throw error;
      await api(`${path}/offline`, { method: "POST" });
      offlined.add(version);
      await api(`${path}${query}`, { method: "DELETE" });
    }
    deleted.add(version);
  };
  try {
    for (const version of versions) {
      try { await remove(version); } catch (error) { (cascadeable(error) ? inUse : failed).push([version, error]); }
    }
    const listed = inUse.map(([version, error]) => `${version} ${referenceText(error.detail)}`).join("\n");
    if (inUse.length && confirm(`${inUse.length} 个版本仍有引用：\n${listed}\n\n确认删除关联的已结束训练实验、评测、trial 和已结束任务及其指标和日志，并继续删除吗？`)) {
      for (const [version] of inUse) {
        try { await remove(version, "?cascade_training_runs=true"); } catch (error) { failed.push([version, error]); }
      }
    }
    for (const version of offlined) {
      if (!deleted.has(version)) await api(`/api/v1/model-versions/${encodeURIComponent(version)}/online`, { method: "POST" }).catch(() => {});
    }
  } finally { modelUi.busy = false; }
  if (failed.length) alert(`${failed.length} 个版本删除失败：\n${failed.map(([version, error]) => `${version}：${error.detail?.message || error.message}`).join("\n")}`);
  route();
}

function runActions(run, next) {
  const path = `/api/v1/training/runs/${encodeURIComponent(run.run_id)}`;
  return ["QUEUED", "RUNNING"].includes(run.status)
    ? actionButton("取消", "取消该实验？排队中和运行中的 rollout 也会被取消。", () => api(`${path}/cancel`, { method: "POST" }))
    : actionButton("删除", "删除该实验及其 rollout 轨迹和指标？已注册的模型 checkpoint 和评测记录会保留。", () => api(path, { method: "DELETE" }), next);
}

function evalActions(ev, next) {
  const path = `/api/v1/evaluations/${encodeURIComponent(ev.eval_id)}`;
  return ev.status === "RUNNING"
    ? actionButton("取消", "取消该评测？未完成的 trial 会被取消，已完成的结果保留。", () => api(`${path}/cancel`, { method: "POST" }))
    : actionButton("删除", "删除该评测及其全部 trial 轨迹？", () => api(path, { method: "DELETE" }), next);
}

function showError(error) {
  view.replaceChildren(h("div", { class: "card error-text" }, `加载失败：${error.message}`));
}

function idempotencyKey() {
  return crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
}

// ---------------------------------------------------------------- charts

// Single-series line chart with crosshair tooltip. Small multiples keep one y-axis per chart.
function lineChart(title, points, { format = (v) => fmt.num(v, 4), xLabel = "step" } = {}) {
  const width = 360, height = 180, pad = { l: 46, r: 12, t: 8, b: 22 };
  const wrap = h("div", { class: "card chart" });
  const last = points.length ? points[points.length - 1] : null;
  wrap.append(h("div", { class: "row" }, h("div", { class: "title" }, title), h("div", { class: "spacer" }),
    last ? h("div", { class: "last" }, `最新 ${format(last[1])}`) : null));
  if (!points.length) { wrap.append(h("div", { class: "muted" }, "暂无数据")); return wrap; }
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  let [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  if (x0 === x1) { x0 -= 1; x1 += 1; }
  if (y0 === y1) { const d = Math.abs(y0) * 0.1 || 1; y0 -= d; y1 += d; }
  const sx = (x) => pad.l + ((x - x0) / (x1 - x0)) * (width - pad.l - pad.r);
  const sy = (y) => height - pad.b - ((y - y0) / (y1 - y0)) * (height - pad.t - pad.b);
  const svg = h("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${title} 折线图` });
  for (let i = 0; i <= 3; i += 1) {
    const value = y0 + ((y1 - y0) * i) / 3;
    svg.append(h("line", { x1: pad.l, x2: width - pad.r, y1: sy(value), y2: sy(value), stroke: "var(--grid)", "stroke-width": 1 }));
    svg.append(h("text", { x: pad.l - 6, y: sy(value) + 4, "text-anchor": "end", class: "tick" }, format(value)));
  }
  svg.append(h("text", { x: pad.l, y: height - 4, class: "tick" }, `${xLabel} ${x0}`));
  svg.append(h("text", { x: width - pad.r, y: height - 4, "text-anchor": "end", class: "tick" }, String(x1)));
  const d = points.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
  svg.append(h("path", { d, fill: "none", stroke: "var(--series-1)", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
  if (points.length <= 30) {
    for (const p of points) svg.append(h("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 4, fill: "var(--series-1)", stroke: "var(--surface-1)", "stroke-width": 2 }));
  }
  const cross = h("line", { y1: pad.t, y2: height - pad.b, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" });
  const dot = h("circle", { r: 5, fill: "var(--series-1)", stroke: "var(--surface-1)", "stroke-width": 2, visibility: "hidden" });
  svg.append(cross, dot);
  const tip = h("div", { class: "tooltip" });
  const hit = h("rect", { x: pad.l, y: 0, width: width - pad.l - pad.r, height, fill: "transparent" });
  const hide = () => { cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); tip.style.display = "none"; };
  hit.addEventListener("pointermove", (event) => {
    const box = svg.getBoundingClientRect();
    const x = ((event.clientX - box.left) / box.width) * width;
    let best = points[0];
    for (const p of points) if (Math.abs(sx(p[0]) - x) < Math.abs(sx(best[0]) - x)) best = p;
    cross.setAttribute("x1", sx(best[0])); cross.setAttribute("x2", sx(best[0])); cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", sx(best[0])); dot.setAttribute("cy", sy(best[1])); dot.setAttribute("visibility", "visible");
    tip.replaceChildren(h("strong", {}, format(best[1])), " ", h("span", { class: "muted" }, `${xLabel} ${best[0]}`));
    tip.style.display = "block";
    tip.style.left = `${(sx(best[0]) / width) * box.width + 8}px`;
    tip.style.top = `${(sy(best[1]) / height) * box.height + 20}px`;
  });
  hit.addEventListener("pointerleave", hide);
  svg.append(hit);
  wrap.append(svg, tip);
  return wrap;
}

// Horizontal bars, one per model; one color because bars differ by category, not by series.
function rateBars(rows) {
  const width = 520, rowH = 30, labelW = 210, height = rows.length * rowH + 6;
  const svg = h("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "各模型任务成功率", style: `height:${height}px` });
  rows.forEach((row, i) => {
    const y = i * rowH + 4;
    const barW = Math.max(0, (row.value ?? 0) * (width - labelW - 60));
    svg.append(h("text", { x: 0, y: y + 16, class: "tick", style: "fill:var(--text-secondary);font-size:12px" }, fmt.short(row.label)));
    svg.append(h("rect", { x: labelW, y: y + 4, width: width - labelW - 60, height: 16, rx: 4, fill: "var(--grid)" }));
    if (barW > 0) svg.append(h("rect", { x: labelW, y: y + 4, width: barW, height: 16, rx: 4, fill: "var(--series-1)" }));
    svg.append(h("text", { x: labelW + barW + 6, y: y + 16, class: "tick", style: "fill:var(--text-primary);font-size:12px" }, row.text));
  });
  return h("div", { class: "chart" }, svg);
}

function actionOverlay(action) {
  if (!action || !["tap", "swipe"].includes(action.type)) return null;
  const svg = h("svg", { viewBox: "0 0 720 1280", preserveAspectRatio: "none" });
  if (action.type === "tap") {
    svg.append(h("circle", { cx: action.x, cy: action.y, r: 34, fill: "none", stroke: "#fff", "stroke-width": 10 }));
    svg.append(h("circle", { cx: action.x, cy: action.y, r: 34, fill: "rgba(208,59,59,0.25)", stroke: "#d03b3b", "stroke-width": 6 }));
  } else {
    svg.append(h("line", { x1: action.x1, y1: action.y1, x2: action.x2, y2: action.y2, stroke: "#fff", "stroke-width": 14, "stroke-linecap": "round" }));
    svg.append(h("line", { x1: action.x1, y1: action.y1, x2: action.x2, y2: action.y2, stroke: "#d03b3b", "stroke-width": 8, "stroke-linecap": "round" }));
    svg.append(h("circle", { cx: action.x2, cy: action.y2, r: 20, fill: "#d03b3b" }));
  }
  return svg;
}

function describeAction(action) {
  if (!action) return "–";
  switch (action.type) {
    case "tap": return `tap (${action.x}, ${action.y})`;
    case "swipe": return `swipe (${action.x1}, ${action.y1}) → (${action.x2}, ${action.y2})`;
    case "text": return `text "${action.text}"`;
    case "wait": return `wait ${action.duration_ms} ms`;
    case "terminate": return `terminate: ${action.reason}`;
    default: return action.type;
  }
}

// ---------------------------------------------------------------- resources

async function resourcesPage() {
  const root = h("div");
  view.replaceChildren(pageHeading("SYSTEM OVERVIEW / 01", "资源总览", "实时查看节点心跳、GPU / KVM 能力、队列压力和当前 sandbox。"), root);
  await poll(async () => {
    if (root.contains(document.activeElement) && ["SELECT", "INPUT"].includes(document.activeElement.tagName)) return;
    const [data, catalog] = await Promise.all([
      api("/api/v1/overview"),
      api("/api/v1/catalog"),
    ]);
    const counts = data.task_status_counts;
    const active = ["ALLOCATING", "RUNNING_ENV", "WAITING_MODEL"].reduce((n, s) => n + (counts[s] || 0), 0);
    const model = data.model_service;
    const queue = data.queue;
    root.replaceChildren(
      h("div", { class: "grid" },
        tile("运行中任务", active, `等待模型 ${counts.WAITING_MODEL || 0}`),
        tile("排队任务", counts.QUEUED || 0, queue.available ? `Stream ${queue.length} 条，未确认 ${queue.pending}` : "Redis 不可用"),
        tile("累计成功 / 失败", `${counts.SUCCEEDED || 0} / ${counts.FAILED || 0}`, `已取消 ${counts.CANCELED || 0}`),
        h("div", { class: "card tile" },
          h("div", { class: "label" }, "模型服务"),
          h("div", { class: "value" }, badge(model.healthy ? "READY" : "UNREACHABLE")),
          h("div", { class: "sub" }, model.backend === "http"
            ? `服务已加载 ${(model.loaded_versions || []).join(", ") || "无"} · p50 ${fmt.ms(model.latency_p50_ms)} · 拒绝 ${model.rejected_outputs ?? "–"}`
            : `后端 ${model.backend}（不使用 GPU 模型服务）`),
          model.error ? h("div", { class: "sub error-text" }, model.error) : null)),
      h("h3", {}, "节点"),
      h("div", { class: "card" }, nodesTable(data.nodes)),
      h("h3", {}, "Sandbox"),
      h("div", { class: "card" }, sandboxTable(data.sandboxes)),
      modelSections(catalog.model_versions, model),
    );
  }, 5000);
}

// Resources-page state that survives the 5 s re-render: expanded run groups, and a delete in progress.
const modelUi = { expanded: new Set(), busy: false };

// Models: one formal model per training run (plus imported and merged ones); the rest of a run's
// versions are its checkpoints, and any of them can be switched in as the run's model.
// LoRA and Checkpoint fold each run's versions under one row that can delete them all.
function modelSections(versions, service) {
  const byRun = new Map();
  for (const m of versions) if (m.run_id) byRun.set(m.run_id, [...(byRun.get(m.run_id) || []), m]);
  const models = versions.filter((m) => m.kind !== "checkpoint");
  const box = h("div");
  const section = (title, hint, rows, options = {}) => [
    h("h3", {}, title, h("span", { class: "hint" }, hint)),
    h("div", { class: "card" }, modelTable(rows, service, byRun, { noun: title, ...options, rerender: render })),
  ];
  const render = () => box.replaceChildren(
    ...section("模型", "每个训练实验只有一个正式模型，可在 Checkpoint 列切换", models.filter((m) => !m.adapter)),
    ...section("LoRA", "只含 adapter 权重；在线策略算法 和全参数训练需先合并", models.filter((m) => m.adapter), { grouped: true }),
    ...section("Checkpoint", "训练中按保存间隔保留的中间模型；可评测，设为正式模型后进入上方模型 / LoRA 栏",
      versions.filter((m) => m.kind === "checkpoint"), { checkpoints: true, grouped: true }),
  );
  render();
  return box;
}

function checkpointName(m) {
  return m.checkpoint_uri ? m.checkpoint_uri.split("/").pop() || m.checkpoint_uri : null;
}

function promote(m) {
  return api(`/api/v1/model-versions/${encodeURIComponent(m.version)}/promote`, { method: "POST" });
}

// A run's model row picks which of the run's versions is formal; other rows just show the directory.
// The pick shows the short directory name, underlined like the concurrency values; a click opens the full list.
function checkpointCell(m, siblings) {
  const live = (siblings || []).filter((s) => s.status !== "RETIRED");
  const name = checkpointName(m);
  if (live.length < 2) {
    return name ? h("span", { class: "mono", title: m.checkpoint_uri }, fmt.short(name)) : h("span", { class: "muted" }, "无 checkpoint");
  }
  const value = h("button", { class: "stepper-value pick mono", title: `${m.checkpoint_uri || m.version}\n点击切换该实验的正式模型`,
    onclick: (event) => { event.stopPropagation(); pick(); } }, fmt.short(name || m.version));
  const pick = () => {
    let open = true;
    const close = () => { if (open) { open = false; select.replaceWith(value); } }; // removal can fire blur again
    const select = h("select", { onclick: (event) => event.stopPropagation(), onblur: close,
      onkeydown: (event) => { if (event.key === "Escape") close(); } },
      live.map((s) => h("option", { value: s.version, selected: s.version === m.version }, `${checkpointName(s)} · ${s.version}`)));
    select.addEventListener("change", async () => {
      const chosen = live.find((s) => s.version === select.value);
      close();
      if (!confirm(`把实验 ${fmt.short(m.run_id)} 的正式模型切换为 ${chosen.version}？${m.version} 会移到 Checkpoint 栏。`)) return;
      try { await promote(chosen); route(); } catch (error) { alert(`切换失败：${error.message}`); }
    });
    value.replaceWith(select);
    select.focus();
    try { select.showPicker(); } catch { /* No showPicker: the list is focused, one more click opens it. */ }
  };
  return h("span", { class: "stepper" }, value);
}

function modelTable(models, service, byRun, { checkpoints = false, grouped = false, noun = "模型", rerender = null } = {}) {
  const loaded = new Set(service.loaded_versions || []);
  const offline = new Set(service.offline_versions || []);
  const remove = (versions, what, runId) => deleteModels(versions, { what, runId, loaded: versions.filter((v) => loaded.has(v)) });
  const columns = [
    { label: "版本", render: (m) => h("div", { class: "row" }, h("span", { class: "mono" }, m.version),
      m.adapter ? h("span", { class: "badge warning", title: ADAPTER_HINT }, "LoRA adapter") : null) },
    { label: "基础模型", render: (m) => m.base_model || "–" },
    { label: "来源", render: (m) => (m.run_id ? h("a", { href: `#/runs/${m.run_id}`, onclick: (event) => event.stopPropagation() }, `实验 ${fmt.short(m.run_id)}`)
      : m.metrics?.merged_from ? `合并自 ${m.metrics.merged_from}` : "导入") },
    { label: "注册状态", render: (m) => modelBadge(m.status) },
    { label: "运行状态", render: (m) => service.backend === "http"
      ? (loaded.has(m.version) ? badge("READY")
        : h("span", { class: "muted" }, offline.has(m.version) ? "已下线" : grouped ? "未加载" : "未上线"))
      : h("span", { class: "muted" }, "不适用") },
    { label: "Checkpoint", render: (m) => checkpointCell(m, checkpoints ? null : byRun.get(m.run_id)) },
    { label: "", render: (m) => {
      const buttons = [];
      if (checkpoints && m.status !== "RETIRED") buttons.push(actionButton(
        "设为正式模型", `把 ${m.version} 设为实验 ${fmt.short(m.run_id)} 的正式模型？原正式模型会移到 Checkpoint 栏。`,
        () => promote(m), undefined, "primary"));
      // LoRA / Checkpoint rows have no 上线 / 下线: evaluations load them on demand, and 删除 takes a
      // loaded version offline first.
      if (service.backend === "http" && m.status !== "RETIRED" && !grouped) {
        const isOnline = loaded.has(m.version);
        buttons.push(actionButton(
          isOnline ? "下线" : "上线",
          isOnline ? `下线模型 ${m.version}？下线后新的推理请求将无法使用它。` : `上线模型 ${m.version}？这会加载 checkpoint 并占用模型服务显存。`,
          () => api(`/api/v1/model-versions/${encodeURIComponent(m.version)}/${isOnline ? "offline" : "online"}`, { method: "POST" }),
          undefined,
          isOnline ? "danger" : "primary",
        ));
      }
      if (service.backend === "http" && m.adapter && !checkpoints && m.status !== "RETIRED") {
        buttons.push(h("button", { title: ADAPTER_HINT, onclick: (event) => { event.stopPropagation(); mergeModel(m); } }, "合并完整权重"));
      }
      if (!m.protected) buttons.push(h("button", { class: "danger", onclick: (event) => {
        event.stopPropagation();
        remove([m.version], `${noun} ${m.version}`, m.run_id);
      } }, "删除"));
      return h("div", { class: "row" }, buttons);
    } },
  ];
  if (!grouped) return table(columns, models);
  const deleteAll = (rows) => {
    const versions = rows.filter((m) => !m.protected).map((m) => m.version);
    return versions.length ? h("button", { class: "danger", onclick: (event) => {
      event.stopPropagation();
      remove(versions, `实验 ${fmt.short(rows[0].run_id)} 的全部 ${versions.length} 个 ${noun}`, rows[0].run_id);
    } }, "全部删除") : null;
  };
  return groupedTable(columns, models, rerender, deleteAll);
}

// Each training run's versions fold under one header row (collapsed by default) with a delete-all button.
function groupedTable(columns, models, rerender, deleteAll) {
  const row = (m, child = false) => h("tr", { class: child ? "group-child" : "" }, columns.map((c) => h("td", {}, c.render(m))));
  const groups = new Map();
  for (const m of models) {
    const key = m.run_id ? `run:${m.run_id}` : `version:${m.version}`;
    groups.set(key, [...(groups.get(key) || []), m]);
  }
  const body = [...groups].flatMap(([key, rows]) => {
    if (!rows[0].run_id) return [row(rows[0])];
    const open = modelUi.expanded.has(key);
    const names = rows.map(checkpointName).filter(Boolean);
    const range = names.length > 1 ? ` · ${names[0]} … ${names[names.length - 1]}` : names.length ? ` · ${names[0]}` : "";
    const toggle = () => {
      if (open) modelUi.expanded.delete(key); else modelUi.expanded.add(key);
      rerender();
    };
    return [
      h("tr", { class: "clickable group", onclick: toggle },
        h("td", { colSpan: columns.length - 1 }, h("div", { class: "row" },
          // Enter / Space click the button; the click bubbles to the row's toggle.
          h("button", { class: "caret", "aria-expanded": String(open), "aria-label": open ? "折叠" : "展开" }, open ? "▾" : "▸"),
          h("a", { href: `#/runs/${rows[0].run_id}`, onclick: (event) => event.stopPropagation() }, `实验 ${fmt.short(rows[0].run_id)}`),
          h("span", { class: "muted" }, `${rows.length} 个版本${range}`))),
        h("td", {}, deleteAll(rows))),
      ...(open ? rows.map((m) => row(m, true)) : []),
    ];
  });
  return h("table", {},
    h("thead", {}, h("tr", {}, columns.map((c) => h("th", {}, c.label)))),
    h("tbody", {}, body.length ? body : h("tr", {}, h("td", { colSpan: columns.length, class: "muted" }, "暂无数据"))));
}

async function mergeModel(m) {
  const target = prompt(
    `把 ${m.version} 的 LoRA adapter 合并进基础模型，注册为新的完整权重模型。\n`
    + "合并在 4090 模型服务的 CPU 上进行，约需 1–3 分钟，不影响正在进行的推理。\n"
    + "BF16 合并可能改变输出，用于训练或评测前请先与原 adapter 对比 validation 输出。\n\n新模型名：",
    `${m.version}-merged`);
  if (target === null) return;
  try {
    const merged = await api(`/api/v1/model-versions/${encodeURIComponent(m.version)}/merge`, {
      method: "POST", body: JSON.stringify({ target_version: target.trim() }) });
    alert(`已注册 ${merged.version}\n${merged.checkpoint_uri}`);
    route();
  } catch (error) { alert(`合并失败：${error.message}`); }
}

function nodesTable(nodes) {
  return table([
    { label: "节点", render: (n) => h("div", {}, h("div", {}, n.node_id), h("div", { class: "muted" }, n.address || "")) },
    { label: "角色", render: (n) => (n.role === "gpu" ? "GPU 训练/推理" : "Android microVM") },
    { label: "状态", render: (n) => badge(n.status) },
    { label: "资源", render: (n) => (n.role === "gpu" ? gpuCell(n.resources) : hostCell(n.resources)) },
    { label: "Sandbox", num: true, render: (n) => (n.role === "gpu" ? (n.resources.active_run ? fmt.short(n.resources.active_run) : "空闲") : `${n.active_sandboxes} / ${n.capacity.sandboxes ?? "?"}`) },
    { label: "并发设置", render: concurrencyCell },
    { label: "最近心跳", render: (n) => fmt.time(n.last_heartbeat_at) },
  ], nodes);
}

const CONCURRENCY = { sandboxes: "Sandbox", training_runs: "训练", evaluations: "评测推理（每个模型一个 Pod）" };
const CEILING_PARTS = { platform: "平台", cpu: "CPU", memory: "内存", network: "网络地址" };

// Why the ceiling is what it is, so the user knows what raising it would take.
function ceilingReason(node, key) {
  const parts = node.resources.sandbox_ceiling;
  if (key === "sandboxes" && parts) {
    const known = Object.entries(parts).filter(([, v]) => v !== null && v !== undefined);
    return `按每个 sandbox 的配额估算：${known.map(([k, v]) => `${CEILING_PARTS[k] || k} ${v}`).join("、")}，取最小值`;
  }
  return "由节点启动参数设定（H800 为 launcher 的 --max-runs / --max-serving，队列配额无法读取）";
}

function concurrencyCell(node) {
  const keys = Object.keys(CONCURRENCY).filter((key) => node.capacity[key] !== undefined);
  if (!keys.length) return h("span", { class: "muted" }, "–");
  return h("div", {}, keys.map((key) => {
    const max = node.capacity[`max_${key}`];
    const label = CONCURRENCY[key];
    if (typeof max !== "number") return h("div", { class: "secondary" }, `${label} ${node.capacity[key]}（不可调）`);
    return h("div", { class: "row", title: `${ceilingReason(node, key)}。\n调低不会中断运行中的任务，只是暂停接收新任务；约 10 秒内生效。` },
      h("span", { class: "stepper" }, label, concurrencyStepper(node, key, max)),
      h("span", { class: "muted" }, `上限 ${max}`));
  }));
}

// The current value; clicking it opens a number input (Enter/blur saves, Esc cancels).
function concurrencyStepper(node, key, max) {
  let value = node.capacity[key];
  const num = h("button", { class: "stepper-value", title: "点击修改", onclick: () => edit() }, value);
  const set = async (next) => {
    if (next === value) return;
    if (!Number.isInteger(next) || next < 1 || next > max) { alert(`请输入 1–${max} 的整数`); return; }
    num.disabled = true; // one request at a time so changes can't land out of order
    try {
      const result = await api(`/api/v1/nodes/${encodeURIComponent(node.node_id)}/limits`, {
        method: "PUT", body: JSON.stringify({ [key]: next }) });
      value = result.capacity[key];
      num.textContent = value;
    } catch (error) { alert(`调整失败：${error.message}`); }
    num.disabled = false;
  };
  const edit = () => {
    let done = false;
    const finish = (commit) => {
      if (done) return; // Enter removes the input, which can fire blur again
      done = true;
      input.replaceWith(num);
      if (commit && input.value !== "") set(Number(input.value));
    };
    const input = h("input", { type: "number", min: 1, max, value, class: "stepper-input",
      onkeydown: (event) => { if (event.key === "Enter") finish(true); else if (event.key === "Escape") finish(false); },
      onblur: () => finish(true) });
    num.replaceWith(input);
    input.focus();
    input.select();
  };
  return num;
}

function meter(fraction) {
  return h("div", { class: "meter", role: "meter", "aria-valuenow": Math.round(fraction * 100), "aria-valuemin": 0, "aria-valuemax": 100 },
    h("div", { style: `width:${Math.min(100, Math.max(0, fraction * 100)).toFixed(0)}%` }));
}

function hostCell(r) {
  const memUsed = r.memory_total_bytes - r.memory_available_bytes;
  const diskUsed = r.disk_total_bytes - r.disk_free_bytes;
  return h("div", { style: "min-width:260px" },
    h("div", {}, `CPU ${r.cpu_count} 核 · load ${fmt.num(r.load_1m, 1)} · KVM ${r.kvm_available ? "可用" : "不可用"} · ${r.runtime_backend}`),
    h("div", { class: "secondary" }, `内存 ${fmt.bytes(memUsed)} / ${fmt.bytes(r.memory_total_bytes)}`), meter(memUsed / r.memory_total_bytes),
    h("div", { class: "secondary" }, `磁盘 ${fmt.bytes(diskUsed)} / ${fmt.bytes(r.disk_total_bytes)}`), meter(diskUsed / r.disk_total_bytes));
}

function gpuCell(r) {
  if (!r.gpus?.length) return h("span", { class: "muted" }, "未上报 GPU");
  return h("div", { style: "min-width:260px" }, r.gpus.map((g) => h("div", {},
    h("div", {}, `GPU${g.index} ${g.name} · ${g.temperature_c}°C`),
    h("div", { class: "secondary" }, `利用率 ${g.utilization_percent}%`), meter(g.utilization_percent / 100),
    h("div", { class: "secondary" }, `显存 ${fmt.num(g.memory_used_mib / 1024, 1)} / ${fmt.num(g.memory_total_mib / 1024, 1)} GiB`),
    meter(g.memory_used_mib / g.memory_total_mib))));
}

function sandboxTable(rows) {
  return table([
    { label: "截图", render: (s) => (s.latest_step !== null
      ? h("img", { class: "thumb", src: `/api/v1/sandboxes/${encodeURIComponent(s.sandbox_id)}/screenshot?step=${s.latest_step}`, alt: "最新截图" })
      : h("div", { class: "thumb" })) },
    { label: "Sandbox", render: (s) => h("div", {}, h("div", { class: "mono" }, s.sandbox_id), h("div", { class: "muted" }, s.node_id || "")) },
    { label: "VM 状态", render: (s) => h("div", {}, s.state, s.healthy ? "" : h("span", { class: "error-text" }, " 不健康")) },
    { label: "任务", render: (s) => h("div", {}, badge(s.task_status), " ", h("a", { href: `#/episodes/${s.episode_id}` }, `${s.task_type} · ${s.policy_version}`),
      h("div", { class: "muted" }, s.eval_id ? `评测 ${fmt.short(s.eval_id)}` : s.run_id ? `训练 ${fmt.short(s.run_id)}` : "手动任务")) },
    { label: "步数", num: true, render: (s) => `${s.current_step} / ${s.max_steps}` },
    { label: "Lease 剩余", num: true, render: (s) => (s.lease_id ? `${s.lease_remaining_seconds}s` : "–") },
    { label: "", render: (s) => h("button", { class: "danger", onclick: async (event) => {
      event.stopPropagation();
      if (!confirm(`销毁 ${s.sandbox_id}？对应任务会被取消。`)) return;
      try { await api(`/api/v1/sandboxes/${encodeURIComponent(s.sandbox_id)}/destroy`, { method: "POST" }); } catch (e) { alert(e.message); }
    } }, "销毁") },
  ], rows);
}

// ---------------------------------------------------------------- training runs

const METRIC_LABELS = {
  "rollout/reward_mean": ["每轮 reward 均值", "round"],
  "rollout/success_rate": ["每轮任务成功率", "round", fmt.pct],
  "rollout/invalid_output_rate": ["每轮非法输出率", "round", fmt.pct],
  "rollout/reward_std": ["每轮 reward 标准差", "round"],
  "rollout/mean_steps": ["每轮平均步数", "round"],
  "rollout/zero_variance_groups": ["零方差组数（无梯度）", "round"],
  "rollout/entropy": ["每轮采样熵 nats/token（按采样 temperature / top-k / top-p）", "round", (v) => fmt.num(v, 3)],
  "train/loss": ["loss", "update"],
  "train/entropy": ["策略熵 nats/token（训练样本，temperature 1）", "update", (v) => fmt.num(v, 3)],
  "train/grad_norm": ["梯度范数", "update"],
  "train/ratio": ["策略比 ratio", "update"],
  "train/clipped": ["是否被 clip", "update"],
  "train/advantage": ["advantage", "update"],
  "train/learning_rate": ["学习率", "update", (v) => Number(v).toExponential(1)],
  "gpu/utilization_percent": ["GPU 利用率 %", "秒", (v) => fmt.num(v, 0)],
  "gpu/memory_used_mib": ["显存 MiB", "秒", (v) => fmt.num(v, 0)],
};

const ALGORITHM_LABELS = { sft: "SFT", grpo: "GRPO", dapo: "DAPO", dr_grpo: "Dr. GRPO", gigpo: "GiGPO" };
function algorithmLabel(algorithm) {
  return ALGORITHM_LABELS[algorithm] || algorithm.toUpperCase();
}

// Config knobs each algorithm consumes; the form shows and submits only these.
const SHARED_KNOBS = ["accelerator", "learning_rate", "epochs", "gradient_accumulation", "batch_size", "cache_preprocessed", "gradient_checkpointing", "max_steps", "checkpoint_every", "eval_trials", "seed"];
// Policy-algorithm components (training_agent.ALGORITHM_PRESETS); blank keeps the algorithm's preset.
const COMPONENT_KNOBS = ["advantage_std", "drop_zero_variance_groups", "clip_epsilon_high", "overlong", "overlong_penalty",
  "loss_normalization", "loss_normalization_constant", "step_advantage_omega", "step_advantage_gamma", "invalid_output_reward"];
const ALGORITHM_KNOBS = {
  sft: ["finetuning_method", "train_merger_mlp", "multiview", "output_format", ...SHARED_KNOBS],
  grpo: ["finetuning_method", "rounds", "tasks_per_round", "group_size", "temperature", "clip_epsilon", ...SHARED_KNOBS, ...COMPONENT_KNOBS],
};
ALGORITHM_KNOBS.dapo = ALGORITHM_KNOBS.dr_grpo = ALGORITHM_KNOBS.gigpo = ALGORITHM_KNOBS.grpo;
// Shown next to each component: what blank means for the chosen algorithm. Mirrors ALGORITHM_PRESETS.
const PRESET_LABELS = {
  advantage_std: { grpo: "除", dapo: "除", dr_grpo: "不除", gigpo: "不除" },
  drop_zero_variance_groups: { grpo: "不舍弃", dapo: "舍弃", dr_grpo: "不舍弃", gigpo: "不舍弃" },
  clip_epsilon_high: { grpo: "同 clip ε", dapo: "0.28", dr_grpo: "同 clip ε", gigpo: "同 clip ε" },
  overlong: { grpo: "当作失败", dapo: "屏蔽", dr_grpo: "当作失败", gigpo: "当作失败" },
  loss_normalization: { grpo: "每条轨迹步数", dapo: "整批总步数", dr_grpo: "常数", gigpo: "整批总步数" },
  step_advantage_omega: { grpo: "0（关）", dapo: "0（关）", dr_grpo: "0（关）", gigpo: "1" },
};
const presetSelect = (options) => h("select", {}, h("option", { value: "" }, "按算法默认"),
  options.map(([value, label]) => h("option", { value }, label)));

// Set by "复用配置" on a run detail page and consumed by the next runs page render.
let pendingRecipe = null;

async function runsPage() {
  const catalog = await api("/api/v1/catalog");
  const list = h("div", { class: "card" });
  const form = runForm(catalog);
  view.replaceChildren(
    pageHeading("TRAINING LAB / 03", "训练实验", "配置在线 rollout、选择 runtime，并跟踪 reward、梯度和产出 checkpoint。"),
    form,
    h("h3", {}, "实验列表"),
    list,
  );
  if (pendingRecipe) { form.load(pendingRecipe); pendingRecipe = null; }
  await poll(async () => {
    const runs = await api("/api/v1/training/runs");
    list.replaceChildren(table([
      { label: "实验", render: (r) => h("a", { href: `#/runs/${r.run_id}` }, fmt.short(r.run_id)) },
      { label: "算法", render: (r) => algorithmLabel(r.algorithm) },
      { label: "Runtime", render: (r) => r.runtime_backend || "unknown" },
      { label: "基础模型", key: "base_model" },
      { label: "数据", render: (r) => runData(r) || "–" },
      { label: "状态", render: (r) => badge(r.status) },
      { label: "reward 均值", num: true, render: (r) => fmt.num(r.metrics["rollout/reward_mean"] ?? r.metrics.reward_mean) },
      { label: "产出模型", render: (r) => r.metrics.model_version || "–" },
      { label: "创建时间", render: (r) => fmt.time(r.created_at) },
      { label: "", render: (r) => runActions(r) },
    ], runs, (r) => { location.hash = `#/runs/${r.run_id}`; }));
  }, 5000);
}

function runData(run) {
  return run.dataset_version || run.config.train_set || (run.config.tasks || []).map((t) => t.app_image).join(", ");
}

function taskSetLabel(set) {
  return `${set.label}（${set.tasks.length} 题）`;
}

// An evaluation's tasks, short: the task set's name when they are exactly one set, else the first few and a count.
// The full list stays available as the hover title.
function taskSummary(tasks, taskSets = [], shown = 3) {
  const images = tasks.map((t) => t.app_image);
  const full = images.join(", ");
  const set = taskSets.find((s) => s.tasks.length === images.length && s.tasks.every((t, i) => t.app_image === images[i]));
  // Tasks of one app share its image name; show it once ("android-sokoban-v1: train-000, train-001").
  const apps = new Set(images.map((image) => image.split(":")[0]));
  const app = apps.size === 1 && images[0].includes(":") ? [...apps][0] : null;
  const names = app ? images.map((image) => image.slice(app.length + 1)) : images;
  const listed = names.length <= shown ? names.join(", ") : `${names.slice(0, shown).join(", ")} … 共 ${names.length} 题`;
  const text = set ? taskSetLabel(set) : app ? `${app}: ${listed}` : listed;
  return h("span", { title: full }, text);
}

function field(label, input, hint) {
  return h("label", {}, label, input, hint ? h("span", { class: "hint" }, hint) : null);
}

function runtimeLabel(backend) {
  return backend === "docker" ? "Docker · Android container" : "Firecracker · Android microVM";
}

function runForm(catalog) {
  const models = catalog.model_versions.filter((model) => model.status !== "RETIRED");
  const runtimeBackends = catalog.runtime_backends || [catalog.runtime_backend || "firecracker"];
  const defaultBackend = catalog.runtime_backend || runtimeBackends[0];
  const message = h("div");
  const onlineSets = h("optgroup", { label: "在线题目集" }, catalog.task_sets.map((s) => h(
    "option", { value: `set:${s.name}`, selected: s.name === "sokoban-train" }, taskSetLabel(s))));
  const inputs = {
    algorithm: h("select", {},
      h("option", { value: "grpo" }, "GRPO（在线 rollout）"),
      h("option", { value: "dapo" }, "DAPO（动态采样 · clip-higher · 截断过滤）"),
      h("option", { value: "dr_grpo", selected: true }, "Dr. GRPO（去掉 std 与长度归一）"),
      h("option", { value: "gigpo" }, "GiGPO（episode + 相同状态步骤级优势）"),
      h("option", { value: "sft" }, "SFT（离线数据集）")),
    accelerator: h("select", {}, h("option", { value: "h800" }, "H800（每个实验独立 Pod，可并行）"), h("option", { value: "4090" }, "4090（共用一张卡，一次一个实验）")),
    finetuning_method: h("select", {}, h("option", { value: "full" }, "全参数"), h("option", { value: "lora" }, "LoRA（FP32 adapter）")),
    train_merger_mlp: h("input", { type: "checkbox" }),
    multiview: h("input", { type: "checkbox" }),
    output_format: h("select", {}, h("option", { value: "action-v1" }, "动作 JSON"), h("option", { value: "sokoban-plan-v1" }, "推箱子：棋盘 / 决策 / 动作")),
    runtime_backend: h("select", {}, runtimeBackends.map((backend) => h(
      "option", { value: backend, selected: backend === defaultBackend }, runtimeLabel(backend)))),
    base_model: h("select", {}, modelOptions(models, (m) => h("option", { value: m.version, title: m.adapter ? ADAPTER_HINT : null },
      m.adapter ? `${m.version}（LoRA adapter，仅可 LoRA SFT）` : m.version))),
    model_name: h("input", { type: "text", maxlength: 100, pattern: "[A-Za-z0-9][A-Za-z0-9._-]*", placeholder: "例如 sokoban-grpo" }),
    data: h("select", { required: true }, h("option", { value: "" }, "请选择"), onlineSets,
      h("optgroup", { label: "离线轨迹数据集" }, catalog.datasets.map((d) => h(
        "option", { value: `ds:${d.dataset_version}` }, `${d.dataset_version}（${d.sample_count} 条）`)))),
    eval_set: h("select", {}, catalog.task_sets.map((s) => h(
      "option", { value: s.name, selected: s.name === "sokoban-validation" }, taskSetLabel(s)))),
    rounds: h("input", { type: "number", min: 1, max: 100, value: 20 }),
    tasks_per_round: h("input", { type: "number", min: 1, max: 16, value: 8 }),
    group_size: h("input", { type: "number", min: 2, max: 16, value: 8 }),
    temperature: h("input", { type: "number", min: 0, max: 2, step: 0.1, value: 1.0 }),
    max_steps: h("input", { type: "number", min: 1, max: 50, value: 20 }),
    learning_rate: h("input", { type: "number", min: 1e-9, max: 1e-3, step: "any", value: 1e-6 }),
    epochs: h("input", { type: "number", min: 1, max: 10, value: 1 }),
    gradient_accumulation: h("input", { type: "number", min: 1, max: 64, value: 16 }),
    batch_size: h("input", { type: "number", min: 1, max: 16, value: 1 }),
    cache_preprocessed: h("input", { type: "checkbox", checked: true }),
    gradient_checkpointing: h("input", { type: "checkbox" }),
    clip_epsilon: h("input", { type: "number", min: 0.01, max: 1, step: 0.01, value: 0.2 }),
    clip_epsilon_high: h("input", { type: "number", min: 0.01, max: 1, step: 0.01, placeholder: "按算法默认" }),
    advantage_std: presetSelect([["true", "除以组内 std"], ["false", "不除（只减均值）"]]),
    drop_zero_variance_groups: presetSelect([["true", "舍弃"], ["false", "不舍弃"]]),
    overlong: presetSelect([["fail", "当作失败（reward 0）"], ["mask", "屏蔽（不训练，仍计入基线）"], ["penalty", "惩罚（reward 取下面的值）"]]),
    overlong_penalty: h("input", { type: "number", min: -1, max: 0, step: 0.05, value: -0.1 }),
    loss_normalization: presetSelect([["per_rollout", "每条轨迹步数 |o_i|"], ["batch_steps", "整批总步数 Σ|o_i|"], ["constant", "常数（见下）"]]),
    loss_normalization_constant: h("input", { type: "number", min: 1, max: 1000, placeholder: "默认 = 最大步数" }),
    step_advantage_omega: h("input", { type: "number", min: 0, max: 10, step: 0.1, placeholder: "按算法默认" }),
    step_advantage_gamma: h("input", { type: "number", min: 0.01, max: 1, step: 0.01, placeholder: "默认 0.95" }),
    invalid_output_reward: h("input", { type: "number", min: -1, max: 0, step: 0.05, value: -0.2 }),
    eval_trials: h("input", { type: "number", min: 0, max: 50, value: 1 }),
    checkpoint_every: h("input", { type: "number", min: 1, max: 100, value: 5, placeholder: "只保留正式模型" }),
    seed: h("input", { type: "number", value: 0 }),
  };
  const knobLabels = {};
  const knob = (key, label, hint) => (knobLabels[key] = field(label, inputs[key], hint));
  const presetHints = {};
  const component = (key, label, hint) => {
    presetHints[key] = h("span", { class: "hint" });
    return (knobLabels[key] = h("label", {}, label, inputs[key], presetHints[key], h("span", { class: "hint" }, hint)));
  };
  // Collapsed by default: every blank component keeps the chosen algorithm's preset (README 策略优化算法).
  const components = h("details", { class: "wide" }, h("summary", {}, "高级：算法组件（留空 = 按算法默认）"),
    h("div", { class: "form", style: "margin-top:8px" },
      component("advantage_std", "优势除以组内 std", "GRPO 的 (r − mean) / std；不除就是 Dr. GRPO 的 r − mean，避免 reward 方差小的组被放大"),
      component("drop_zero_variance_groups", "舍弃优势全为 0 的组", "组内 reward 全相同时没有梯度。舍弃后它不再占用长度归一化的分母（DAPO 的动态采样，但不补采）"),
      component("clip_epsilon_high", "clip ε 上界", "比值上界为 1 + 这个值，下界仍是 1 − clip ε。留空：DAPO 0.28，其余同 clip ε（对称）"),
      component("overlong", "超长轨迹（到最大步数）", "当作失败：reward 0；屏蔽：不训练，但计入组内基线（DAPO overlong filtering）；惩罚：reward 改为下面的值"),
      component("overlong_penalty", "超长惩罚 reward", "仅在“超长轨迹 = 惩罚”时生效，−1 到 0；无效输出默认 −0.2，未解出 0"),
      component("loss_normalization", "长度归一化", "每步 loss 的权重分母。每条轨迹步数：每条轨迹权重相同；整批总步数：每步权重相同；常数：Dr. GRPO 的 1 / (常数 × 轨迹数)"),
      component("loss_normalization_constant", "归一化常数", "仅在“长度归一化 = 常数”时生效，留空 = 每条 rollout 最大步数"),
      component("step_advantage_omega", "步级优势权重 ω", "GiGPO 的 A_E + ω · A_S：同组里相同棋盘的步按折扣回报比较。0 关闭；其它算法也可以打开"),
      component("step_advantage_gamma", "步级折扣 γ", "GiGPO 的回报 γ^(T−t) · R，留空 0.95"),
      component("invalid_output_reward", "无效输出 reward", "JSON 不合法或点到按钮以外的轨迹的 reward，−1 到 0，默认 −0.2；只影响训练，评测成功率不变")));
  const submit = h("button", { class: "primary", type: "submit", disabled: !models.length }, "创建实验");
  const form = h("form", { class: "form", onsubmit: async (event) => {
    event.preventDefault();
    const data = inputs.data.value;
    const backend = inputs.runtime_backend.value;
    const knobs = ALGORITHM_KNOBS[inputs.algorithm.value].filter((key) => !inputs[key].disabled).map((key) => {
      const input = inputs[key];
      if (input.tagName === "SELECT" && COMPONENT_KNOBS.includes(key)) {
        return [key, input.value === "" ? null : input.value === "true" ? true : input.value === "false" ? false : input.value];
      }
      return [key, input.type === "number" ? (input.value === "" ? null : Number(input.value)) : input.type === "checkbox" ? input.checked : input.value];
    });
    const body = {
      algorithm: inputs.algorithm.value,
      base_model: inputs.base_model.value,
      dataset_version: data.startsWith("ds:") ? data.slice(3) : null,
      environment_version: `android-${backend}-v1`,
      reward_version: "android-task-v1",
      runtime_backend: backend,
      config: { model_name: inputs.model_name.value.trim() || null, ...Object.fromEntries(knobs),
        train_set: data.startsWith("set:") ? data.slice(4) : null, eval_set: inputs.eval_set.value },
    };
    submit.disabled = true;
    try {
      const run = await api("/api/v1/training/runs", { method: "POST", body: JSON.stringify(body), headers: { "Idempotency-Key": idempotencyKey() } });
      location.hash = `#/runs/${run.run_id}`;
    } catch (error) {
      message.replaceChildren(h("p", { class: "error-text" }, `创建失败：${error.message}`));
    } finally { submit.disabled = false; }
  } },
  field("算法", inputs.algorithm, "4 种在线算法共用同一套 rollout，只是 advantage 和每步 loss 权重不同（见使用教程）。默认 Dr. GRPO：多步推箱子里不按轨迹长度归一，短的成功轨迹不会被压低权重"),
  knob("accelerator", "训练资源", "H800：在线算法每个实验常驻 2 张卡（训练 + sglang 采样），整个实验期间都占着；每轮 checkpoint 后台复制到 4090。4090：和评测、模型服务共用一张卡，一次只跑一个实验"),
  field("Runtime", inputs.runtime_backend, "rollout 用的 Android 环境。Firecracker：microVM，模板快照克隆约 1 秒就绪；Docker：Redroid 容器，不支持快照，截图与 Firecracker 不完全相同，成功率不能直接比"),
  field("基础模型", inputs.base_model, models.length ? "也可以选择 checkpoint 接着训练。在线算法要求完整权重；LoRA adapter 只能继续 LoRA SFT" : "尚未注册模型，请先运行 import_artifacts"),
  field("模型产物名称", inputs.model_name, "可选，默认用实验 ID。只允许字母、数字、点、下划线和短横线；每轮自动加后缀，如 -r0、-r11"),
  field("数据集", inputs.data, "在线算法选题目集：每轮从中抽题，在真实 Android 上采样。选轨迹数据集则离线训练。SFT 只能选轨迹数据集"),
  knob("finetuning_method", "微调方式", "全参数：最近几个在线实验都用它。LoRA：rank 16 / alpha 32，只训语言侧注意力层，FP32 adapter，lr 1e-5 时 clip 比例约 45%，需要更小的学习率；在线算法每轮把 adapter 合并成完整权重再用 sglang 采样"),
  knob("train_merger_mlp", "全量训练视觉 merger", "仅 LoRA；额外全量训练视觉-语言投影层（MiniCPM-V：model.merger.mlp.0，约 2,597 万参数；Qwen3-VL：model.visual.merger）"),
  knob("multiview", "双视图输入", "完整截图 + 棋盘裁剪两张图；推理时按 checkpoint 的配置自动使用"),
  knob("output_format", "输出协议", "必须与数据集一致。结构化输出多出棋盘和决策文本，执行器仍只执行其中的 action"),
  field("评测题目集", inputs.eval_set, "训练结束后自动评测用，最多 48 题。要和训练题目集不同，否则测不出泛化；选训练集时默认换成 sokoban-validation"),
  knob("rounds", "轮数", "每轮：用上一轮的策略采样、再训练一次，最多 100 轮。每道题练到的次数 = 轮数 × 每轮题目数 ÷ 题目集大小，默认 20 × 8 ÷ 48 ≈ 3.3 次；想多练同一批题就缩小题目集或加轮数。H800 上每轮 4 题约 10 分钟，8 题约 15–20 分钟（估计），实际随题目和步数变化"),
  knob("tasks_per_round", "每轮题目数", "按 seed 洗牌后依次取题，题目集做完一遍才会重复，最多 16。每轮 rollout 数 = 题目数 × group size，默认 8 × 8 = 64。题少则每次更新只来自几道题、噪声大；题多则每轮更慢、两次采样之间策略改得更多"),
  knob("group_size", "每组 rollout 数 (group size)", "同一道题采样几次，advantage 在组内比较。组内 reward 全相同时这组不参与训练；8 比 4 更容易出现成功和失败混在一组"),
  knob("temperature", "采样 temperature", "rollout 的采样温度。组内需要 reward 差异，建议 ≥ 0.7；训练时 logprob 也按这个温度计算"),
  knob("max_steps", "每条 rollout 最大步数", "到步数还没通关记为失败。训练题最优解 4–10 步，20 给模型留出试错的余地"),
  knob("learning_rate", "学习率", "全参数默认 1e-6：1e-7 时 BF16 几乎舍掉全部更新（12 轮后与基础模型逐关结果相同），1e-5 曾破坏输出格式"),
  knob("epochs", "epochs", "每轮样本训练几遍。在线算法建议 1：多遍会偏离采样时的策略，clip 比例上升"),
  knob("gradient_accumulation", "梯度累积", "每多少个样本做一次 optimizer 更新（有效 batch = micro-batch × 梯度累积）。默认 16：一轮约 300–500 行，约 20–30 次更新；设成 1 相当于每个样本更新一次"),
  knob("clip_epsilon", "clip ε", "PPO-clip 范围：新旧策略概率比超出 1 ± ε 的样本不再贡献梯度"),

  knob("batch_size", "micro-batch", "一次前向放几个样本。在线算法建议 1：H800 上 batch 4 只快 10%，峰值显存已到 78.7 GB（80 GB 卡），长样本会爆显存。SFT 在 H800 常用 4"),
  knob("cache_preprocessed", "缓存预处理输入", "把截图预处理结果放在 CPU 内存里复用。在线算法的 old_logprob 和训练共用一份，预处理时间减半；SFT 多个 epoch 也只做一遍。333 条双视图约需 10 GiB 内存"),
  knob("gradient_checkpointing", "梯度检查点", "开启后反向时重算前向，省显存但更慢。H800 上关闭训练快约 21%，在线算法 batch 1 显存 20 → 30 GB，远低于 80 GB；4090 显存不够时再开"),
  knob("checkpoint_every", "checkpoint 保存间隔", "在线算法每 N 轮、SFT 每 N 个 epoch 保留一个 checkpoint，其余训练结束后删除；最后一轮总是正式模型。留空只保留正式模型"),
  knob("eval_trials", "训练后每任务评测次数", "训练成功后在 4090 上对比评测基础模型和正式模型，题目集见上，步数同 rollout，0 表示不评测。自动评测用 temperature 0，同一题重复结果基本相同，所以默认 1"),
  knob("seed", "随机 seed", "决定抽题顺序和采样 seed；同一个 seed 重跑，每轮抽到的题相同"),
  components,
  h("div", { class: "wide row" }, submit, message));
  const syncAlgorithm = () => {
    const visible = new Set(ALGORITHM_KNOBS[inputs.algorithm.value]);
    if (inputs.finetuning_method.value !== "lora") visible.delete("train_merger_mlp");

    // Disabled controls skip validation, so a hidden knob can never block submit.
    for (const [key, label] of Object.entries(knobLabels)) label.hidden = inputs[key].disabled = !visible.has(key);
    components.hidden = inputs.algorithm.value === "sft";
    for (const [key, hint] of Object.entries(presetHints)) {
      const preset = PRESET_LABELS[key]?.[inputs.algorithm.value];
      hint.textContent = preset ? `当前算法默认：${preset}` : "";
    }
    onlineSets.disabled = inputs.algorithm.value === "sft";
    if (inputs.data.selectedOptions[0]?.parentElement.disabled) inputs.data.value = "";
  };
  inputs.algorithm.addEventListener("change", syncAlgorithm);
  inputs.finetuning_method.addEventListener("change", syncAlgorithm);
  // Default the evaluation set to the held-out split of the chosen pool, else the pool itself.
  inputs.data.addEventListener("change", () => {
    const name = inputs.data.value.startsWith("set:") ? inputs.data.value.slice(4) : null;
    const evalName = name === "sokoban-train" ? "sokoban-validation" : name;
    if ([...inputs.eval_set.options].some((option) => option.value === evalName)) inputs.eval_set.value = evalName;
  });
  syncAlgorithm();
  const details = h("details", { class: "card" }, h("summary", {}, "新建训练实验"), h("div", { style: "margin-top:12px" }, form));
  details.load = (run) => {
    // Runs created before task sets stored a single task in config.tasks, used for training and evaluation.
    const legacy = (run.config.tasks || []).length === 1 ? run.config.tasks[0].app_image : null;
    const trainSet = run.config.train_set ?? legacy;
    const wanted = { algorithm: run.algorithm, base_model: run.base_model,
      data: run.dataset_version ? `ds:${run.dataset_version}` : trainSet && `set:${trainSet}`,
      runtime_backend: run.runtime_backend, accelerator: "4090", ...run.config, eval_set: run.config.eval_set ?? legacy };
    const missing = [];
    for (const [key, value] of Object.entries(wanted)) {
      if (!inputs[key] || key === "model_name" || value === undefined) continue;
      // Blank is meaningful only where the input says so (placeholder); elsewhere keep the default.
      if (value === null) { if (inputs[key].placeholder) inputs[key].value = ""; continue; }
      const input = inputs[key];
      if (input.type === "checkbox") { input.checked = Boolean(value); continue; }
      input.value = typeof value === "boolean" ? String(value) : value;
      if (input.type === "number" ? Number(input.value) !== Number(value) : input.value !== String(value)) {
        missing.push(`${key}=${value}`);
      }
    }
    // A reused name would collide with the earlier run's registered version.
    inputs.model_name.value = "";
    syncAlgorithm();
    message.replaceChildren(h("p", { class: missing.length ? "error-text" : "muted" }, missing.length
      ? `已载入 ${fmt.short(run.run_id)} 的配置，以下项当前不可用，请重新选择：${missing.join("，")}`
      : `已载入 ${fmt.short(run.run_id)} 的配置，模型产物名称已清空`));
    details.open = true;
    details.scrollIntoView({ behavior: "smooth" });
  };
  return details;
}

async function runDetailPage(runId) {
  const root = h("div");
  view.replaceChildren(h("div", { class: "row" }, h("a", { href: "#/runs" }, "← 训练实验")), root);
  await poll(async () => {
    const [run, metrics, rollouts] = await Promise.all([
      api(`/api/v1/training/runs/${runId}`),
      api(`/api/v1/training/runs/${runId}/metrics`),
      api(`/api/v1/training/runs/${runId}/rollouts`),
    ]);
    const evals = await Promise.all(run.evaluations.map((id) => api(`/api/v1/evaluations/${id}`)));
    const active = ["QUEUED", "RUNNING"].includes(run.status);
    const series = metrics.series;
    const names = Object.keys(METRIC_LABELS).filter((name) => series[name]);
    const extra = Object.keys(series).filter((name) => !METRIC_LABELS[name]);
    root.replaceChildren(
      h("div", { class: "row" }, h("h2", {}, `${algorithmLabel(run.algorithm)} 实验 ${fmt.short(run.run_id)}`), badge(run.status), h("div", { class: "spacer" }),
        runActions(run, "#/runs")),
      h("div", { class: "card" }, h("dl", { class: "kv" },
        h("dt", {}, "基础模型"), h("dd", {}, run.base_model),
        h("dt", {}, "Runtime"), h("dd", {}, run.runtime_backend || "unknown"),
        h("dt", {}, "数据"), h("dd", {}, runData(run) || "–"),
        h("dt", {}, "评测题目"), h("dd", {}, run.config.eval_set || (run.config.eval_tasks?.length ? run.config.eval_tasks : run.config.tasks || []).map((t) => t.app_image).join(", ") || "–"),
        h("dt", {}, "配置"), h("dd", {}, h("div", { class: "row" },
          h("span", { class: "mono" }, JSON.stringify({ ...run.config, tasks: undefined, eval_tasks: undefined })),
          h("button", { onclick: () => { pendingRecipe = run; location.hash = "#/runs"; } }, "复用配置"))),
        h("dt", {}, "创建 / 更新"), h("dd", {}, `${fmt.time(run.created_at)} / ${fmt.time(run.updated_at)}`),
        h("dt", {}, "产出模型"), h("dd", {}, run.metrics.model_version || "–"),
        h("dt", {}, "checkpoint"), h("dd", { class: "mono" }, run.checkpoint_uri || "–"),
        run.metrics.error ? [h("dt", {}, "错误"), h("dd", { class: "error-text" }, run.metrics.error)] : null)),
      h("h3", {}, "指标监控"),
      names.length + extra.length
        ? h("div", { class: "charts" },
          names.map((name) => { const [title, xLabel, format] = METRIC_LABELS[name]; return lineChart(title, series[name], { xLabel, format }); }),
          extra.map((name) => lineChart(name, series[name])))
        : h("div", { class: "card muted" }, active ? "等待训练 agent 上报指标…" : "该实验没有逐步指标"),
      h("h3", {}, "Rollout 轨迹"),
      rolloutBrowser(rollouts),
      h("h3", {}, "实验结果评测"),
      evals.length ? evals.map((ev) => h("div", { class: "card" }, h("a", { href: `#/evaluations/${ev.eval_id}` }, `评测 ${fmt.short(ev.eval_id)}`), " ", badge(ev.status), evalSummary(ev))) : h("div", { class: "card muted" }, "训练成功后会自动评测基础模型与产出模型"),
      h("h3", {}, "训练日志"),
      h("div", { class: "card" }, h("pre", { class: "log" }, run.log_tail || "（暂无日志）")),
    );
    if (!active && !rollouts.some((r) => ["queued", "running"].includes(r.outcome)) && !evals.some((e) => e.status === "RUNNING")) clearInterval(pollTimer);
  }, 4000);
}

function rolloutBrowser(rollouts) {
  if (!rollouts.length) return h("div", { class: "card muted" }, "没有在线 rollout（离线数据集训练或尚未开始）");
  const rounds = new Map();
  for (const r of rollouts) {
    if (!rounds.has(r.round)) rounds.set(r.round, new Map());
    const groups = rounds.get(r.round);
    if (!groups.has(r.group_id)) groups.set(r.group_id, []);
    groups.get(r.group_id).push(r);
  }
  return h("div", {}, [...rounds].map(([round, groups]) => h("details", { class: "card", open: round === Math.max(...rounds.keys()) },
    h("summary", {}, `第 ${round} 轮 · ${[...groups.values()].flat().length} 条 rollout · 成功 ${[...groups.values()].flat().filter((r) => r.outcome === "success").length}`),
    [...groups].map(([groupId, members]) => {
      const rewards = members.map((m) => m.reward_total);
      return h("div", { style: "margin-top:10px" },
        h("div", { class: "secondary" }, `${groupId} · ${members[0].task_type}/${members[0].app_image} · policy ${members[0].policy_version}`,
          new Set(rewards).size <= 1 && members.every((m) => !["queued", "running"].includes(m.outcome)) ? h("span", { class: "hint" }, "（组内 reward 相同，不产生梯度）") : null),
        table([
          { label: "seed", num: true, key: "seed" },
          { label: "结果", render: (r) => badge(r.outcome) },
          { label: "reward", num: true, render: (r) => fmt.num(r.reward_total) },
          { label: "步数", num: true, key: "steps" },
          { label: "尝试", num: true, render: (r) => (r.attempts > 1 ? h("span", { title: "运行错误后用同一 seed 重跑" }, `${r.attempts} 次`) : "1") },
          { label: "动作序列", render: (r) => (r.step_records || []).map((s) => describeAction(s.action)).join(" → ") || r.error || "–" },
        ], members, (r) => { location.hash = `#/episodes/${r.episode_id}`; }));
    }))));
}

// ---------------------------------------------------------------- data

async function dataPage() {
  const catalog = await api("/api/v1/catalog");
  view.replaceChildren(
    pageHeading("DATA REGISTRY / 02", "数据管理", "SFT 使用离线轨迹数据集；在线策略算法 每轮从题目集抽题在线 rollout，训练后在评测题目集上自动评测。"),
    h("h3", {}, "离线轨迹数据集"),
    h("div", { class: "card" }, table([
      { label: "版本", render: (d) => h("span", { class: "mono" }, d.dataset_version) },
      { label: "schema", key: "schema_version" },
      { label: "样本数", num: true, render: (d) => fmt.num(d.sample_count, 0) },
      { label: "来源", render: (d) => h("span", { class: "mono" }, d.source) },
      { label: "注册时间", render: (d) => fmt.time(d.created_at) },
    ], catalog.datasets)),
    h("h3", {}, "题目集"),
    h("div", { class: "card" },
      h("p", { class: "muted" }, "单次评测最多 48 题；训练和评测应选不同 split，避免在训练题上评测。"),
      table([
        { label: "名称", render: (s) => h("span", { class: "mono" }, s.name) },
        { label: "说明", key: "label" },
        { label: "用途", render: () => "训练 / 评测" },
        { label: "题目", render: (s) => h("details", {}, h("summary", {}, `${s.tasks.length} 题`),
          h("div", { class: "mono" }, s.tasks.map((t) => t.app_image).join(", "))) },
      ], catalog.task_sets)),
  );
}

// ---------------------------------------------------------------- evaluations

async function evaluationsPage() {
  const catalog = await api("/api/v1/catalog");
  const list = h("div", { class: "card" });
  view.replaceChildren(pageHeading("EVALUATION DECK / 04", "模型评测", "用相同任务、seed 和 runtime 对比基础模型与训练产出，结果可回放、可追溯。"), evalForm(catalog), h("h3", {}, "评测记录"), list);
  await poll(async () => {
    const rows = await api("/api/v1/evaluations");
    list.replaceChildren(table([
      { label: "评测", render: (e) => h("div", {},
        h("a", { href: `#/evaluations/${e.eval_id}` }, e.name || fmt.short(e.eval_id)),
        h("div", { class: "muted mono" }, fmt.short(e.eval_id))) },
      { label: "Runtime", render: (e) => e.runtime_backend || "unknown" },
      { label: "推理", render: (e) => inferenceLabel(e) },
      { label: "模型", render: (e) => e.model_versions.join(" vs ") },
      { label: "任务", render: (e) => taskSummary(e.task_set, catalog.task_sets) },
      { label: "状态", render: (e) => badge(e.status) },
      { label: "成功 / trial", num: true, render: (e) => `${e.successes} / ${e.trials}` },
      { label: "来源", render: (e) => (e.run_id ? h("a", { href: `#/runs/${e.run_id}` }, "训练后自动评测") : "手动") },
      { label: "创建时间", render: (e) => fmt.time(e.created_at) },
      { label: "", render: (e) => evalActions(e) },
    ], rows, (e) => { location.hash = `#/evaluations/${e.eval_id}`; }));
  }, 5000);
}

function evalForm(catalog) {
  const message = h("div");
  const runtimeBackends = catalog.runtime_backends || [catalog.runtime_backend || "firecracker"];
  const defaultBackend = catalog.runtime_backend || runtimeBackends[0];
  const models = catalog.model_versions.filter((model) => model.status !== "RETIRED");
  const name = h("input", { type: "text", maxlength: 100, placeholder: "例如 sokoban-baseline" });
  const runtimeBackend = h("select", {}, runtimeBackends.map((backend) => h(
    "option", { value: backend, selected: backend === defaultBackend }, runtimeLabel(backend))));
  const accelerator = h("select", {}, h("option", { value: "h800", selected: true }, "H800（为本次评测部署所选模型）"),
    h("option", { value: "4090" }, "4090（常驻模型服务）"));
  const formal = models.filter((m) => m.kind !== "checkpoint");
  const modelBoxes = models.map((m) => h("input", { type: "checkbox", value: m.version, checked: m === formal[formal.length - 1] }));
  const boxes = (rows) => h("div", { class: "checks" }, rows.map((m) => h("label", {}, modelBoxes[models.indexOf(m)], modelLabel(m))));
  const checkpoints = models.filter((m) => m.kind === "checkpoint");
  const evalSets = catalog.task_sets;
  const taskSelect = h("select", {},
    h("optgroup", { label: "单个任务" }, catalog.tasks.map((t, i) => h("option", { value: `task:${i}` }, `${t.label}（${t.app_image}）`))),
    evalSets.length ? h("optgroup", { label: "题目集" }, evalSets.map((s) => h(
      "option", { value: `set:${s.name}`, selected: s.name === "sokoban-validation" }, taskSetLabel(s)))) : null);
  const selectedTasks = () => (taskSelect.value.startsWith("set:")
    ? evalSets.find((s) => s.name === taskSelect.value.slice(4)).tasks
    : [catalog.tasks[Number(taskSelect.value.slice(5))]]);
  // Every task of a set shares its type's instruction (one Sokoban instruction for all levels).
  const defaultInstruction = () => catalog.tasks.find((t) => t.task_type === selectedTasks()[0]?.task_type)?.instruction || "";
  const instruction = h("textarea", { value: defaultInstruction() });
  taskSelect.addEventListener("change", () => { instruction.value = defaultInstruction(); });
  const trials = h("input", { type: "number", min: 1, max: 50, value: 1 });
  const maxSteps = h("input", { type: "number", min: 1, max: 50, value: 20 });
  const temperature = h("input", { type: "number", min: 0, max: 2, step: 0.1, value: 0 });
  const seed = h("input", { type: "number", value: 0 });
  const submit = h("button", { class: "primary", type: "submit", disabled: !models.length }, "开始评测");
  const form = h("form", { class: "form", onsubmit: async (event) => {
    event.preventDefault();
    const versions = modelBoxes.filter((b) => b.checked).map((b) => b.value);
    if (!versions.length) { message.replaceChildren(h("p", { class: "error-text" }, "至少选择一个模型")); return; }
    const body = {
      model_versions: versions,
      tasks: selectedTasks().map((t) => ({ task_type: t.task_type, app_image: t.app_image, instruction: instruction.value.trim() || null })),
      trials_per_task: Number(trials.value), max_steps: Number(maxSteps.value),
      temperature: Number(temperature.value), seed: Number(seed.value),
      name: name.value.trim() || null,
      runtime_backend: runtimeBackend.value,
      accelerator: accelerator.value,
    };
    submit.disabled = true;
    try {
      const created = await api("/api/v1/evaluations", { method: "POST", body: JSON.stringify(body), headers: { "Idempotency-Key": idempotencyKey() } });
      location.hash = `#/evaluations/${created.eval_id}`;
    } catch (error) {
      message.replaceChildren(h("p", { class: "error-text" }, `创建失败：${error.message}`));
    } finally { submit.disabled = false; }
  } },
  h("div", { class: "wide" }, h("div", { class: "secondary" }, "模型 / checkpoint（最多 6 个；多选即对比评测，所有模型用相同的题目和 seed）"),
    boxes(formal),
    checkpoints.length ? h("details", {}, h("summary", { class: "secondary" }, `Checkpoint（${checkpoints.length} 个）`), boxes(checkpoints)) : null),
  field("评测名称", name, "可选，不填时自动生成"),
  field("Runtime", runtimeBackend, "Firecracker：Android microVM，约 1 秒就绪。Docker：Redroid 容器。两者截图不完全相同，成功率只在同一种 runtime 之间比"),
  field("推理资源", accelerator, "H800：每个模型单独一个 Pod、一张卡，哪个模型先加载好，它的 trial 就先开始；MiniCPM-V 和 Qwen3-VL 的完整权重走 sglang 确定性推理（重跑结果一致），LoRA 在 Pod 里合并，perception 等特殊模型走 HF。选几个模型就起几个 Pod，集群没空卡时 Pod 会等调度。结束自动删除。4090：常驻 HF 服务，和训练、其他评测共用"),
  field("任务 / 题目集", taskSelect, "题目集里每题都会跑。默认 sokoban-validation（8 题，训练时没见过）；sokoban-test 已反复用于回归，不宜作泛化结论。单次最多 48 题、2304 个 trial（模型数 × 题数 × 每题次数）"),
  field("每题 trial 次数", trials, "temperature 0 时同一题重复结果基本相同，默认 1；要更多样本就换更大的题目集。temperature > 0 时才值得重复，对比成功率建议每模型总 trial ≥ 20"),
  field("最大步数", maxSteps, "到步数还没通关记为失败。推箱子题最优解 4–10 步，默认 20 与训练 rollout 一致"),
  field("temperature", temperature, "0 为贪心解码，结果可复现，模型之间可以直接比较；> 0 时按采样评估，需要多次 trial"),
  field("起始 seed", seed, "第 n 次 trial（从 0 数）用 起始 seed + n；temperature 0 时不影响结果"),
  h("label", { class: "wide" }, "指令（可修改，默认是该任务类型的标准指令；改了之后和其他评测的结果不能直接比）", instruction),
  h("div", { class: "wide row" }, submit, message));
  return h("details", { class: "card" }, h("summary", {}, "新建模型评测"), h("div", { style: "margin-top:12px" }, form));
}

function inferenceLabel(ev) {
  if (ev.accelerator !== "h800") return "4090";
  if (ev.model_ready) return "H800（模型已就绪）";
  if (ev.status !== "RUNNING") return "H800";
  // One pod per model: each model's trials start once its own pod is ready.
  const ready = (ev.ready_models || []).length;
  return ready ? `H800（${ready} / ${ev.model_versions.length} 个模型就绪）` : "H800（部署中）";
}

function evalSummary(ev) {
  const rows = ev.summary;
  const smallSample = rows.some((r) => r.scored < 20);
  return h("div", {},
    rows.some((r) => r.success_rate !== null)
      ? rateBars(rows.map((r) => ({ label: r.model_version, value: r.success_rate, text: `${fmt.pct(r.success_rate)}（${r.successes}/${r.scored}）` })))
      : null,
    table([
      { label: "模型", key: "model_version" },
      { label: "成功率", num: true, render: (r) => `${fmt.pct(r.success_rate)}（${r.successes}/${r.scored}）` },
      { label: "平均 reward", num: true, render: (r) => fmt.num(r.mean_reward) },
      { label: "平均步数", num: true, render: (r) => fmt.num(r.mean_steps, 1) },
      { label: "动作格式合法率", num: true, render: (r) => fmt.pct(r.format_rate) },
      { label: "非法输出", num: true, key: "invalid_outputs" },
      { label: "推理 p50 / p95", num: true, render: (r) => `${fmt.ms(r.latency_p50_ms)} / ${fmt.ms(r.latency_p95_ms)}` },
      { label: "进度", num: true, render: (r) => `${r.finished} / ${r.trials}` },
    ], rows),
    smallSample ? h("p", { class: "hint" }, "提示：每个模型的有效 trial 少于 20 次，成功率差异可能只是采样波动。") : null);
}

async function evaluationDetailPage(evalId) {
  const root = h("div");
  const live = h("div");
  view.replaceChildren(h("div", { class: "row" }, h("a", { href: "#/evaluations" }, "← 模型评测")), root, live);
  let liveEpisode = null;
  const taskSets = (await api("/api/v1/catalog")).task_sets;
  await poll(async () => {
    const ev = await api(`/api/v1/evaluations/${evalId}`);
    root.replaceChildren(
      h("div", { class: "row" }, h("h2", {}, ev.name || `评测 ${fmt.short(ev.eval_id)}`), badge(ev.status), h("div", { class: "spacer" }),
        evalActions(ev, "#/evaluations")),
      h("div", { class: "card" },
        h("div", { class: "secondary" }, `Runtime ${ev.runtime_backend || "unknown"} · 推理 ${inferenceLabel(ev)} · 任务 `,
          taskSummary(ev.task_set, taskSets), ` · 每模型 ${ev.trials_per_task} 次 · 最多 ${ev.max_steps} 步 · temperature ${ev.temperature} · seeds ${ev.seeds.length > 4 ? `${ev.seeds.slice(0, 4).join(", ")} … 共 ${ev.seeds.length} 个` : ev.seeds.join(", ")}`),
        evalSummary(ev)),
      h("h3", {}, "每次 trial"),
      h("div", { class: "card" }, table([
        { label: "模型", key: "policy_version" },
        { label: "seed", num: true, key: "seed" },
        { label: "结果", render: (t) => badge(t.outcome) },
        { label: "reward", num: true, render: (t) => fmt.num(t.reward_total) },
        { label: "步数", num: true, key: "steps" },
        { label: "说明", render: (t) => t.error || "–" },
      ], ev.trials, (t) => { location.hash = `#/episodes/${t.episode_id}`; })),
    );
    // A single trial is the "try the model" flow: watch the episode inline.
    if (ev.trials.length === 1 && liveEpisode !== ev.trials[0].episode_id) {
      liveEpisode = ev.trials[0].episode_id;
    }
    if (liveEpisode) live.replaceChildren(h("h3", {}, "执行过程"), await episodeView(liveEpisode));
    if (ev.status !== "RUNNING") clearInterval(pollTimer);
  }, 3000);
}

// ---------------------------------------------------------------- episodes

function modelOutput(raw, invalid) {
  if (invalid) return raw ? `${raw}（非法输出，未执行）` : "（非法输出，模型服务未返回原文）";
  if (!raw) return "（固定策略）";
  try {
    const output = JSON.parse(raw);
    if (output.board && output.action) return h("details", {},
      h("summary", {}, "查看棋盘、决策和动作预测"),
      h("pre", { style: "white-space:pre-wrap" }, JSON.stringify(output, null, 2)));
  } catch (_) { /* Keep malformed model output visible. */ }
  return raw;
}

async function episodeView(episodeId) {
  const ep = await api(`/api/v1/episodes/${episodeId}`);
  const base = `/api/v1/episodes/${encodeURIComponent(episodeId)}`;
  const finalReward = ep.result?.final_reward;
  return h("div", { "data-active": ["queued", "running"].includes(ep.outcome) ? "1" : "0" },
    h("div", { class: "card" },
      h("div", { class: "row" }, badge(ep.outcome), h("strong", {}, `${ep.task_type} · ${ep.policy_version}`),
        h("span", { class: "muted" }, `seed ${ep.seed ?? "–"} · temperature ${ep.temperature} · ${ep.steps.length} / ${ep.max_steps} 步`)),
      h("dl", { class: "kv", style: "margin-top:8px" },
        h("dt", {}, "Runtime"), h("dd", {}, `${ep.runtime_backend || "unknown"} → ${ep.actual_runtime_backend || "未执行"}`),
        h("dt", {}, "指令"), h("dd", {}, ep.instruction),
        h("dt", {}, "最终结果"), h("dd", {}, finalReward
          ? `reward ${fmt.num(finalReward.total)} · ${finalReward.success ? "任务完成" : "任务未完成"} · 前台 ${finalReward.foreground_package || "–"}${finalReward.termination_requested ? " · 模型主动结束" : ""}`
          : ep.result?.error || (ep.outcome === "success" ? "成功" : "进行中")),
        ep.sandbox_id ? [h("dt", {}, "sandbox"), h("dd", { class: "mono" }, ep.sandbox_id)] : null)),
    h("div", { class: "steps" },
      ep.steps.map((s) => h("div", { class: "card step" },
        h("div", { class: "shot" }, s.observation?.has_screenshot ? h("img", { src: `${base}/steps/${s.step_id}/screenshot`, alt: `第 ${s.step_id} 步截图`, loading: "lazy" }) : h("div", { class: "thumb" }), actionOverlay(s.action)),
        h("dl", { class: "kv" },
          h("dt", {}, "步骤"), h("dd", {}, String(s.step_id)),
          h("dt", {}, "动作"), h("dd", {}, describeAction(s.action)),
          h("dt", {}, "模型输出"), h("dd", { class: "mono" }, modelOutput(s.model_output?.raw_output, s.model_output?.invalid)),
          h("dt", {}, "延迟"), h("dd", {}, fmt.ms(s.model_output?.latency_ms)),
          s.model_output?.entropy != null ? [h("dt", {}, "采样熵"), h("dd", {}, `${fmt.num(s.model_output.entropy, 3)} nats/token`)] : null,
          h("dt", {}, "reward"), h("dd", {}, `${fmt.num(s.reward.total)}${s.reward.success ? " · 成功" : ""}${s.done ? " · 结束" : ""}`),
          h("dt", {}, "前台应用"), h("dd", {}, s.reward.foreground_package || s.observation?.foreground_package || "–")))),
      ep.has_final_screenshot ? h("div", { class: "card step" },
        h("div", { class: "shot" }, h("img", { src: `${base}/final-screenshot`, alt: "最终截图" })),
        h("dl", { class: "kv" }, h("dt", {}, "最终画面"), h("dd", {}, finalReward?.success ? "任务完成" : "任务未完成"))) : null,
      !ep.steps.length ? h("div", { class: "card muted" }, ep.outcome === "queued" ? "排队中，等待空闲 sandbox…" : ep.outcome === "running" ? "Android sandbox 启动中，首张截图稍后出现…" : "没有执行步骤") : null));
}

async function episodePage(episodeId) {
  const root = h("div");
  view.replaceChildren(h("div", { class: "row" }, h("a", { href: "javascript:history.back()" }, "← 返回")), h("h2", {}, "轨迹回放"), root);
  await poll(async () => {
    const node = await episodeView(episodeId);
    root.replaceChildren(node);
    if (node.dataset.active === "0") clearInterval(pollTimer);
  }, 3000);
}

// ---------------------------------------------------------------- guide

function pageHeading(kicker, title, description) {
  return h("div", { class: "page-heading" },
    h("div", { class: "eyebrow" }, kicker),
    h("h2", {}, title),
    h("p", {}, description));
}

// The guide is the repository README (served by /console/readme), so the two stay in sync.
// marked and mermaid are vendored under vendor/ and loaded only on this page.
let guideLibs = null;
function loadScript(src) {
  return new Promise((resolve, reject) => {
    document.head.append(h("script", { src, onload: resolve, onerror: () => reject(new Error(`加载失败：${src}`)) }));
  });
}

function rewriteReadmeLinks(root) {
  for (const img of root.querySelectorAll("img")) {
    const src = img.getAttribute("src") || "";
    if (src.startsWith("docs/images/")) img.src = `/console/${src}`;
    img.loading = "lazy";
  }
  for (const a of root.querySelectorAll("a[href]")) {
    const href = a.getAttribute("href");
    if (href.startsWith("#")) {
      // In-page anchors would collide with the hash router; scroll instead.
      a.addEventListener("click", (event) => {
        event.preventDefault();
        document.getElementById(decodeURIComponent(href.slice(1)))?.scrollIntoView({ behavior: "smooth" });
      });
    } else if (!/^https?:/.test(href)) {
      a.removeAttribute("href");  // repo-relative docs are not served by the console
      a.title = "仓库内文档";
    }
  }
}

async function guidePage() {
  const token = routeToken;
  const [markdown] = await Promise.all([
    fetch("/console/readme", { credentials: "same-origin", cache: "no-store" }).then((r) => {
      if (!r.ok) throw new Error(`README 加载失败（${r.status}）`);
      return r.text();
    }),
    guideLibs ||= Promise.all([loadScript("/console/static/vendor/marked.min.js"), loadScript("/console/static/vendor/mermaid.min.js")]),
  ]);
  if (token !== routeToken) return;
  // README is repo-controlled, but still escape raw HTML so only Markdown-generated tags reach the DOM.
  const slugs = new Map();
  const escape = (text) => text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const md = new window.marked.Marked({ gfm: true, renderer: {
    html: ({ text }) => escape(text),
    heading({ tokens, depth, text }) {
      let slug = text.toLowerCase().trim().replace(/[^\p{L}\p{N}\s-]/gu, "").replace(/\s+/g, "-");
      const seen = slugs.get(slug) || 0;
      slugs.set(slug, seen + 1);
      if (seen) slug += `-${seen}`;
      return `<h${depth} id="${slug}">${this.parser.parseInline(tokens)}</h${depth}>`;
    },
    code: ({ text, lang }) => lang === "mermaid" ? `<div class="mermaid">${escape(text)}</div>` : false,
  } });
  const doc = new DOMParser().parseFromString(md.parse(markdown), "text/html");
  const article = h("article", { class: "readme card" }, ...doc.body.childNodes);
  rewriteReadmeLinks(article);
  view.replaceChildren(
    pageHeading("PLAYBOOK / 05", "使用教程", "本页内容即仓库根目录 README.md，修改 README 后刷新即可同步。"),
    article);
  const dark = document.documentElement.dataset.theme !== "light";
  window.mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: dark ? "dark" : "default" });
  await window.mermaid.run({ nodes: article.querySelectorAll(".mermaid") });
}

// ---------------------------------------------------------------- router

const routes = [
  [/^#\/resources$/, resourcesPage, "resources"],
  [/^#\/guide$/, guidePage, "guide"],
  [/^#\/data$/, dataPage, "data"],
  [/^#\/runs$/, runsPage, "runs"],
  [/^#\/runs\/([\w-]+)$/, runDetailPage, "runs"],
  [/^#\/evaluations$/, evaluationsPage, "evaluations"],
  [/^#\/evaluations\/([\w-]+)$/, evaluationDetailPage, "evaluations"],
  [/^#\/episodes\/([\w-]+)$/, episodePage, null],
];

async function route() {
  routeToken += 1;
  clearInterval(pollTimer);
  const hash = location.hash || "#/resources";
  for (const [pattern, page, nav] of routes) {
    const match = hash.match(pattern);
    if (!match) continue;
    if (nav) document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === nav));
    view.replaceChildren(h("p", { class: "muted" }, "加载中…"));
    try { await page(...match.slice(1)); } catch (error) { showError(error); }
    return;
  }
  location.hash = "#/resources";
}

window.addEventListener("hashchange", route);
route();
