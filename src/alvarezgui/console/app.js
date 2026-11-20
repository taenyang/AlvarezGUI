// AlvarezGUI console. All dynamic text goes through textContent (h()), never innerHTML.

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
const ADAPTER_HINT = "只含 LoRA adapter 权重，加载时依赖基础模型；只能继续 LoRA SFT，GRPO / GIGPO 和全参数训练需先合并为完整权重";
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
    ...section("LoRA", "只含 adapter 权重；GRPO / GIGPO 和全参数训练需先合并", models.filter((m) => m.adapter), { grouped: true }),
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

const CONCURRENCY = { sandboxes: "Sandbox", training_runs: "训练", evaluations: "评测推理" };
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

function algorithmLabel(algorithm) {
  return algorithm === "gigpo" ? "GIGPO（GRPO 兼容版）" : algorithm.toUpperCase();
}

// Config knobs each algorithm consumes; the form shows and submits only these.
const SHARED_KNOBS = ["accelerator", "learning_rate", "epochs", "gradient_accumulation", "gradient_checkpointing", "max_steps", "checkpoint_every", "eval_trials", "seed"];
const ALGORITHM_KNOBS = {
  sft: ["finetuning_method", "train_merger_mlp", "multiview", "output_format", "batch_size", "cache_preprocessed", ...SHARED_KNOBS],
  grpo: ["rounds", "tasks_per_round", "group_size", "temperature", "clip_epsilon", ...SHARED_KNOBS],
};
ALGORITHM_KNOBS.gigpo = ALGORITHM_KNOBS.grpo;

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
      h("option", { value: "gigpo" }, "GIGPO（GRPO 兼容版）"),
      h("option", { value: "sft" }, "SFT（离线数据集）")),
    accelerator: h("select", {}, h("option", { value: "h800" }, "H800（按需创建 Pod，可并行）"), h("option", { value: "4090" }, "4090（常驻，单任务）")),
    finetuning_method: h("select", {}, h("option", { value: "full" }, "全参数"), h("option", { value: "lora" }, "LoRA（仅 SFT，FP32 adapter）")),
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
    eval_set: h("select", {}, catalog.task_sets.filter((s) => s.tasks.length <= 8).map((s) => h(
      "option", { value: s.name, selected: s.name === "sokoban-validation" }, taskSetLabel(s)))),
    rounds: h("input", { type: "number", min: 1, max: 20, value: 2 }),
    tasks_per_round: h("input", { type: "number", min: 1, max: 16, value: 4 }),
    group_size: h("input", { type: "number", min: 2, max: 16, value: 4 }),
    temperature: h("input", { type: "number", min: 0, max: 2, step: 0.1, value: 1.0 }),
    max_steps: h("input", { type: "number", min: 1, max: 50, value: 3 }),
    learning_rate: h("input", { type: "number", min: 1e-9, max: 1e-3, step: "any", value: 1e-7 }),
    epochs: h("input", { type: "number", min: 1, max: 10, value: 1 }),
    gradient_accumulation: h("input", { type: "number", min: 1, max: 64, value: 1 }),
    batch_size: h("input", { type: "number", min: 1, max: 16, value: 1 }),
    cache_preprocessed: h("input", { type: "checkbox" }),
    gradient_checkpointing: h("input", { type: "checkbox", checked: true }),
    clip_epsilon: h("input", { type: "number", min: 0.01, max: 1, step: 0.01, value: 0.2 }),
    eval_trials: h("input", { type: "number", min: 0, max: 50, value: 4 }),
    checkpoint_every: h("input", { type: "number", min: 1, max: 100, value: 5, placeholder: "只保留正式模型" }),
    seed: h("input", { type: "number", value: 0 }),
  };
  const knobLabels = {};
  const knob = (key, label, hint) => (knobLabels[key] = field(label, inputs[key], hint));
  const submit = h("button", { class: "primary", type: "submit", disabled: !models.length }, "创建实验");
  const form = h("form", { class: "form", onsubmit: async (event) => {
    event.preventDefault();
    const data = inputs.data.value;
    const backend = inputs.runtime_backend.value;
    const knobs = ALGORITHM_KNOBS[inputs.algorithm.value].filter((key) => !inputs[key].disabled).map((key) => {
      const input = inputs[key];
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
  field("算法", inputs.algorithm, "GIGPO 当前复用 GRPO 更新，不含论文 GiGPO 的步骤级分组"),
  knob("accelerator", "训练资源", "H800 每个实验独立 Pod，可并行，checkpoint 复制到 4090；在线 GRPO 每轮在 H800 训练、同步到 4090 后采样"),
  field("Runtime", inputs.runtime_backend, "由当前服务配置的 runtime worker 执行；Docker 不支持快照"),
  field("基础模型", inputs.base_model, models.length ? "也可以选择 checkpoint 继续训练" : "尚未注册模型，请先运行 import_artifacts"),
  field("模型产物名称", inputs.model_name, "可选；只允许字母、数字、点、下划线和短横线，round 会自动追加版本后缀"),
  field("数据集", inputs.data, "GRPO / GIGPO 选题目集时每轮抽题在线 rollout，选轨迹数据集时离线训练；SFT 只能选轨迹数据集。详见数据管理"),
  knob("finetuning_method", "微调方式", "LoRA：rank=16 / alpha=32，语言侧注意力层"),
  knob("train_merger_mlp", "全量训练视觉 merger", "仅 LoRA；额外全量训练视觉-语言投影层（MiniCPM-V：model.merger.mlp.0，约 2,597 万参数；Qwen3-VL：model.visual.merger）"),
  knob("multiview", "双视图输入", "完整截图 + 棋盘裁剪；推理按 checkpoint 配置自动使用"),
  knob("output_format", "输出协议", "必须与数据集一致；执行器仍只执行 action"),
  field("评测题目集", inputs.eval_set, "训练后自动评测用，最多 8 题；应与训练题目集不同，否则测不出泛化"),
  knob("rounds", "轮数"),
  knob("tasks_per_round", "每轮题目数", "按 seed 洗牌后依次取题，全部做完一遍才会重复；每轮 rollout 数 = 题目数 × group size"),
  knob("group_size", "每组 rollout 数 (group size)"),
  knob("temperature", "采样 temperature", "组内需要 reward 差异，建议 ≥ 0.7"), knob("max_steps", "每条 rollout 最大步数"),
  knob("learning_rate", "学习率", "1e-5 曾破坏输出格式，默认 1e-7"), knob("epochs", "epochs"),
  knob("gradient_accumulation", "梯度累积", "有效 batch = micro-batch × 梯度累积"), knob("clip_epsilon", "clip ε"),
  knob("batch_size", "micro-batch", "H800 加速配置为 4（配合梯度累积 1）；需确认显存"),
  knob("cache_preprocessed", "缓存预处理输入", "CPU 内存换速度；333 条双视图约需 10 GiB 以上"),
  knob("gradient_checkpointing", "梯度检查点", "关闭更快但显存更高；H800 加速配置为关闭"),
  knob("checkpoint_every", "checkpoint 保存间隔", "GRPO / GIGPO 每 N 轮、SFT 每 N 个 epoch 保留一个 checkpoint；最后一个总是正式模型；留空只保留正式模型"),
  knob("eval_trials", "训练后每任务评测次数", "0 表示不自动评测"), knob("seed", "随机 seed"),
  h("div", { class: "wide row" }, submit, message));
  const syncAlgorithm = () => {
    const visible = new Set(ALGORITHM_KNOBS[inputs.algorithm.value]);
    if (inputs.finetuning_method.value !== "lora") visible.delete("train_merger_mlp");

    // Disabled controls skip validation, so a hidden knob can never block submit.
    for (const [key, label] of Object.entries(knobLabels)) label.hidden = inputs[key].disabled = !visible.has(key);
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
      input.value = value;
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
    pageHeading("DATA REGISTRY / 02", "数据管理", "SFT 使用离线轨迹数据集；GRPO / GIGPO 每轮从题目集抽题在线 rollout，训练后在评测题目集上自动评测。"),
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
      h("p", { class: "muted" }, "单次评测最多 8 题，更大的题目集只能用于训练；训练和评测应选不同 split，避免在训练题上评测。"),
      table([
        { label: "名称", render: (s) => h("span", { class: "mono" }, s.name) },
        { label: "说明", key: "label" },
        { label: "用途", render: (s) => (s.tasks.length <= 8 ? "训练 / 评测" : "训练") },
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
      { label: "任务", render: (e) => e.task_set.map((t) => t.app_image).join(", ") },
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
  const accelerator = h("select", {}, h("option", { value: "4090" }, "4090（常驻模型服务）"),
    h("option", { value: "h800" }, "H800（按需部署所选模型）"));
  const formal = models.filter((m) => m.kind !== "checkpoint");
  const modelBoxes = models.map((m) => h("input", { type: "checkbox", value: m.version, checked: m === formal[formal.length - 1] }));
  const boxes = (rows) => h("div", { class: "checks" }, rows.map((m) => h("label", {}, modelBoxes[models.indexOf(m)], modelLabel(m))));
  const checkpoints = models.filter((m) => m.kind === "checkpoint");
  // A single evaluation takes at most 8 tasks, so the 48-level training set is not offered.
  const evalSets = catalog.task_sets.filter((s) => s.tasks.length <= 8);
  const taskSelect = h("select", {},
    h("optgroup", { label: "单个任务" }, catalog.tasks.map((t, i) => h("option", { value: `task:${i}` }, `${t.label}（${t.app_image}）`))),
    evalSets.length ? h("optgroup", { label: "题目集" }, evalSets.map((s) => h("option", { value: `set:${s.name}` }, taskSetLabel(s)))) : null);
  const selectedTasks = () => (taskSelect.value.startsWith("set:")
    ? evalSets.find((s) => s.name === taskSelect.value.slice(4)).tasks
    : [catalog.tasks[Number(taskSelect.value.slice(5))]]);
  // Every task of a set shares its type's instruction (one Sokoban instruction for all levels).
  const defaultInstruction = () => catalog.tasks.find((t) => t.task_type === selectedTasks()[0]?.task_type)?.instruction || "";
  const instruction = h("textarea", { value: catalog.tasks[0]?.instruction || "" });
  taskSelect.addEventListener("change", () => { instruction.value = defaultInstruction(); });
  const trials = h("input", { type: "number", min: 1, max: 50, value: 1 });
  const maxSteps = h("input", { type: "number", min: 1, max: 50, value: 5 });
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
  h("div", { class: "wide" }, h("div", { class: "secondary" }, "模型 / checkpoint（多选即为对比评测，所有模型使用相同 seed）"),
    boxes(formal),
    checkpoints.length ? h("details", {}, h("summary", { class: "secondary" }, `Checkpoint（${checkpoints.length} 个）`), boxes(checkpoints)) : null),
  field("评测名称", name, "可选；不填写时自动生成名称"),
  field("Runtime", runtimeBackend, "Firecracker 使用 Android microVM；Docker 使用 Android 容器"),
  field("推理资源", accelerator, "H800 为本次评测单独创建 Pod 并预加载所选模型，加载完成后 trial 才开始；评测结束自动删除"),
  field("任务 / 题目集", taskSelect, "题目集的每一题都会跑；训练关卡 48 题超过单次评测上限 8 题，不能选"),
  field("每题 trial 次数", trials, "temperature 为 0 时同一题重复 trial 结果基本相同，要增加样本请用题目集；对比成功率建议总 trial ≥ 20"),
  field("最大步数", maxSteps), field("temperature", temperature, "0 为贪心解码"), field("起始 seed", seed),
  h("label", { class: "wide" }, "指令（可修改）", instruction),
  h("div", { class: "wide row" }, submit, message));
  return h("details", { class: "card" }, h("summary", {}, "新建模型评测"), h("div", { style: "margin-top:12px" }, form));
}

function inferenceLabel(ev) {
  if (ev.accelerator !== "h800") return "4090";
  if (ev.model_ready) return "H800（模型已就绪）";
  return ev.status === "RUNNING" ? "H800（部署中）" : "H800";
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
  await poll(async () => {
    const ev = await api(`/api/v1/evaluations/${evalId}`);
    root.replaceChildren(
      h("div", { class: "row" }, h("h2", {}, ev.name || `评测 ${fmt.short(ev.eval_id)}`), badge(ev.status), h("div", { class: "spacer" }),
        evalActions(ev, "#/evaluations")),
      h("div", { class: "card" },
        h("div", { class: "secondary" }, `Runtime ${ev.runtime_backend || "unknown"} · 推理 ${inferenceLabel(ev)} · 任务 ${ev.task_set.map((t) => `${t.task_type}/${t.app_image}`).join(", ")} · 每模型 ${ev.trials_per_task} 次 · 最多 ${ev.max_steps} 步 · temperature ${ev.temperature} · seeds ${ev.seeds.join(", ")}`),
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

function modelOutput(raw) {
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
          h("dt", {}, "模型输出"), h("dd", { class: "mono" }, modelOutput(s.model_output?.raw_output)),
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

function guideStep(number, title, description, meta) {
  return h("article", { class: "guide-step card" },
    h("div", { class: "step-number" }, number),
    h("div", {}, h("h3", {}, title), h("p", {}, description), h("span", { class: "step-meta" }, meta)));
}

function architectureNode(x, y, width, height, title, details, kind = "") {
  return h("g", { class: `arch-node ${kind}` },
    h("rect", { x, y, width, height, rx: 12 }),
    h("text", { x: x + 16, y: y + 29, class: "arch-title" }, title),
    details.map((detail, index) => h("text", { x: x + 16, y: y + 53 + index * 18, class: "arch-detail" }, detail)));
}

function architectureArrow(x1, y1, x2, y2, kind = "") {
  return h("line", { x1, y1, x2, y2, class: `arch-arrow ${kind}`, "marker-end": "url(#arch-arrow)" });
}

function architectureDiagram() {
  const svg = h("svg", {
    class: "architecture-svg",
    viewBox: "0 0 1080 620",
    role: "img",
    "aria-label": "AlvarezGUI 从控制台到 Android sandbox 和训练服务的架构图",
  });
  svg.append(h("defs", {},
    h("marker", { id: "arch-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" },
      h("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "context-stroke" }))));
  svg.append(
    architectureArrow(222, 112, 270, 112),
    architectureArrow(510, 112, 570, 112),
    architectureArrow(760, 112, 820, 112),
    architectureArrow(390, 195, 390, 265, "secondary"),
    architectureArrow(930, 195, 930, 270),
    architectureArrow(820, 150, 260, 300, "model-link"),
    architectureArrow(150, 410, 150, 500, "secondary"),
    architectureArrow(260, 540, 300, 540, "secondary"),
    architectureArrow(820, 340, 790, 410),
    architectureArrow(820, 320, 490, 540, "secondary"),
    architectureNode(30, 70, 192, 84, "实验控制台", ["创建实验 · 查看轨迹", "资源 / 训练 / 评测"], "user"),
    architectureNode(270, 45, 240, 150, "Control Plane", ["FastAPI + Scheduler", "任务状态 · Lease · API", "runtime backend 路由"], "control"),
    architectureNode(570, 45, 190, 90, "Redis Streams", ["任务队列 · Consumer Group"], "queue"),
    architectureNode(820, 45, 220, 150, "Consumer Worker Pool", ["并发 sandbox worker", "XAUTOCLAIM · ACK", "动作与 reward 记录"], "worker"),
    architectureNode(300, 265, 190, 90, "PostgreSQL", ["实验 · 任务 · 轨迹", "指标与 checkpoint 元数据"], "storage"),
    architectureNode(30, 300, 230, 110, "Model Service", ["MiniCPM-V / Qwen3-VL HTTP API", "截图 → 结构化动作", "GPU / max-loaded=1"], "model"),
    architectureNode(30, 500, 230, 80, "Training Agent", ["SFT / GRPO / GIGPO · 训练后评测"], "training"),
    architectureNode(300, 500, 190, 80, "Artifacts", ["checkpoint · rollout", "可复现运行记录"], "storage"),
    architectureNode(820, 270, 220, 100, "Runtime Layer", ["Firecracker / Docker Android", "sandbox 生命周期"], "runtime"),
    architectureNode(560, 410, 230, 110, "Android sandbox", ["ADB · screenshot · input", "本地 fixture 应用", "backend 可配置"], "vm"),
  );
  return h("div", { class: "architecture card" },
    h("div", { class: "section-kicker" }, "SYSTEM MAP"),
    h("h3", {}, "AlvarezGUI 执行架构"),
    h("p", { class: "section-lead" }, "一条 rollout 从控制台进入队列，经 Consumer 分配隔离 runtime，在 Android sandbox 中执行，再把轨迹送回模型和训练链路。"),
    svg,
    h("div", { class: "diagram-legend" },
      h("span", {}, h("i", { class: "legend-dot cyan" }), "控制面"),
      h("span", {}, h("i", { class: "legend-dot violet" }), "模型与训练"),
      h("span", {}, h("i", { class: "legend-dot green" }), "Android runtime"),
      h("span", {}, h("i", { class: "legend-dot gray" }), "持久化与队列")));
}

function guidePage() {
  view.replaceChildren(
    pageHeading("PLAYBOOK / 05", "从实验到结果", "用 AlvarezGUI 管理模型、runtime、rollout 和评测。下面的流程图对应平台当前的真实执行链路。"),
    h("div", { class: "guide-banner card" },
      h("div", {}, h("div", { class: "eyebrow" }, "QUICK START"), h("h3", {}, "四步启动一次可复现实验"), h("p", {}, "先准备模型和数据，再选择 runtime。在线 rollout 会保留 seed、动作、截图和 reward，方便定位每一步发生了什么。")),
      h("a", { class: "primary action-link", href: "#/runs" }, "创建训练实验 →")),
    h("div", { class: "guide-steps" },
      guideStep("01", "准备模型与数据", "在数据管理页查看轨迹数据集和题目集；训练实验页选择基础模型、训练数据和评测题目集。", "模型版本 · 数据集 · 题目集"),
      guideStep("02", "使用 Runtime", "平台按配置选择 Firecracker 或 Docker，为每条 rollout 分配独立 Android sandbox。", "runtime_backend · sandbox capacity"),
      guideStep("03", "观察 rollout", "Scheduler 将任务发布到对应 Redis Stream，Consumer worker 并发执行动作，并持续更新 lease 和轨迹。", "queue · worker pool · lease"),
      guideStep("04", "分析训练结果", "训练 agent 汇总 reward 和指标，GRPO 只对有 reward 方差的 group 更新模型，训练后可自动发起评测。", "reward · metrics · checkpoint")),
    architectureDiagram(),
    h("div", { class: "guide-columns" },
      h("section", { class: "card guide-section" },
        h("div", { class: "section-kicker" }, "RUNTIME GUIDE"),
        h("h3", {}, "Runtime Guide"),
        h("div", { class: "runtime-choice" },
          h("div", { class: "choice-card firecracker" }, h("span", { class: "choice-label" }, "DEFAULT PATH"), h("strong", {}, "Firecracker / Docker Android"), h("p", {}, "每条 rollout 使用独立 Android sandbox。Firecracker 提供 microVM 隔离与快照；Docker 适合 Windows Docker Desktop，本身不提供快照。"))),
        h("div", { class: "callout" }, h("strong", {}, "运行原则"), " 新建训练、rollout 和评测统一进入配置的 backend stream；历史 fake 仅用于审计和回放。")),
      h("section", { class: "card guide-section" },
        h("div", { class: "section-kicker" }, "READ THE RUN"),
        h("h3", {}, "看懂一次训练实验"),
        h("ol", { class: "guide-list" },
          h("li", {}, h("strong", {}, "状态"), "：QUEUED → RUNNING → SUCCEEDED / FAILED"),
          h("li", {}, h("strong", {}, "Runtime"), "：请求 backend 与 actual backend 都会展示，便于确认实际执行环境"),
          h("li", {}, h("strong", {}, "Reward"), "：同一 group 全部相同会显示零方差提示，不产生有效梯度"),
          h("li", {}, h("strong", {}, "轨迹"), "：点击 rollout 可回放每一步截图、动作、模型原始输出和延迟"),
          h("li", {}, h("strong", {}, "资源"), "：资源总览展示节点心跳、GPU、KVM、sandbox 和 lease")))),
    h("div", { class: "guide-footer card" }, h("span", { class: "live-dot" }), "所有动态数据来自平台 API；页面每 3–5 秒自动刷新，教程内容本身不依赖后端状态。"));
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
