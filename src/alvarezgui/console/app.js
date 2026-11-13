// AlvarezGUI console. All dynamic text goes through textContent (h()), never innerHTML.

const view = document.getElementById("view");
let pollTimer = null;
let routeToken = 0;

// ---------------------------------------------------------------- helpers

function h(tag, attrs = {}, ...children) {
  const isSvg = ["svg", "line", "path", "circle", "rect", "text", "g", "polyline"].includes(tag);
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
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (response.status === 401) { location.href = "/console/login"; throw new Error("未登录"); }
  const body = response.headers.get("content-type")?.includes("json") ? await response.json() : null;
  if (!response.ok) {
    const detail = body?.detail;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail ?? response.statusText));
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
function badge(status) {
  const [kind, label] = STATUS[status] || ["", status];
  return h("span", { class: `badge ${kind}`, title: status }, label);
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
  view.replaceChildren(h("h2", {}, "资源总览"), root);
  await poll(async () => {
    const data = await api("/api/v1/overview");
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
            ? `已加载 ${(model.loaded_versions || []).join(", ") || "无"} · p50 ${fmt.ms(model.latency_p50_ms)} · 拒绝 ${model.rejected_outputs ?? "–"}`
            : `后端 ${model.backend}`),
          model.error ? h("div", { class: "sub error-text" }, model.error) : null)),
      h("h3", {}, "节点"),
      h("div", { class: "card" }, nodesTable(data.nodes)),
      h("h3", {}, "Sandbox"),
      h("div", { class: "card" }, sandboxTable(data.sandboxes)),
    );
  }, 5000);
}

function nodesTable(nodes) {
  return table([
    { label: "节点", render: (n) => h("div", {}, h("div", {}, n.node_id), h("div", { class: "muted" }, n.address || "")) },
    { label: "角色", render: (n) => (n.role === "gpu" ? "GPU 训练/推理" : "Android microVM") },
    { label: "状态", render: (n) => badge(n.status) },
    { label: "资源", render: (n) => (n.role === "gpu" ? gpuCell(n.resources) : hostCell(n.resources)) },
    { label: "Sandbox", num: true, render: (n) => (n.role === "gpu" ? (n.resources.active_run ? fmt.short(n.resources.active_run) : "空闲") : `${n.active_sandboxes} / ${n.capacity.sandboxes ?? "?"}`) },
    { label: "最近心跳", render: (n) => fmt.time(n.last_heartbeat_at) },
  ], nodes);
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
  "train/loss": ["loss", "update"],
  "train/grad_norm": ["梯度范数", "update"],
  "train/ratio": ["策略比 ratio", "update"],
  "train/clipped": ["是否被 clip", "update"],
  "train/advantage": ["advantage", "update"],
  "train/learning_rate": ["学习率", "update", (v) => Number(v).toExponential(1)],
  "gpu/utilization_percent": ["GPU 利用率 %", "秒", (v) => fmt.num(v, 0)],
  "gpu/memory_used_mib": ["显存 MiB", "秒", (v) => fmt.num(v, 0)],
};

async function runsPage() {
  const catalog = await api("/api/v1/catalog");
  const list = h("div", { class: "card" });
  view.replaceChildren(
    h("div", { class: "row" }, h("h2", {}, "训练实验")),
    runForm(catalog),
    h("h3", {}, "实验列表"),
    list,
  );
  await poll(async () => {
    const runs = await api("/api/v1/training/runs");
    list.replaceChildren(table([
      { label: "实验", render: (r) => h("a", { href: `#/runs/${r.run_id}` }, fmt.short(r.run_id)) },
      { label: "算法", render: (r) => r.algorithm.toUpperCase() },
      { label: "基础模型", key: "base_model" },
      { label: "数据", render: (r) => r.dataset_version || (r.config.tasks || []).map((t) => t.app_image).join(", ") || "–" },
      { label: "状态", render: (r) => badge(r.status) },
      { label: "reward 均值", num: true, render: (r) => fmt.num(r.metrics["rollout/reward_mean"] ?? r.metrics.reward_mean) },
      { label: "产出模型", render: (r) => r.metrics.model_version || "–" },
      { label: "创建时间", render: (r) => fmt.time(r.created_at) },
    ], runs, (r) => { location.hash = `#/runs/${r.run_id}`; }));
  }, 5000);
}

function field(label, input, hint) {
  return h("label", {}, label, input, hint ? h("span", { class: "hint" }, hint) : null);
}

function runForm(catalog) {
  const models = catalog.model_versions;
  const message = h("div");
  const inputs = {
    algorithm: h("select", {}, h("option", { value: "grpo" }, "GRPO（在线 rollout）"), h("option", { value: "sft" }, "SFT（离线数据集）")),
    base_model: h("select", {}, models.map((m) => h("option", { value: m.version }, m.version))),
    dataset_version: h("select", {}, h("option", { value: "" }, "不使用（在线 rollout）"), catalog.datasets.map((d) => h("option", { value: d.dataset_version }, `${d.dataset_version}（${d.sample_count} 条）`))),
    rounds: h("input", { type: "number", min: 1, max: 20, value: 2 }),
    group_size: h("input", { type: "number", min: 2, max: 16, value: 4 }),
    temperature: h("input", { type: "number", min: 0, max: 2, step: 0.1, value: 1.0 }),
    max_steps: h("input", { type: "number", min: 1, max: 50, value: 3 }),
    learning_rate: h("input", { type: "number", min: 1e-9, max: 1e-3, step: "any", value: 1e-7 }),
    epochs: h("input", { type: "number", min: 1, max: 10, value: 1 }),
    gradient_accumulation: h("input", { type: "number", min: 1, max: 64, value: 1 }),
    clip_epsilon: h("input", { type: "number", min: 0.01, max: 1, step: 0.01, value: 0.2 }),
    eval_trials: h("input", { type: "number", min: 0, max: 50, value: 4 }),
    seed: h("input", { type: "number", value: 0 }),
  };
  const taskBoxes = catalog.tasks.map((t, i) => h("input", { type: "checkbox", checked: i === 0, value: i }));
  const submit = h("button", { class: "primary", type: "submit", disabled: !models.length }, "创建实验");
  const form = h("form", { class: "form", onsubmit: async (event) => {
    event.preventDefault();
    const tasks = taskBoxes.filter((b) => b.checked).map((b) => ({ task_type: catalog.tasks[b.value].task_type, app_image: catalog.tasks[b.value].app_image }));
    const num = (k) => Number(inputs[k].value);
    const body = {
      algorithm: inputs.algorithm.value,
      base_model: inputs.base_model.value,
      dataset_version: inputs.dataset_version.value || null,
      environment_version: "android-firecracker-v1",
      reward_version: "android-task-v1",
      config: {
        rounds: num("rounds"), group_size: num("group_size"), temperature: num("temperature"), max_steps: num("max_steps"),
        learning_rate: num("learning_rate"), epochs: num("epochs"), gradient_accumulation: num("gradient_accumulation"),
        clip_epsilon: num("clip_epsilon"), eval_trials: num("eval_trials"), seed: num("seed"), tasks,
      },
    };
    submit.disabled = true;
    try {
      const run = await api("/api/v1/training/runs", { method: "POST", body: JSON.stringify(body), headers: { "Idempotency-Key": idempotencyKey() } });
      location.hash = `#/runs/${run.run_id}`;
    } catch (error) {
      message.replaceChildren(h("p", { class: "error-text" }, `创建失败：${error.message}`));
    } finally { submit.disabled = false; }
  } },
  field("算法", inputs.algorithm),
  field("基础模型 / checkpoint", inputs.base_model, models.length ? "可以选择已有 checkpoint 继续训练" : "尚未注册模型，请先运行 import_artifacts"),
  field("数据集", inputs.dataset_version, "SFT 必选；GRPO 不选时在线采样"),
  h("div", { class: "wide" }, h("div", { class: "secondary" }, "训练任务（GRPO 在线采样与训练后自动评测）"), h("div", { class: "checks" },
    catalog.tasks.map((t, i) => h("label", {}, taskBoxes[i], `${t.label}（${t.app_image}）`)))),
  field("轮数", inputs.rounds), field("每组 rollout 数 (group size)", inputs.group_size),
  field("采样 temperature", inputs.temperature, "组内需要 reward 差异，建议 ≥ 0.7"), field("每条 rollout 最大步数", inputs.max_steps),
  field("学习率", inputs.learning_rate, "1e-5 曾破坏输出格式，默认 1e-7"), field("epochs", inputs.epochs),
  field("梯度累积", inputs.gradient_accumulation), field("clip ε", inputs.clip_epsilon),
  field("训练后每任务评测次数", inputs.eval_trials, "0 表示不自动评测"), field("随机 seed", inputs.seed),
  h("div", { class: "wide row" }, submit, message));
  return h("details", { class: "card" }, h("summary", {}, "新建训练实验"), h("div", { style: "margin-top:12px" }, form));
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
      h("div", { class: "row" }, h("h2", {}, `${run.algorithm.toUpperCase()} 实验 ${fmt.short(run.run_id)}`), badge(run.status), h("div", { class: "spacer" }),
        active ? h("button", { class: "danger", onclick: async () => {
          if (!confirm("取消该实验？排队中的 rollout 也会被取消。")) return;
          try { await api(`/api/v1/training/runs/${runId}/cancel`, { method: "POST" }); } catch (e) { alert(e.message); }
        } }, "取消实验") : null),
      h("div", { class: "card" }, h("dl", { class: "kv" },
        h("dt", {}, "基础模型"), h("dd", {}, run.base_model),
        h("dt", {}, "数据"), h("dd", {}, run.dataset_version || (run.config.tasks || []).map((t) => `${t.task_type}/${t.app_image}`).join(", ")),
        h("dt", {}, "配置"), h("dd", { class: "mono" }, JSON.stringify({ ...run.config, tasks: undefined })),
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
          { label: "动作序列", render: (r) => (r.step_records || []).map((s) => describeAction(s.action)).join(" → ") || r.error || "–" },
        ], members, (r) => { location.hash = `#/episodes/${r.episode_id}`; }));
    }))));
}

// ---------------------------------------------------------------- evaluations

async function evaluationsPage() {
  const catalog = await api("/api/v1/catalog");
  const list = h("div", { class: "card" });
  view.replaceChildren(h("h2", {}, "模型评测"), evalForm(catalog), h("h3", {}, "评测记录"), list);
  await poll(async () => {
    const rows = await api("/api/v1/evaluations");
    list.replaceChildren(table([
      { label: "评测", render: (e) => h("a", { href: `#/evaluations/${e.eval_id}` }, fmt.short(e.eval_id)) },
      { label: "模型", render: (e) => e.model_versions.join(" vs ") },
      { label: "任务", render: (e) => e.task_set.map((t) => t.app_image).join(", ") },
      { label: "状态", render: (e) => badge(e.status) },
      { label: "成功 / trial", num: true, render: (e) => `${e.successes} / ${e.trials}` },
      { label: "来源", render: (e) => (e.run_id ? h("a", { href: `#/runs/${e.run_id}` }, "训练后自动评测") : "手动") },
      { label: "创建时间", render: (e) => fmt.time(e.created_at) },
    ], rows, (e) => { location.hash = `#/evaluations/${e.eval_id}`; }));
  }, 5000);
}

function evalForm(catalog) {
  const message = h("div");
  const modelBoxes = catalog.model_versions.map((m, i) => h("input", { type: "checkbox", value: m.version, checked: i === catalog.model_versions.length - 1 }));
  const taskSelect = h("select", {}, catalog.tasks.map((t, i) => h("option", { value: i }, `${t.label}（${t.app_image}）`)));
  const instruction = h("textarea", { value: catalog.tasks[0]?.instruction || "" });
  taskSelect.addEventListener("change", () => { instruction.value = catalog.tasks[taskSelect.value].instruction; });
  const trials = h("input", { type: "number", min: 1, max: 50, value: 1 });
  const maxSteps = h("input", { type: "number", min: 1, max: 50, value: 5 });
  const temperature = h("input", { type: "number", min: 0, max: 2, step: 0.1, value: 0 });
  const seed = h("input", { type: "number", value: 0 });
  const submit = h("button", { class: "primary", type: "submit", disabled: !catalog.model_versions.length }, "开始评测");
  const form = h("form", { class: "form", onsubmit: async (event) => {
    event.preventDefault();
    const versions = modelBoxes.filter((b) => b.checked).map((b) => b.value);
    if (!versions.length) { message.replaceChildren(h("p", { class: "error-text" }, "至少选择一个模型")); return; }
    const task = catalog.tasks[taskSelect.value];
    const body = {
      model_versions: versions,
      tasks: [{ task_type: task.task_type, app_image: task.app_image, instruction: instruction.value.trim() || null }],
      trials_per_task: Number(trials.value), max_steps: Number(maxSteps.value),
      temperature: Number(temperature.value), seed: Number(seed.value),
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
    h("div", { class: "checks" }, catalog.model_versions.map((m, i) => h("label", {}, modelBoxes[i], m.version)))),
  field("任务", taskSelect),
  field("每个模型的 trial 次数", trials, "选 1 次可实时观看单次执行；对比成功率建议 ≥ 20"),
  field("最大步数", maxSteps), field("temperature", temperature, "0 为贪心解码"), field("起始 seed", seed),
  h("label", { class: "wide" }, "指令（可修改）", instruction),
  h("div", { class: "wide row" }, submit, message));
  return h("div", { class: "card" }, form);
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
      h("div", { class: "row" }, h("h2", {}, `评测 ${fmt.short(ev.eval_id)}`), badge(ev.status)),
      h("div", { class: "card" },
        h("div", { class: "secondary" }, `任务 ${ev.task_set.map((t) => `${t.task_type}/${t.app_image}`).join(", ")} · 每模型 ${ev.trials_per_task} 次 · 最多 ${ev.max_steps} 步 · temperature ${ev.temperature} · seeds ${ev.seeds.join(", ")}`),
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
    if (ev.status === "FINISHED") clearInterval(pollTimer);
  }, 3000);
}

// ---------------------------------------------------------------- episodes

async function episodeView(episodeId) {
  const ep = await api(`/api/v1/episodes/${episodeId}`);
  const base = `/api/v1/episodes/${encodeURIComponent(episodeId)}`;
  const finalReward = ep.result?.final_reward;
  return h("div", { "data-active": ["queued", "running"].includes(ep.outcome) ? "1" : "0" },
    h("div", { class: "card" },
      h("div", { class: "row" }, badge(ep.outcome), h("strong", {}, `${ep.task_type} · ${ep.policy_version}`),
        h("span", { class: "muted" }, `seed ${ep.seed ?? "–"} · temperature ${ep.temperature} · ${ep.steps.length} / ${ep.max_steps} 步`)),
      h("dl", { class: "kv", style: "margin-top:8px" },
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
          h("dt", {}, "模型输出"), h("dd", { class: "mono" }, s.model_output?.raw_output ?? "（固定策略）"),
          h("dt", {}, "延迟"), h("dd", {}, fmt.ms(s.model_output?.latency_ms)),
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

// ---------------------------------------------------------------- router

const routes = [
  [/^#\/resources$/, resourcesPage, "resources"],
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
