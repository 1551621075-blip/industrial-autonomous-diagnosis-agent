"use strict";
const el = id => document.getElementById(id);
let token = "", currentActor = null, currentTask = null, currentReport = null, cases = [], versions = [], reports = [];
let session = 0, busy = false;
const uid = prefix => prefix + "-" + crypto.randomUUID().replaceAll("-", "");
const show = (id, value) => { el(id).textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2); };
function message(text) { show("message", text); el("message").style.display = "block"; }
async function api(path, body) {
  if (!token) throw new Error("请先验证本地身份。");
  const epoch = session;
  const response = await fetch(path, {method: body === undefined ? "GET" : "POST", cache: "no-store", credentials: "omit",
    headers: {"Authorization": "Bearer " + token, ...(body !== undefined ? {"Content-Type": "application/json"} : {})},
    ...(body !== undefined ? {body: JSON.stringify(body)} : {})});
  const data = await response.json();
  if (epoch !== session) throw new Error("身份已更换，旧请求结果已丢弃。");
  if (!response.ok) throw new Error((data.error || "request_failed") + ": " + (data.message || "请求失败"));
  return data;
}
function updateControls() {
  const authenticated = !!token && !!currentActor;
  const admin = currentActor?.role === "admin", author = admin || currentActor?.role === "analyst";
  const terminal = ["submitted", "escalated", "cancelled"].includes(currentTask?.status);
  document.querySelectorAll("button").forEach(button => {
    if (button.id === "logout") { button.disabled = false; return; }
    if (button.closest("#login")) { button.disabled = busy; return; }
    let allowed = authenticated;
    const form = button.closest("form")?.id;
    if (form === "build") allowed &&= author;
    if (form === "publish") allowed &&= admin;
    if (form === "signoff") allowed &&= author && !!currentReport && !currentReport.signoff;
    if (form === "reconcile") allowed &&= admin && !!currentTask?.pending_call_id && !terminal;
    if (form === "answer") allowed &&= currentTask?.status === "waiting_human";
    if (["resume", "cancel"].includes(button.id)) allowed &&= !!currentTask && !terminal;
    if (button.id === "resume") allowed &&= currentTask?.status !== "waiting_human" && !currentTask?.pending_call_id;
    if (["make-report", "refresh-reports", "refresh-audit"].includes(button.id)) allowed &&= !!currentTask;
    if (button.id === "make-report") allowed &&= terminal;
    button.disabled = busy || !allowed;
  });
}
function on(id, event, fn) {
  el(id).addEventListener(event, async e => {
    e.preventDefault();
    if (busy) return;
    if (id === "login") session += 1;
    const epoch = session;
    busy = true; updateControls();
    try { await fn(); } catch (error) { if (epoch === session) message(error.message); }
    finally { busy = false; updateControls(); }
  });
}
function choices(id, rows, value, label) {
  const select = el(id); select.replaceChildren();
  const empty = document.createElement("option"); empty.value = ""; empty.textContent = "请选择"; select.append(empty);
  rows.forEach(row => { const option = document.createElement("option"); option.value = value(row); option.textContent = label(row); select.append(option); });
}
function requireTask() { if (!currentTask) throw new Error("请先选择任务。"); return currentTask.task_id; }
function drawTask(task) {
  if (currentTask?.task_id !== task.task_id) {
    currentReport = null; reports = []; choices("reports", [], r => r, r => r);
    show("report", "请加载当前任务报告。"); show("audit", "请加载当前任务审计。");
  }
  if (currentTask?.pending_human?.question_id !== task.pending_human?.question_id) el("answer-value").value = "";
  currentTask = task;
  show("task-status", `${task.task_id} · ${task.status}${task.error ? " · " + task.error.code : ""}`);
  show("budget", `模型调用 ${task.calls}/${task.limits.calls} · 工具 ${task.tools}/${task.limits.tools} · token ${task.tokens}/${task.limits.tokens} · 图内累计 ${task.elapsed_s.toFixed(2)} 秒 · 压缩 ${task.compaction_count} 次`);
  el("human").hidden = task.status !== "waiting_human";
  show("question", task.pending_human ? task.pending_human.question + "\n" + task.pending_human.instruction : "");
  show("human-predicate", task.pending_human ? {predicate: task.pending_human.predicate, tree_version: task.pending_human.tree_version, tree_hash: task.pending_human.tree_hash} : "");
  el("evidence").replaceChildren();
  task.evidence.forEach(item => {
    const row = document.createElement("tr");
    [item.condition_id, item.state, item.source, JSON.stringify(item.details), item.evidence_id].forEach((text, i) => {
      const cell = document.createElement("td"); cell.textContent = text;
      if (i === 1) cell.className = item.state;
      row.append(cell);
    }); el("evidence").append(row);
  });
  show("submission", task.submission || "尚未通过提交校验。");
  el("call-id").value = task.pending_call_id || "";
}
async function loadCases() {
  cases = await api("/api/cases");
  choices("cases", cases, c => c.event_id + ":" + c.device_id + ":" + c.plant_id, c => `${c.case_label} · ${c.event_id} · ${c.plant_id}（合成标签仅供演示）`);
}
async function loadKnowledge() {
  versions = await api("/api/knowledge");
  choices("versions", versions, v => v.version_id, v => `${v.status} · ${v.version_id}`);
}
async function loadTasks() {
  const tasks = await api("/api/tasks"); choices("tasks", tasks, t => t.task_id, t => `${t.status} · ${t.request.event_id} · ${t.task_id}`);
}
async function loadReports() {
  reports = await api(`/api/tasks/${requireTask()}/reports`);
  choices("reports", reports, r => r.report_id, r => `${r.status} · ${r.report_id}`);
}
function clearSessionView() {
  currentActor = null; currentTask = null; currentReport = null; cases = []; versions = []; reports = [];
  ["cases", "versions", "tasks", "reports"].forEach(id => choices(id, [], x => x, x => x));
  ["identity", "knowledge", "submission", "audit", "report", "task-status", "budget", "question", "human-predicate"].forEach(id => show(id, "已清除本地身份及页面数据。"));
  ["complaint", "observation", "review-note", "signoff-note", "call-id", "reconcile-note", "reconcile-tokens", "knowledge-file"].forEach(id => { el(id).value = ""; });
  el("no-result").checked = false; el("answer-value").value = ""; el("evidence").replaceChildren(); el("human").hidden = true;
}
el("logout").addEventListener("click", () => {
  token = ""; session += 1; el("token").value = ""; clearSessionView(); updateControls();
  message("令牌已清除。服务器上已开始的只读任务不会因页面退出而被强行中断。");
});
on("login", "submit", async () => {
  token = el("token").value.trim(); el("token").value = "";
  clearSessionView();
  let actor;
  try { actor = await api("/api/me"); } catch (error) { token = ""; throw error; }
  currentActor = actor;
  show("identity", `${actor.actor_id} · ${actor.role} · ${actor.plants.join(", ")}`);
  await loadCases(); await loadKnowledge(); await loadTasks(); message("本地身份已验证。");
});
on("refresh-cases", "click", loadCases);
on("refresh-knowledge", "click", loadKnowledge);
on("refresh-tasks", "click", loadTasks);
on("cases", "change", async () => {
  const item = cases.find(c => c.event_id + ":" + c.device_id + ":" + c.plant_id === el("cases").value);
  el("complaint").value = item ? item.complaint : "";
});
on("versions", "change", async () => {
  const version = versions.find(v => v.version_id === el("versions").value); show("knowledge", version || "未选择版本。");
});
on("build", "submit", async () => {
  const file = el("knowledge-file").files[0]; if (!file || file.size > 256000) throw new Error("请选择不超过256KB的知识构建JSON。");
  const built = await api("/api/knowledge/build", JSON.parse(await file.text())); show("knowledge", built); await loadKnowledge();
  el("versions").value = built.version_id; message("知识构建结果已记录，仍需人工检查后发布。");
});
on("publish", "submit", async () => {
  const version = versions.find(v => v.version_id === el("versions").value); if (!version) throw new Error("请选择需要审核的知识版本。");
  const result = await api("/api/knowledge/publish", {action_id: uid("publish"), version_id: version.version_id, content_hash: version.content_hash, review_note: el("review-note").value});
  show("knowledge", result); await loadKnowledge(); message("精确知识版本已人工发布。");
});
on("new-task", "submit", async () => {
  const item = cases.find(c => c.event_id + ":" + c.device_id + ":" + c.plant_id === el("cases").value);
  if (!item) throw new Error("请选择事件。");
  const task = await api("/api/tasks", {request_id: uid("request"), plant_id: item.plant_id, device_id: item.device_id, event_id: item.event_id, window: item.window, complaint: el("complaint").value});
  drawTask(task); drawTask(await api(`/api/tasks/${task.task_id}/run`, {})); await loadTasks(); message("本次诊断执行已返回。");
});
on("tasks", "change", async () => { if (el("tasks").value) { drawTask(await api(`/api/tasks/${el("tasks").value}`)); currentReport = null; show("report", "请加载当前任务报告。"); } });
on("resume", "click", async () => drawTask(await api(`/api/tasks/${requireTask()}/run`, {})));
on("cancel", "click", async () => drawTask(await api(`/api/tasks/${requireTask()}/cancel`, {action_id: uid("cancel")})));
on("answer", "submit", async () => {
  const id = requireTask(); if (!currentTask.pending_human) throw new Error("当前没有待回答问题。");
  if (!["true", "false"].includes(el("answer-value").value)) throw new Error("请明确选择已审核观察事实的真假。");
  drawTask(await api(`/api/tasks/${id}/answer`, {action_id: uid("answer"), question_id: currentTask.pending_human.question_id, answer: el("answer-value").value === "true", observation_note: el("observation").value}));
  el("observation").value = ""; drawTask(await api(`/api/tasks/${id}/run`, {}));
});
on("refresh-audit", "click", async () => show("audit", await api(`/api/tasks/${requireTask()}/audit`)));
on("make-report", "click", async () => {
  currentReport = await api(`/api/tasks/${requireTask()}/reports`, {action_id: uid("report")}); show("report", currentReport); await loadReports();
});
on("refresh-reports", "click", loadReports);
on("reports", "change", async () => { if (el("reports").value) { currentReport = await api(`/api/reports/${el("reports").value}`); show("report", currentReport); } });
on("signoff", "submit", async () => {
  if (!currentReport) throw new Error("请选择报告。");
  const result = await api(`/api/reports/${currentReport.report_id}/review`, {action_id: uid("review"), content_hash: currentReport.content_hash, decision: el("decision").value, note: el("signoff-note").value});
  currentReport.signoff = result;
  show("report", result); message("审签已记录；不会触发设备操作。");
});
on("reconcile", "submit", async () => {
  const raw = el("reconcile-tokens").value.trim();
  const usage = raw === "" ? {} : {tokens: Number(raw)};
  if (raw !== "" && (!Number.isInteger(usage.tokens) || usage.tokens < 0 || usage.tokens > 100000)) throw new Error("用量必须为 0–100000 的已核实整数，未知请留空。");
  drawTask(await api(`/api/tasks/${requireTask()}/reconcile`, {action_id: uid("reconcile"), call_id: el("call-id").value, no_result: true, ...usage, note: el("reconcile-note").value}));
  el("reconcile-tokens").value = "";
  message("对账完成。已知用量保留，未知用量保守占用剩余预算；可显式继续检查终态。");
});
updateControls();
fetch("/health", {cache: "no-store", credentials: "omit"}).then(r => r.json()).then(value => show("mode", value.provider_mode === "test-only" ? "TEST-ONLY · 测试提供器" : "LOCAL MODEL · 本地模型")).catch(() => show("mode", "本地服务不可用"));
