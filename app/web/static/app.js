"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const form = byId("settings-form");
  const fields = byId("settings-fields");
  const target = byId("target-group");
  const model = byId("model-input");
  const style = byId("chat-style");
  const limit = byId("reply-limit");
  const modeInputs = [...form.querySelectorAll('input[name="mode"]')];
  const buttonIds = ["power-button", "save-button", "clear-button", "refresh-models"];
  let status = null;
  let dirty = false;
  let busy = false;
  let polling = false;
  let modelsLoading = false;
  let apiAvailable = false;
  let activitySignature = null;
  let noticeTimer;
  let mutationVersion = 0;

  async function request(path, options = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 25000);
    try {
      const response = await fetch(path, {
        cache: "no-store",
        credentials: "same-origin",
        signal: controller.signal,
        ...options,
        headers: {
          ...(options.method ? { "Content-Type": "application/json", "X-QQ-Control": "1" } : {}),
          ...(options.headers || {}),
        },
      });
      let data;
      try { data = await response.json(); } catch { throw new Error("控制服务返回了无法读取的响应，请稍后重试。"); }
      if (!response.ok) {
        const detail = data.detail || data.error;
        if (typeof detail === "string") throw new Error(detail);
        if (Array.isArray(detail)) {
          throw new Error(detail.map((item) => {
            const location = Array.isArray(item.loc) ? item.loc.filter((part) => part !== "body").join(".") : "设置";
            return `${location}：${item.msg || "参数无效"}`;
          }).join("；"));
        }
        throw new Error(`请求失败（${response.status}），请稍后重试。`);
      }
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("控制服务响应超时，请检查项目是否仍在运行。");
      if (error instanceof TypeError) throw new Error("无法连接本地控制服务，请确认项目仍在运行。");
      throw error;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function notify(message, error = false) {
    window.clearTimeout(noticeTimer);
    const notice = byId("notice");
    notice.textContent = message;
    notice.className = `notice${error ? " notice-error" : ""}`;
    notice.hidden = false;
    if (!error) noticeTimer = window.setTimeout(() => { notice.hidden = true; }, 6000);
  }

  function draft() {
    return {
      mode: modeInputs.find((input) => input.checked)?.value || "assistant",
      target_group_id: target.value.trim(),
      model: model.value.trim(),
      reply_limit_per_minute: Number(limit.value),
      chat_style: style.value.trim(),
      enabled: status?.settings.enabled ?? false,
    };
  }

  function updateMode() {
    byId("chat-style-field").hidden = !modeInputs.find((input) => input.value === "chat")?.checked;
  }

  function fill(settings) {
    modeInputs.forEach((input) => { input.checked = input.value === settings.mode; });
    target.value = settings.target_group_id || "";
    model.value = settings.model || "";
    style.value = settings.chat_style || "朋友群损友风格，善意互怼、吐槽、自嘲，适量粗口，简短不说教。";
    limit.value = settings.reply_limit_per_minute ?? 10;
    updateMode();
    dirty = false;
    updateButtons();
  }

  function updateButtons() {
    fields.disabled = busy || !status || !apiAvailable;
    buttonIds.forEach((id) => { byId(id).disabled = busy || !status || !apiAvailable; });
    byId("refresh-models").disabled ||= modelsLoading;
    byId("save-button").disabled ||= !dirty;
    byId("dirty-label").textContent = !status ? "加载中" : dirty ? "未保存" : "已同步";
    byId("dirty-label").classList.toggle("dirty", dirty);
    byId("save-hint").textContent = dirty ? "有未保存的更改，保存后立即生效" : "设置已生效，更改后可保存";
    if (busy) byId("save-hint").textContent = "正在应用，请稍候…";
    byId("power-label").textContent = !status ? "正在连接…" : status.settings.enabled ? "暂停机器人" : "启用机器人";
    byId("power-button").classList.toggle("is-running", Boolean(status?.settings.enabled));
    const count = Number(limit.value);
    byId("limit-minus").disabled = busy || !apiAvailable || count <= 1;
    byId("limit-plus").disabled = busy || !apiAvailable || count >= 60;
  }

  function markDirty() {
    if (!status) return;
    const current = draft();
    dirty = Object.keys(current).some((key) => key !== "enabled" && current[key] !== status.settings[key]);
    updateMode();
    updateButtons();
  }

  const activityLabels = {
    message: "群内消息", received: "群内消息", incoming: "群内消息", reply: "AI 回复", sent: "AI 回复",
    error: "运行提示", warning: "运行提示", settings: "设置已更新", config: "设置已更新",
    system: "运行动态", status: "运行动态", info: "运行动态", skip: "保持安静", skipped: "保持安静",
    decision: "聊天判断", silent: "保持安静", limited: "发言上限", rate_limit: "发言上限", clear: "记忆已清空", paused: "机器人已暂停",
  };

  function renderActivity(items) {
    const signature = JSON.stringify(items);
    if (signature === activitySignature) return;
    activitySignature = signature;
    const list = byId("activity-list");
    if (!items.length) {
      // The initial empty state is static markup; preserve it until there is activity.
      if (!list.querySelector(".activity-item")) return;
      list.replaceChildren();
      const empty = document.createElement("div");
      empty.className = "empty-state";
      const title = document.createElement("strong");
      title.textContent = "还没有新的群聊动态";
      const hint = document.createElement("p");
      hint.textContent = "目标群的消息和运行动态会显示在这里。";
      empty.append(title, hint);
      list.append(empty);
      return;
    }
    const fragment = document.createDocumentFragment();
    items.map((item, index) => ({ ...item, index })).sort((a, b) => {
      const difference = Date.parse(b.time) - Date.parse(a.time);
      return Number.isFinite(difference) && difference !== 0 ? difference : b.index - a.index;
    }).slice(0, 80).forEach((item) => {
      const row = document.createElement("article");
      row.className = "activity-item";
      row.classList.toggle("is-reply", ["reply", "sent"].includes(item.kind));
      row.classList.toggle("is-error", ["error", "warning"].includes(item.kind));
      const heading = document.createElement("div");
      heading.className = "activity-item-header";
      const kind = document.createElement("span");
      kind.className = "activity-kind";
      const dot = document.createElement("span");
      dot.className = "dot";
      kind.append(dot, document.createTextNode(activityLabels[item.kind] || "运行动态"));
      const time = document.createElement("time");
      const date = new Date(item.time);
      if (Number.isFinite(date.getTime())) {
        time.dateTime = date.toISOString();
        time.textContent = date.toLocaleTimeString("zh-CN", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
        time.title = date.toLocaleString("zh-CN", { hour12: false });
      } else {
        time.textContent = typeof item.time === "string" ? item.time : "";
      }
      const text = document.createElement("p");
      text.className = "activity-text";
      text.textContent = typeof item.text === "string" ? item.text : "";
      heading.append(kind, time);
      row.append(heading, text);
      fragment.append(row);
    });
    list.replaceChildren(fragment);
  }

  function render(next, forceFill = false) {
    const initial = !status;
    status = next;
    apiAvailable = true;
    if (initial || forceFill || !dirty) fill(next.settings);
    if (initial) byId("notice").hidden = true;
    byId("connection-state").textContent = next.connected ? "已连接" : "未连接";
    byId("connection-dot").className = `status-dot ${next.connected ? "good" : "bad"}`;
    byId("model-state").textContent = next.model_configured ? "已配置" : "未配置";
    byId("model-dot").className = `status-dot ${next.model_configured ? "good" : "bad"}`;
    byId("reply-count").textContent = next.replies_last_minute ?? 0;
    byId("reply-budget").textContent = next.settings.reply_limit_per_minute;
    byId("context-count").textContent = next.context_messages ?? 0;
    const enabled = next.settings.enabled;
    byId("active-mode").textContent = `${enabled ? "" : "已暂停 · "}${next.settings.mode === "chat" ? "聊天模式" : "群助手模式"}`;
    byId("active-group").textContent = next.settings.target_group_id ? `群 ${next.settings.target_group_id}` : "尚未设置目标群";
    const live = byId("activity-live");
    live.classList.toggle("live", Boolean(next.connected && enabled));
    live.lastChild.textContent = enabled ? next.connected ? "运行中" : "等待连接" : "已暂停";
    byId("sync-label").textContent = `自动同步 · ${new Date().toLocaleTimeString("zh-CN", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;
    const error = byId("server-error");
    error.hidden = !next.last_error;
    error.textContent = next.last_error ? `运行提示：${next.last_error}` : "";
    renderActivity(Array.isArray(next.activity) ? next.activity : []);
    updateButtons();
  }

  async function poll() {
    if (polling || busy) return;
    polling = true;
    const version = mutationVersion;
    try {
      const next = await request("/api/status");
      if (!busy && version === mutationVersion) render(next);
    } catch (error) {
      if (busy || version !== mutationVersion) return;
      apiAvailable = false;
      byId("sync-label").textContent = "控制服务离线，正在尝试重新连接";
      byId("activity-live").classList.remove("live");
      byId("activity-live").lastChild.textContent = "已断开";
      byId("connection-state").textContent = "状态未知";
      byId("connection-dot").className = "status-dot bad";
      updateButtons();
      if (!status) notify(error.message, true);
    } finally {
      polling = false;
    }
  }

  async function mutate(path, method, body, success, forceFill = false) {
    if (busy || !status) return;
    mutationVersion += 1;
    busy = true;
    updateButtons();
    try {
      const next = await request(path, { method, body: JSON.stringify(body) });
      render(next, forceFill);
      notify(success);
    } catch (error) {
      notify(error.message, true);
    } finally {
      busy = false;
      updateButtons();
    }
  }

  async function loadModels() {
    if (modelsLoading) return;
    modelsLoading = true;
    byId("refresh-models").disabled = true;
    byId("refresh-models").textContent = "读取中…";
    const hint = byId("model-hint");
    try {
      const result = await request("/api/models");
      const options = byId("model-options");
      options.replaceChildren();
      const models = Array.isArray(result.models) ? result.models.filter((name) => typeof name === "string") : [];
      models.forEach((name) => {
        const option = document.createElement("option");
        option.value = name;
        options.append(option);
      });
      hint.textContent = result.error ? `${result.error}；可以手动填写模型名称。` : models.length ? `已读取 ${models.length} 个可用模型，也可以手动填写。` : "暂无模型列表，可以手动填写模型名称。";
      hint.classList.toggle("error", Boolean(result.error));
    } catch (error) {
      hint.textContent = `${error.message} 也可以手动填写模型。`;
      hint.classList.add("error");
    } finally {
      modelsLoading = false;
      byId("refresh-models").textContent = "刷新列表";
      updateButtons();
    }
  }

  form.addEventListener("input", markDirty);
  form.addEventListener("change", markDirty);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!form.reportValidity()) return;
    const settings = draft();
    if (!settings.model || !settings.chat_style || (settings.enabled && !settings.target_group_id) || (settings.target_group_id && !/^[1-9][0-9]*$/.test(settings.target_group_id))) {
      notify("请填写有效的目标群号、模型名称和聊天风格。", true);
      return;
    }
    if (!Number.isInteger(settings.reply_limit_per_minute) || settings.reply_limit_per_minute < 1 || settings.reply_limit_per_minute > 60) {
      notify("每分钟回复上限必须是 1 到 60 之间的整数。", true);
      return;
    }
    mutate("/api/settings", "PUT", settings, "设置已保存并生效。", true);
  });
  byId("power-button").addEventListener("click", () => {
    if (!status) return;
    const enabled = !status.settings.enabled;
    const message = enabled ? "机器人已启用，开始监听目标群。" : dirty ? "机器人已暂停发言。未保存的设置仍保留在表单中。" : "机器人已暂停发言。";
    mutate("/api/settings", "PUT", { ...status.settings, enabled }, message);
  });
  byId("clear-button").addEventListener("click", () => {
    mutate("/api/context/clear", "POST", {}, "短期聊天记忆已清空，群友画像和群内梗仍然保留。");
  });
  byId("refresh-models").addEventListener("click", loadModels);
  [["limit-minus", -1], ["limit-plus", 1]].forEach(([id, change]) => {
    byId(id).addEventListener("click", () => {
      const current = Number(limit.value) || 10;
      limit.value = Math.min(60, Math.max(1, current + change));
      markDirty();
    });
  });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });
  poll().then(loadModels);
  window.setInterval(() => { if (!document.hidden) poll(); }, 3000);
})();
