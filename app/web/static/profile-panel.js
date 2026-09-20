"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const refresh = byId("profiles-refresh");
  const search = byId("profiles-search");
  const aliasForm = byId("profile-alias-form");
  const aliasMember = byId("profile-alias-member");
  const aliasName = byId("profile-alias-name");
  const opened = new Set();
  let snapshot = null;
  let polling = false;
  let busy = false;
  let available = false;
  let version = 0;
  let visibleCount = 30;
  let contentSignature = "";
  let memberOptionsSignature = "";

  const categories = {
    style: "说话方式", speaking_style: "说话方式", speech_style: "说话方式",
    tone: "语气", humor: "幽默习惯", humor_style: "玩梗风格",
    catchphrase: "口头禅", catchphrases: "口头禅", expression: "常用表达", expressions: "常用表达",
    emoji: "表情习惯", emojis: "表情习惯", emoji_habits: "表情习惯",
    topics: "常聊话题", topic: "常聊话题", interests: "常聊话题", topic_preferences: "话题偏好",
    nickname: "群内称呼", nicknames: "群内称呼", aliases: "群内称呼",
    interaction: "互动习惯", interactions: "互动习惯", relationships: "群内互动",
    joke: "玩梗记录", jokes: "玩梗记录", memes: "玩梗记录", callbacks: "常见回梗",
    behavior: "聊天习惯", habits: "聊天习惯", persona: "群内风格", preferences: "聊天偏好",
    chat_traits: "聊天特点", interaction_style: "互动习惯", boundaries: "玩笑边界",
  };
  const stageLabels = {
    idle: "等待更新", starting: "正在准备更新", history: "正在读取近期群历史",
    backfill: "正在补充近期群历史", fetching: "正在读取近期群历史",
    roster: "正在读取群友名单", analyzing: "正在整理聊天画像与群内梗",
    analysis: "正在整理聊天画像与群内梗", saving: "正在保存画像",
    complete: "画像已更新", completed: "画像已更新", done: "画像已更新",
    error: "本次更新未完成", failed: "本次更新未完成",
  };

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  async function request(path, options = {}) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 25000);
    try {
      const response = await fetch(path, {
        credentials: "same-origin", cache: "no-store", signal: controller.signal,
        ...options,
        headers: options.method ? { "Content-Type": "application/json", "X-QQ-Control": "1" } : {},
      });
      let data;
      try { data = await response.json(); } catch { throw new Error("画像服务返回了无法读取的响应。"); }
      if (!response.ok) {
        const message = typeof data.detail === "string" ? data.detail
          : typeof data.error === "string" ? data.error
          : Array.isArray(data.detail) ? "称呼或群友选择无效，请检查后重试。"
          : `画像请求未完成（${response.status}），请稍后重试。`;
        throw Object.assign(new Error(message), { status: response.status });
      }
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("画像服务响应超时，请稍后重试。");
      if (error instanceof TypeError) throw new Error("画像服务暂时无法连接，恢复后会自动同步。");
      throw error;
    } finally {
      window.clearTimeout(timer);
    }
  }

  function notice(text, isError = false) {
    const node = byId("profiles-notice");
    node.textContent = text;
    node.className = `notice${isError ? " notice-error" : ""}`;
    node.hidden = !text;
  }

  function buttons() {
    const running = Boolean(snapshot?.job?.running);
    refresh.disabled = !available || busy || running || !snapshot?.group_id;
    refresh.querySelector("span").textContent = running ? "正在更新画像…" : "读取历史并更新画像";
    aliasMember.disabled = !available || busy || !snapshot?.members?.length;
    byId("profile-alias-save").disabled = !available || busy || !snapshot?.members?.length;
  }

  function badge(text, kind = "") {
    return el("span", `profile-badge${kind ? ` ${kind}` : ""}`, text);
  }

  function evidenceDetails(ids, evidence) {
    const sourceIds = Array.isArray(ids) ? [...new Set(ids.map(String))] : [];
    if (!sourceIds.length) return null;
    const details = el("details", "profile-evidence");
    details.append(el("summary", "", `${sourceIds.length} 条聊天依据`));
    const list = el("div", "profile-evidence-list");
    sourceIds.forEach((id) => {
      const source = evidence.get(id);
      const quote = el("blockquote", "profile-source");
      if (source) {
        const sourceDate = source.time ? new Date(typeof source.time === "number" ? source.time * 1000 : source.time) : null;
        const sourceTime = sourceDate && Number.isFinite(sourceDate.getTime()) ? ` · ${sourceDate.toLocaleString("zh-CN", { hour12: false })}` : "";
        quote.append(el("div", "profile-source-heading", `${source.nickname || source.user_id || "群友"}${sourceTime}`));
        quote.append(el("p", "", source.text || "（非文字消息）"));
        quote.append(el("small", "", `消息 ${id}`));
      } else {
        quote.append(el("p", "", `消息 ${id}`));
        quote.append(el("small", "", "这条依据的原文暂未加载。"));
      }
      list.append(quote);
    });
    details.append(list);
    return details;
  }

  function memberName(id, names) {
    return names.get(String(id)) || `QQ ${id}`;
  }

  function memberCard(member, evidence) {
    const id = String(member.user_id);
    const profile = member.profile;
    const name = member.display_name || member.names?.[0] || `QQ ${id}`;
    const card = el("details", "profile-member-card");
    card.open = opened.has(id);
    card.addEventListener("toggle", () => { if (card.open) opened.add(id); else opened.delete(id); });
    const heading = el("summary", "profile-member-heading");
    const avatar = el("span", "profile-avatar", [...name].slice(0, 1).join("") || "群");
    avatar.setAttribute("aria-hidden", "true");
    const headingText = el("span", "profile-member-title");
    headingText.append(el("strong", "", name), el("span", "", `QQ ${id} · ${member.message_count || 0} 条消息`));
    heading.append(avatar, headingText, badge(profile ? "已整理" : "待积累", profile ? "observed" : ""));
    card.append(heading);
    const body = el("div", "profile-member-body");
    if (profile?.summary) {
      const summary = el("p", "profile-summary");
      summary.append(el("span", "", profile.summary));
      body.append(summary);
    }
    else body.append(el("p", "profile-no-analysis", "还没有足够的聊天画像。继续积累消息，或点击“读取历史并更新画像”。"));
    const observedNames = [...new Set((Array.isArray(member.names) ? member.names : []).filter(Boolean))];
    if (observedNames.length) {
      const names = el("div", "profile-tags");
      names.append(el("span", "profile-tags-label", "出现过的名字"));
      observedNames.forEach((item) => names.append(badge(item)));
      body.append(names);
    }
    if (Array.isArray(member.emoji_habits) && member.emoji_habits.length) {
      const emojis = el("div", "profile-tags");
      emojis.append(el("span", "profile-tags-label", "常用表情"));
      member.emoji_habits.forEach((item) => emojis.append(badge(`${item.emoji} ×${item.count}`)));
      body.append(emojis);
    }
    const facts = Array.isArray(profile?.facts) ? profile.facts : [];
    facts.forEach((fact) => {
      const row = el("div", "profile-fact");
      const title = el("div", "profile-fact-title");
      title.append(el("span", "", categories[fact.category] || fact.category || "聊天观察"));
      title.append(badge(fact.confidence === "observed" ? "有记录" : "待核实", fact.confidence === "observed" ? "observed" : "likely"));
      row.append(title, el("p", "", fact.text));
      const sources = evidenceDetails(fact.evidence_ids, evidence);
      if (sources) row.append(sources);
      body.append(row);
    });
    card.append(body);
    return card;
  }

  function renderMembers() {
    if (!snapshot) return;
    const query = search.value.trim().toLocaleLowerCase();
    const members = Array.isArray(snapshot.members) ? snapshot.members : [];
    const aliasesByMember = new Map();
    (snapshot.aliases || []).forEach((alias) => {
      (alias.user_ids || []).forEach((id) => {
        const key = String(id);
        aliasesByMember.set(key, `${aliasesByMember.get(key) || ""} ${alias.alias}`);
      });
    });
    const filtered = members.filter((member) => {
      if (!query) return true;
      const text = [member.user_id, member.display_name, ...(member.names || []), member.profile?.summary,
        ...(member.profile?.facts || []).map((fact) => fact.text), aliasesByMember.get(String(member.user_id))].join(" ");
      return text.toLocaleLowerCase().includes(query);
    });
    const evidence = new Map((snapshot.evidence || []).map((item) => [String(item.message_id), item]));
    const list = byId("profile-member-list");
    const fragment = document.createDocumentFragment();
    filtered.slice(0, visibleCount).forEach((member) => fragment.append(memberCard(member, evidence)));
    if (!filtered.length) {
      const empty = el("div", "profile-empty");
      empty.append(el("strong", "", query ? "没有找到匹配的群友" : "先认识一下群里的朋友"));
      empty.append(el("p", "", query ? "试试群昵称、QQ 号、绰号或某句口头禅。" : "读取历史并更新画像后，这里会出现群友的聊天习惯与群内称呼。"));
      fragment.append(empty);
    }
    list.replaceChildren(fragment);
    byId("profiles-result-count").textContent = query ? `找到 ${filtered.length} 位群友` : `${members.length} 位群友 · 展开卡片查看聊天画像与依据`;
    const more = byId("profiles-more");
    more.hidden = filtered.length <= visibleCount;
    more.textContent = `再显示 ${Math.min(30, filtered.length - visibleCount)} 位群友`;
  }

  function renderContext() {
    const names = new Map((snapshot.members || []).map((member) => [String(member.user_id), member.display_name || `QQ ${member.user_id}`]));
    const evidence = new Map((snapshot.evidence || []).map((item) => [String(item.message_id), item]));
    const aliases = Array.isArray(snapshot.aliases) ? snapshot.aliases : [];
    const aliasList = byId("profile-alias-list");
    aliasList.replaceChildren();
    byId("aliases-count").textContent = aliases.length;
    aliases.forEach((alias) => {
      const item = el("article", "profile-context-item");
      const heading = el("div", "profile-context-heading");
      heading.append(el("strong", "", alias.alias), badge({ confirmed: "已确认", likely: "待核实", ambiguous: "有歧义" }[alias.status] || "待核实", alias.status === "confirmed" ? "observed" : "likely"));
      const people = el("p", "alias-people", (alias.user_ids || []).map((id) => memberName(id, names)).join(" / ") || "暂不确定指代谁");
      item.append(heading, people);
      if (alias.explanation) item.append(el("p", "profile-description", alias.explanation));
      const sources = evidenceDetails(alias.evidence_ids, evidence);
      if (sources) item.append(sources);
      const edit = el("button", "text-button alias-edit-button", "纠正");
      edit.type = "button";
      edit.addEventListener("click", () => {
        aliasName.value = alias.alias;
        aliasMember.value = alias.user_ids?.length === 1 ? String(alias.user_ids[0]) : "";
        document.querySelector(".alias-editor").open = true;
        aliasMember.focus();
      });
      item.append(edit);
      aliasList.append(item);
    });
    if (!aliases.length) aliasList.append(el("p", "profile-empty-line", "还没有整理出的群内称呼。可以手动补充，也可以等画像分析识别。"));
    const memes = Array.isArray(snapshot.memes) ? snapshot.memes : [];
    byId("memes-count").textContent = memes.length;
    const memeList = byId("profile-meme-list");
    memeList.replaceChildren();
    memes.forEach((meme) => {
      const item = el("article", "profile-context-item");
      item.append(el("strong", "meme-name", meme.name), el("p", "meme-meaning", meme.meaning));
      if (meme.user_ids?.length) item.append(el("p", "profile-description", `相关群友：${meme.user_ids.map((id) => memberName(id, names)).join("、")}`));
      const sources = evidenceDetails(meme.evidence_ids, evidence);
      if (sources) item.append(sources);
      memeList.append(item);
    });
    if (!memes.length) memeList.append(el("p", "profile-empty-line", "新的群内梗正在积累中。更新画像后，可查看梗的含义、相关群友与原始聊天依据。"));
  }

  function renderOptions() {
    const signature = JSON.stringify([snapshot.group_id, (snapshot.members || []).map((member) => [member.user_id, member.display_name])]);
    if (signature === memberOptionsSignature) return;
    memberOptionsSignature = signature;
    const current = aliasMember.value;
    const placeholder = el("option", "", snapshot.members?.length ? "选择一位群友" : "先读取群友名单");
    placeholder.value = "";
    const options = document.createDocumentFragment();
    options.append(placeholder);
    (snapshot.members || []).forEach((member) => {
      const option = el("option", "", `${member.display_name || "群友"} · ${member.user_id}`);
      option.value = String(member.user_id);
      options.append(option);
    });
    aliasMember.replaceChildren(options);
    aliasMember.value = [...aliasMember.options].some((option) => option.value === current) ? current : "";
  }

  function render(next) {
    const recovering = !available;
    const previousJobError = snapshot?.job?.error;
    const groupChanged = snapshot && String(next.group_id) !== String(snapshot.group_id);
    if (groupChanged) {
      opened.clear();
      search.value = "";
      aliasName.value = "";
      aliasMember.value = "";
      visibleCount = 30;
      notice("");
    }
    snapshot = next;
    available = true;
    if (recovering || (previousJobError && !next.job?.error)) notice("");
    byId("profiles-group").textContent = next.group_id ? `仅限目标群 ${next.group_id}` : "尚未设置目标群";
    const stats = next.stats || {};
    byId("profiles-message-count").textContent = Number(stats.messages || 0).toLocaleString("zh-CN");
    byId("profiles-member-count").textContent = Number(stats.members || 0).toLocaleString("zh-CN");
    byId("profiles-ready-count").textContent = Number(stats.profiled_members || 0).toLocaleString("zh-CN");
    const job = next.job || {};
    const cleaning = job.cleaning;
    byId("profiles-cleaning").hidden = !cleaning;
    if (cleaning) byId("profiles-cleaning").textContent = `本次清洗：${cleaning.original_count} 条原记录 → ${cleaning.ai_kept_count} 条分析素材，合并 ${cleaning.duplicate_count} 条重复内容。${cleaning.ai_used ? "已由接入模型筛选。" : "使用规则清洗结果。"}${cleaning.ai_failed_batches ? " 部分筛选请求失败，已保留规则清洗结果。" : ""}`;
    const updated = job.updated_at ? new Date(job.updated_at) : null;
    const jobLabel = stageLabels[job.stage] || job.stage || "等待更新";
    byId("profiles-job").textContent = job.running ? jobLabel : job.error ? "更新未完成，可重试" : updated && Number.isFinite(updated.getTime()) ? `最近更新 ${updated.toLocaleString("zh-CN", { hour12: false })}` : jobLabel;
    byId("profiles-job").classList.toggle("is-updating", Boolean(job.running));
    if (job.error) notice(`画像更新提示：${job.error}`, true);
    const signature = JSON.stringify([next.group_id, next.members, next.aliases, next.memes, next.evidence]);
    if (signature !== contentSignature) {
      contentSignature = signature;
      renderMembers();
      renderContext();
      renderOptions();
    }
    buttons();
  }

  async function poll() {
    if (polling || busy) return;
    polling = true;
    const started = version;
    try {
      const next = await request("/api/profiles");
      if (!busy && started === version) render(next);
    } catch (error) {
      if (busy || started !== version) return;
      available = false;
      byId("profiles-job").textContent = "暂时离线，正在尝试重新连接";
      byId("profiles-job").classList.remove("is-updating");
      if (!snapshot) notice(error.message, true);
      buttons();
    } finally {
      polling = false;
    }
  }

  async function mutate(path, method, body, message) {
    if (busy || !available) return false;
    busy = true;
    version += 1;
    buttons();
    try {
      const next = await request(path, { method, body: JSON.stringify(body) });
      render(next);
      notice(message);
      return true;
    } catch (error) {
      if (error.status === 400) {
        try { render(await request("/api/profiles")); } catch { /* Keep the useful mutation error. */ }
      }
      notice(error.message, true);
      return false;
    } finally {
      busy = false;
      buttons();
    }
  }

  refresh.addEventListener("click", () => {
    mutate("/api/profiles/refresh", "POST", { group_id: String(snapshot.group_id) }, "正在读取目标群近期历史并整理画像。完成后会自动更新，也会用于后续接话。");
  });
  search.addEventListener("input", () => { visibleCount = 30; renderMembers(); });
  byId("profiles-more").addEventListener("click", () => { visibleCount += 30; renderMembers(); });
  aliasForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!aliasForm.reportValidity()) return;
    const alias = aliasName.value.trim();
    if (!alias || !aliasMember.value) { notice("请填写称呼，并选择实际指代的群友。", true); return; }
    const saved = await mutate("/api/profiles/alias", "PUT", { group_id: String(snapshot.group_id), alias, user_ids: [aliasMember.value] }, "称呼已确认，对应关系会用于后续聊天。");
    if (saved) aliasName.value = "";
  });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });
  poll();
  window.setInterval(() => { if (!document.hidden) poll(); }, 6000);
})();
