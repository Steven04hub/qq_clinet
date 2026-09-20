"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const notice = (message, error = false) => {
    const node = byId("memory-tools-notice");
    node.textContent = message;
    node.hidden = !message;
    node.className = `notice${error ? " notice-error" : ""}`;
  };
  function el(tag, className, text) {
    const node = document.createElement(tag);
    node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  async function request(path, body) {
    const response = await fetch(path, body === undefined ? {cache:"no-store"} : {
      method:"POST", headers:{"Content-Type":"application/json", "X-QQ-Control":"1"},
      body:JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "操作未完成，请稍后重试。");
    return data;
  }
  let searchVersion = 0;
  let displayedGroup = "";
  byId("memory-search-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const version = ++searchVersion;
    const button = event.currentTarget.querySelector("button");
    button.disabled = true;
    try {
      const status = await request("/api/status");
      const group = status.settings.target_group_id;
      const data = await request("/api/memory/search", {group_id:group, query:byId("memory-query").value.trim()});
      const current = await request("/api/status");
      if (version !== searchVersion || current.settings.target_group_id !== data.group_id) return;
      displayedGroup = data.group_id;
      const list = byId("memory-results");
      list.replaceChildren();
      data.results.forEach((row) => {
        const card = el("article", "memory-result");
        const when = new Date(row.time * 1000).toLocaleString("zh-CN", {hour12:false});
        card.append(el("strong", "", row.nickname || row.user_id), el("small", "", when), el("p", "", row.text));
        list.append(card);
      });
      if (!data.results.length) list.append(el("p", "profile-empty-line", "暂时没找到相关记录，试试另一个称呼或关键词。"));
      notice("");
    } catch (error) { notice(error.message, true); }
    finally { button.disabled = false; }
  });
  // A settings switch must not leave old-group search results on screen.
  window.setInterval(async () => {
    if (!displayedGroup || document.hidden) return;
    try {
      const status = await request("/api/status");
      if (displayedGroup !== status.settings.target_group_id) {
        ++searchVersion;
        displayedGroup = "";
        byId("memory-results").replaceChildren(el("p", "profile-empty-line", "目标群已切换，请重新搜索。"));
      }
    } catch { /* Existing connection indicator covers temporary network failure. */ }
  }, 4000);

  function renderStickers(data) {
    const list = byId("sticker-list");
    list.replaceChildren();
    data.stickers.forEach((sticker) => {
      const card = el("div", "sticker-item");
      if (sticker.kind === "image") {
        const img = el("img", "sticker-preview");
        img.src = `/api/stickers/${encodeURIComponent(sticker.id)}/image`;
        img.alt = sticker.label;
        img.loading = "lazy";
        card.append(img);
      } else card.append(el("span", "sticker-face", sticker.emoji || "🙂"));
      card.append(el("span", "", sticker.label));
      card.title = (sticker.tags || []).join("、");
      list.append(card);
    });
  }
  request("/api/stickers").then(renderStickers).catch((error) => notice(error.message, true));
  byId("sticker-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = byId("sticker-file").files[0];
    if (!file || file.size > 8 * 1024 * 1024) { notice("请选择不超过 8 MB 的图片。", true); return; }
    const button = event.currentTarget.querySelector("button");
    button.disabled = true;
    try {
      const encoded = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result).split(",", 2)[1]);
        reader.onerror = () => reject(new Error("图片读取失败。"));
        reader.readAsDataURL(file);
      });
      const data = await request("/api/stickers", {
        filename:file.name, label:byId("sticker-label").value.trim(),
        tags:byId("sticker-tags").value.split(/[,，]/).map((s) => s.trim()).filter(Boolean),
        data_base64:encoded,
      });
      renderStickers(data);
      byId("sticker-form").reset();
      notice("已加入素材库，机器人接话时可以选用。");
    } catch (error) { notice(error.message, true); }
    finally { button.disabled = false; }
  });
  byId("memory-export").addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    try {
      const response = await fetch("/api/memory/export", {method:"POST", headers:{"X-QQ-Control":"1"}});
      if (!response.ok) throw new Error((await response.json()).error || "迁移包导出失败。");
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = "qq-memory.zip";
      document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 30000);
      notice("迁移包已下载。新电脑按 README 的导入步骤操作，再配置当地的 QQ/NapCat 与模型密钥。");
    } catch (error) { notice(error.message, true); }
    finally { button.disabled = false; }
  });
})();
