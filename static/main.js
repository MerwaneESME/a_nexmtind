const messagesEl = document.getElementById("messages");
const quickRepliesEl = document.getElementById("quick-replies");
const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const attachBtn = document.getElementById("attach-btn");
const fileInput = document.getElementById("file-input");

let pendingFile = null;
let loading = false;
let loadingBubble = null;
let history = [];

const escapeHtml = (str = "") =>
  str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

const renderMarkdownLite = (text = "") => {
  const escaped = escapeHtml(text);
  return escaped
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
};

const addMessage = (role, content, meta = "", cls = "") => {
  const wrap = document.createElement("div");
  wrap.className = `bubble ${role} ${cls}`.trim();
  wrap.innerHTML = content;
  history.push({ role, content });
  if (meta) {
    const m = document.createElement("div");
    m.className = "meta";
    m.textContent = meta;
    wrap.appendChild(m);
  }
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  const placeholder = messagesEl.querySelector(".placeholder");
  if (placeholder) placeholder.remove();
  return wrap;
};

const removeLoading = () => {
  if (loadingBubble && loadingBubble.parentNode) {
    loadingBubble.remove();
    loadingBubble = null;
  }
};

const clearQuickReplies = () => {
  if (quickRepliesEl) quickRepliesEl.innerHTML = "";
};

const sendChatSafe = (text) => {
  const t = (text || "").trim();
  if (!t || loading) return;
  sendChat(t);
};

const addQuickReplies = (options = []) => {
  clearQuickReplies();
  if (!options.length || !quickRepliesEl) return;
  options.slice(0, 8).forEach((opt) => {
    const btn = document.createElement("button");
    btn.className = "quick-reply";
    btn.textContent = opt;
    btn.addEventListener("click", () => {
      sendChatSafe(opt);
    });
    quickRepliesEl.appendChild(btn);
  });
};

const parseOptions = (text = "") => {
  const matches = [...text.matchAll(/\[([^\]]+)\]/g)].map((m) => m[1]);
  const opts = [];
  matches.forEach((m) => {
    m.split("/").forEach((part) => {
      const clean = part.trim();
      if (clean && !opts.includes(clean)) opts.push(clean);
    });
  });
  return opts;
};

const formatJson = (data) => {
  try {
    return JSON.stringify(data, null, 2);
  } catch {
    return String(data);
  }
};

const renderResult = (payload) => {
  const { data, totals, inconsistencies, supabase, error, details, reply } = payload || {};
  removeLoading();
  clearQuickReplies();
  if (reply) {
    const bubble = addMessage("assistant", renderMarkdownLite(reply));
    const opts = parseOptions(reply);
    if (opts.length) {
      const inline = document.createElement("div");
      inline.className = "quick-inline";
      opts.slice(0, 3).forEach((opt) => {
        const btn = document.createElement("button");
        btn.className = "quick-reply";
        btn.textContent = opt;
        btn.addEventListener("click", () => {
          sendChatSafe(opt);
        });
        inline.appendChild(btn);
      });
      // Champs libres si l'agent demande des infos ouvertes
      const lower = reply.toLowerCase();
      const addInlineField = (placeholder) => {
        const group = document.createElement("div");
        group.className = "quick-inline";
        const inp = document.createElement("input");
        inp.type = "text";
        inp.className = "inline-input";
        inp.placeholder = placeholder;
        const sendSmall = document.createElement("button");
        sendSmall.className = "inline-submit";
        sendSmall.textContent = "OK";
        sendSmall.addEventListener("click", () => {
          sendChatSafe(inp.value);
          inp.value = "";
        });
        group.appendChild(inp);
        group.appendChild(sendSmall);
        inline.appendChild(group);
      };
      if (lower.includes("adresse") || lower.includes("chantier")) {
        addInlineField("Adresse du chantier");
      }
      if (lower.includes("tva") || lower.includes("taxe")) {
        addInlineField("Taux de TVA (ex: 20%)");
      }
      if (lower.includes("budget") || lower.includes("montant") || lower.includes("surface")) {
        addInlineField("Montant / surface / budget");
      }
      bubble.appendChild(inline);
    }
    return;
  }
  let html = "";
  if (error) html += `<div class="block"><strong>Erreur :</strong> ${escapeHtml(error)} ${escapeHtml(details || "")}</div>`;
  if (totals) html += `<div class="block"><strong>Totaux :</strong><pre>${formatJson(totals)}</pre></div>`;
  if (inconsistencies?.length) html += `<div class="block"><strong>Incohérences :</strong><pre>${formatJson(inconsistencies)}</pre></div>`;
  if (data) html += `<div class="block"><strong>Données :</strong><pre>${formatJson(data)}</pre></div>`;
  if (supabase) html += `<div class="block"><strong>Supabase :</strong><pre>${formatJson(supabase)}</pre></div>`;
  addMessage("assistant", html || "OK");
};

const sendFile = async (file) => {
  const fd = new FormData();
  fd.append("file", file);
  addMessage("user", renderMarkdownLite(`Fichier : ${file.name}`));
  loadingBubble = addMessage("assistant", `<span class="spinner"></span> Analyse...`, "", "loading");
  try {
    const resp = await fetch(`/analyze`, { method: "POST", body: fd });
    const text = await resp.text();
    const json = (() => {
      try { return JSON.parse(text); } catch { return { error: "parse_error", details: text }; }
    })();
    renderResult(json);
  } catch (err) {
    renderResult({ error: "Erreur", details: err.message });
  }
};

const sendChat = async (text) => {
  addMessage("user", renderMarkdownLite(text));
  loadingBubble = addMessage("assistant", `<span class="spinner"></span>`, "", "loading");
  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, history }),
    });
    const json = await resp.json();
    renderResult(json);
  } catch (err) {
    renderResult({ error: "Erreur", details: err.message });
  }
};

const onSend = async () => {
  if (loading) return;
  const text = inputEl.value.trim();
  if (!text && !pendingFile) return;
  inputEl.value = "";
  loading = true;
  sendBtn.disabled = true;
  try {
    if (pendingFile) {
      await sendFile(pendingFile);
      pendingFile = null;
    } else {
      await sendChat(text);
    }
  } finally {
    loading = false;
    sendBtn.disabled = false;
  }
};

attachBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (f) {
    pendingFile = f;
    addMessage("assistant", renderMarkdownLite(`Fichier prêt : ${f.name}. Clique sur Envoyer pour lancer l'analyse.`));
  }
});
sendBtn.addEventListener("click", onSend);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    onSend();
  }
});
