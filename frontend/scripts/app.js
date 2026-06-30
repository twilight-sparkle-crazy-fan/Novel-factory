import { api, ApiError, request, stream } from "./api.js";
import { escapeText, renderMarkdown } from "./markdown.js";

const icons = {
  left: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m14 7-5 5 5 5"/></svg>',
  right: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m10 7 5 5-5 5"/></svg>',
  check: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 4 4L19 6"/></svg>',
  refresh: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 7v5h-5M4 17v-5h5"/><path d="M6.1 9A7 7 0 0 1 18.6 7M17.9 15A7 7 0 0 1 5.4 17"/></svg>',
  copy: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="11" height="11" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></svg>',
  trash: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 7h14M9 7V4h6v3M8 10v7M12 10v7M16 10v7M6.5 7l1 13h9l1-13"/></svg>',
};

const elements = {
  app: document.querySelector("#app"),
  sidebar: document.querySelector("#sidebar"),
  sidebarOverlay: document.querySelector("#sidebar-overlay"),
  conversationList: document.querySelector("#conversation-list"),
  conversationTitle: document.querySelector("#conversation-title"),
  messages: document.querySelector("#messages"),
  welcome: document.querySelector("#welcome"),
  chatScroll: document.querySelector("#chat-scroll"),
  composerForm: document.querySelector("#composer-form"),
  composerInput: document.querySelector("#composer-input"),
  sendButton: document.querySelector("#send-button"),
  previewNotice: document.querySelector("#preview-notice"),
  runtimeStatus: document.querySelector("#runtime-status"),
  runtimeStatusText: document.querySelector("#runtime-status-text"),
  settingsBackdrop: document.querySelector("#settings-backdrop"),
  settingsPanel: document.querySelector("#settings-panel"),
  settingsSaveState: document.querySelector("#settings-save-state"),
  toastRegion: document.querySelector("#toast-region"),
  temperature: document.querySelector("#temperature"),
  temperatureValue: document.querySelector("#temperature-value"),
  topP: document.querySelector("#top-p"),
  topPValue: document.querySelector("#top-p-value"),
  maxTokens: document.querySelector("#max-tokens"),
  minCompletionTokens: document.querySelector("#min-completion-tokens"),
  repeatPenalty: document.querySelector("#repeat-penalty"),
  randomSeed: document.querySelector("#random-seed"),
  seedField: document.querySelector("#seed-field"),
  seed: document.querySelector("#seed"),
  systemPrompt: document.querySelector("#system-prompt"),
  pinnedContext: document.querySelector("#pinned-context"),
  styleGuide: document.querySelector("#style-guide"),
  styleLexicon: document.querySelector("#style-lexicon"),
  presetLabel: document.querySelector("#preset-label"),
  contextUsage: document.querySelector("#context-usage"),
  contextUsageText: document.querySelector("#context-usage-text"),
  contextUsageBar: document.querySelector("#context-usage-bar"),
  projectBackdrop: document.querySelector("#project-backdrop"),
  projectPanel: document.querySelector("#project-panel"),
  projectCounts: document.querySelector("#project-counts"),
  documentList: document.querySelector("#document-list"),
  documentSelect: document.querySelector("#document-select"),
  libraryEnabled: document.querySelector("#library-enabled"),
  txtFile: document.querySelector("#txt-file"),
  importTxt: document.querySelector("#import-txt"),
  materialPackageFile: document.querySelector("#material-package-file"),
  materialPackageMode: document.querySelector("#material-package-mode"),
  materialScopeStart: document.querySelector("#material-scope-start"),
  materialScopeEnd: document.querySelector("#material-scope-end"),
  exportMaterialPackage: document.querySelector("#export-material-package"),
  importMaterialPackage: document.querySelector("#import-material-package"),
  rebuildMaterialSystem: document.querySelector("#rebuild-material-system"),
  previewMaterialPlan: document.querySelector("#preview-material-plan"),
  editMaterialBudget: document.querySelector("#edit-material-budget"),
  refreshMaterialReviews: document.querySelector("#refresh-material-reviews"),
  inspectMaterialSystem: document.querySelector("#inspect-material-system"),
  materialPackageReport: document.querySelector("#material-package-report"),
  materialBudgetEditor: document.querySelector("#material-budget-editor"),
  materialInspector: document.querySelector("#material-inspector"),
  materialReviewList: document.querySelector("#material-review-list"),
  summaryEnabled: document.querySelector("#summary-enabled"),
  globalSummary: document.querySelector("#global-summary"),
  summarizeProject: document.querySelector("#summarize-project"),
  analysisProgress: document.querySelector("#analysis-progress"),
  analysisProgressText: document.querySelector("#analysis-progress-text"),
  analysisProgressCount: document.querySelector("#analysis-progress-count"),
  analysisProgressBar: document.querySelector("#analysis-progress-bar"),
  chapterList: document.querySelector("#chapter-list"),
  characterList: document.querySelector("#character-list"),
  analysisTokenNote: document.querySelector("#analysis-token-note"),
  recentChaptersEnabled: document.querySelector("#recent-chapters-enabled"),
  charactersEnabled: document.querySelector("#characters-enabled"),
  factsEnabled: document.querySelector("#facts-enabled"),
  factList: document.querySelector("#fact-list"),
  previewPrompt: document.querySelector("#preview-prompt"),
  promptPreviewBox: document.querySelector("#prompt-preview-box"),
  promptPreviewContent: document.querySelector("#prompt-preview-content"),
  analysisStart: document.querySelector("#analysis-start"),
  analysisEnd: document.querySelector("#analysis-end"),
  resumeAnalysis: document.querySelector("#resume-analysis"),
  outlineBackdrop: document.querySelector("#outline-backdrop"),
  outlinePanel: document.querySelector("#outline-panel"),
  outlineInstruction: document.querySelector("#outline-instruction"),
  outlineContent: document.querySelector("#outline-content"),
  outlineCounter: document.querySelector("#outline-counter"),
  outlineState: document.querySelector("#outline-state"),
  outlinePrev: document.querySelector("#outline-prev"),
  outlineNext: document.querySelector("#outline-next"),
  newOutline: document.querySelector("#new-outline"),
  rerollOutline: document.querySelector("#reroll-outline"),
  saveOutline: document.querySelector("#save-outline"),
  selectOutline: document.querySelector("#select-outline"),
  outlineEnabled: document.querySelector("#outline-enabled"),
  outlineTokenNote: document.querySelector("#outline-token-note"),
  deleteOutlineCandidate: document.querySelector("#delete-outline-candidate"),
  clearOutline: document.querySelector("#clear-outline"),
  incrementBackdrop: document.querySelector("#increment-backdrop"),
  incrementDialog: document.querySelector("#increment-dialog"),
  incrementTarget: document.querySelector("#increment-target"),
  incrementTitleField: document.querySelector("#increment-title-field"),
  incrementChapterTitle: document.querySelector("#increment-chapter-title"),
  incrementContent: document.querySelector("#increment-content"),
  incrementStatus: document.querySelector("#increment-status"),
  confirmIncrement: document.querySelector("#confirm-increment"),
  incrementSummarizeNow: document.querySelector("#increment-summarize-now"),
};

const state = {
  conversations: [],
  conversation: null,
  viewedCandidates: new Map(),
  generating: false,
  activeCandidateId: null,
  streamController: null,
  runtime: null,
  shouldFollowStream: true,
  project: null,
  workspace: null,
  materialReviewItems: [],
  materialReviewsLoaded: false,
  materialOverview: null,
  materialInspectorLoaded: false,
  materialBudgetProfile: null,
  materialBudgetLoaded: false,
  analysisRunning: false,
  outline: null,
  outlineDrafts: [],
  previousOutlineId: null,
  outlineViewedCandidateId: null,
  outlineGenerating: false,
  outlineStreamController: null,
  contextStats: null,
  contextTimer: null,
  tabId: crypto.randomUUID(),
  tabChannel: null,
  tabClaims: new Set(),
  pendingConversationId: null,
  incrementCandidate: null,
  incrementRunning: false,
  appendedCandidateIds: new Set(),
};

const TAB_CONVERSATION_KEY = "llm4chat-tab-conversation";

const presets = {
  steady: { label: "稳健", temperature: 0.55, top_p: 0.85, repeat_penalty: 1.1 },
  creative: { label: "创意", temperature: 0.9, top_p: 0.95, repeat_penalty: 1.08 },
  wild: { label: "放飞", temperature: 1.2, top_p: 1, repeat_penalty: 1.04 },
};

const styleTemplates = {
  suspense: {
    label: "冷峻悬疑",
    guide: "叙述保持冷峻克制，减少解释性心理旁白，用动作、环境细节和短句推进紧张感。\n避免把危险、怀疑和秘密说透，保留读者可以推理的空白。",
    lexicon: "余温\n裂隙\n回声\n暗线\n旧案\n冷光",
  },
  classical: {
    label: "古风雅致",
    guide: "句式偏雅，保留古典意象和节制的抒情，不堆砌生僻词。\n人物对白要符合身份和礼法，叙事节奏稳，转折处用含蓄表达。",
    lexicon: "檐雨\n青灯\n故人\n长阶\n旧约\n余香",
  },
  dialogue: {
    label: "对话口语",
    guide: "对白自然、有停顿和潜台词，避免所有人物说话同一种腔调。\n叙述服务场面调度，少用抽象总结，多让冲突在对话里显形。",
    lexicon: "别急\n说清楚\n你知道的\n就现在\n等等\n听我说",
  },
  direct: {
    label: "直白强表达",
    guide: "表达直接，动作和情绪落到具体词，不主动把强烈表达改成含混或委婉说法。\n保持场景连续，不用空泛形容替代人物的明确选择。",
    lexicon: "逼近\n失控\n欲望\n疼痛\n占有\n撕裂",
  },
};

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${type === "error" ? "is-error" : ""}`;
  toast.textContent = message;
  elements.toastRegion.append(toast);
  window.setTimeout(() => toast.remove(), 3200);
}

function announceTabConversation(type = "active") {
  state.tabChannel?.postMessage({
    type,
    tab_id: state.tabId,
    conversation_id: state.pendingConversationId || state.conversation?.id || null,
  });
}

function initializeWindowIsolation() {
  if (!("BroadcastChannel" in window)) return;
  state.tabChannel = new BroadcastChannel("llm4chat-window-isolation-v1");
  state.tabChannel.addEventListener("message", (event) => {
    const message = event.data || {};
    if (message.tab_id === state.tabId) return;
    if (message.type === "probe") {
      announceTabConversation();
    } else if (message.type === "active" && message.conversation_id) {
      state.tabClaims.add(message.conversation_id);
    }
  });
  window.addEventListener("beforeunload", () => announceTabConversation("release"));
}

async function conversationOpenElsewhere(conversationId) {
  if (!state.tabChannel) return false;
  state.pendingConversationId = conversationId;
  state.tabClaims.delete(conversationId);
  state.tabChannel.postMessage({ type: "probe", tab_id: state.tabId });
  await new Promise((resolve) => window.setTimeout(resolve, 90));
  const claimed = state.tabClaims.has(conversationId);
  if (claimed) state.pendingConversationId = state.conversation?.id || null;
  return claimed;
}

function errorMessage(error) {
  if (error instanceof ApiError) return error.message;
  if (error?.name === "AbortError") return "生成已停止";
  return error?.message || "发生了未知错误";
}

function setTheme(theme) {
  localStorage.setItem("llm4chat-theme", theme);
  const resolved =
    theme === "system"
      ? window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light"
      : theme;
  document.documentElement.dataset.theme = resolved;
  document.querySelector('meta[name="theme-color"]').content = resolved === "dark" ? "#212121" : "#ffffff";
  document.querySelectorAll("[data-theme]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.theme === theme);
  });
}

function closeMobileSidebar() {
  elements.app.classList.remove("mobile-sidebar-open");
}

function syncBodyLock() {
  const panelOpen = !elements.settingsPanel.hidden || !elements.projectPanel.hidden || !elements.outlinePanel.hidden || !elements.incrementDialog.hidden;
  document.body.style.overflow = panelOpen ? "hidden" : "";
}

function closeIncrement() {
  if (state.incrementRunning) return;
  elements.incrementBackdrop.hidden = true;
  elements.incrementDialog.hidden = true;
  state.incrementCandidate = null;
  syncBodyLock();
}

function closeProject() {
  elements.projectBackdrop.hidden = true;
  elements.projectPanel.hidden = true;
  syncBodyLock();
}

function closeOutline() {
  elements.outlineBackdrop.hidden = true;
  elements.outlinePanel.hidden = true;
  syncBodyLock();
}

function autoResizeComposer() {
  const input = elements.composerInput;
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
  updateSendButton();
}

function updateSendButton() {
  const hasText = Boolean(elements.composerInput.value.trim());
  elements.sendButton.classList.toggle("is-generating", state.generating);
  const blockedByTask = state.analysisRunning || state.outlineGenerating;
  elements.sendButton.disabled = state.generating ? false : blockedByTask || !hasText || state.runtime?.status !== "ready";
  elements.sendButton.setAttribute("aria-label", state.generating ? "停止生成" : "发送");
}

function compactTokens(value) {
  const number = Number(value || 0);
  if (number >= 1000) return `${(number / 1000).toFixed(number >= 10000 ? 0 : 1)}K`;
  return String(number);
}

function renderContextUsage(stats = state.contextStats) {
  const size = Number(stats?.context_size || state.runtime?.context_size || 32768);
  const input = Number(stats?.input_tokens || 0);
  const reserved = Number(stats?.reserved_output_tokens || currentGenerationSettings().max_tokens + 384);
  const used = input + reserved;
  const ratio = Math.min(1, used / size);
  elements.contextUsageText.textContent = stats
    ? `输入 ${compactTokens(input)} · 预留 ${compactTokens(reserved)} / ${Math.round(size / 1024)}K`
    : `上下文 ${Math.round(size / 1024)}K`;
  elements.contextUsage.title = stats
    ? `提示词 ${input} tokens · 为输出预留 ${reserved} tokens · 窗口 ${size} tokens`
    : `当前上下文窗口 ${size} tokens`;
  elements.contextUsageBar.style.width = `${Math.round(ratio * 100)}%`;
  elements.contextUsage.classList.toggle("is-warning", ratio >= 0.72 && ratio < 0.9);
  elements.contextUsage.classList.toggle("is-danger", ratio >= 0.9);
}

async function updateContextUsage() {
  if (!state.conversation || state.runtime?.status !== "ready" || state.generating || state.analysisRunning || state.outlineGenerating) {
    renderContextUsage();
    return;
  }
  try {
    state.contextStats = await api.countContext(state.conversation.id, elements.composerInput.value.trim());
    renderContextUsage();
  } catch {
    renderContextUsage();
  }
}

function scheduleContextUsage() {
  window.clearTimeout(state.contextTimer);
  state.contextTimer = window.setTimeout(updateContextUsage, 650);
}

function saveDraft() {
  if (!state.conversation) return;
  localStorage.setItem(`llm4chat-draft:${state.conversation.id}`, elements.composerInput.value);
}

function restoreDraft() {
  const value = state.conversation
    ? localStorage.getItem(`llm4chat-draft:${state.conversation.id}`) || ""
    : "";
  elements.composerInput.value = value;
  autoResizeComposer();
}

function chapterStatusLabel(status) {
  return {
    pending: "待总结",
    processing: "总结中",
    completed: "已完成",
    failed: "失败",
  }[status] || status;
}

function renderProject() {
  const project = state.project;
  const workspace = state.workspace;
  if (!project) return;
  elements.projectCounts.textContent = `${project.documents.length} 个 TXT${workspace ? ` · ${workspace.chapter_count} 章 · ${workspace.character_count} 人` : ""}`;
  elements.documentSelect.replaceChildren();
  for (const sourceDocument of project.documents) {
    const option = document.createElement("option");
    option.value = sourceDocument.id;
    option.textContent = `${sourceDocument.filename}（${sourceDocument.chapter_count} 章）`;
    elements.documentSelect.append(option);
  }
  if (!project.documents.length) {
    const option = document.createElement("option");
    option.textContent = "尚未导入 TXT";
    option.value = "";
    elements.documentSelect.append(option);
  }
  elements.documentSelect.value = workspace?.id || "";
  elements.documentSelect.disabled = state.analysisRunning || !project.documents.length;
  elements.documentList.innerHTML = project.documents.map((item) => `
    <div class="document-row ${item.id === workspace?.id ? "is-active" : ""}" data-document-id="${item.id}">
      <button class="document-switch" type="button">${escapeText(item.filename)}</button>
      <button class="danger-link delete-document" type="button">删除</button>
    </div>`).join("");
  elements.documentList.querySelectorAll(".document-switch").forEach((button) => {
    button.addEventListener("click", () => selectDocument(button.closest(".document-row").dataset.documentId));
  });
  elements.documentList.querySelectorAll(".delete-document").forEach((button) => {
    button.addEventListener("click", async () => {
      const documentId = button.closest(".document-row").dataset.documentId;
      if (!window.confirm("删除这本 TXT 及其章节、总览、人物卡、事实和任务记录吗？")) return;
      try {
        state.project = await api.deleteDocument(documentId);
        const next = state.project.documents[0]?.id || null;
        state.workspace = next ? await api.getDocumentWorkspace(next) : null;
        if (state.conversation) state.conversation = await api.updateConversation(state.conversation.id, { document_id: next });
        renderProject();
        scheduleContextUsage();
      } catch (error) { showToast(errorMessage(error), "error"); }
    });
  });

  const disabled = !workspace;
  [elements.libraryEnabled, elements.summaryEnabled, elements.recentChaptersEnabled,
   elements.charactersEnabled, elements.factsEnabled, elements.globalSummary,
   elements.summarizeProject, elements.analysisStart, elements.analysisEnd,
   elements.previewPrompt, elements.exportMaterialPackage, elements.rebuildMaterialSystem,
   elements.previewMaterialPlan, elements.editMaterialBudget, elements.refreshMaterialReviews,
   elements.inspectMaterialSystem].forEach((element) => { element.disabled = disabled; });
  elements.importMaterialPackage.disabled = state.analysisRunning;
  if (!workspace) {
    elements.globalSummary.value = "";
    elements.chapterList.className = "workspace-list empty-list";
    elements.chapterList.textContent = "请先导入或选择 TXT";
    elements.characterList.className = "workspace-list empty-list";
    elements.characterList.textContent = "请先导入或选择 TXT";
    elements.factList.className = "workspace-list empty-list";
    elements.factList.textContent = "请先导入或选择 TXT";
    elements.analysisStart.replaceChildren();
    elements.analysisEnd.replaceChildren();
    elements.resumeAnalysis.hidden = true;
    elements.promptPreviewBox.hidden = true;
    elements.materialPackageReport.hidden = true;
    elements.materialPackageReport.textContent = "";
    state.materialReviewItems = [];
    state.materialReviewsLoaded = false;
    state.materialOverview = null;
    state.materialInspectorLoaded = false;
    state.materialBudgetProfile = null;
    state.materialBudgetLoaded = false;
    renderMaterialBudgetEditor();
    renderMaterialInspector();
    renderMaterialReviewItems();
    return;
  }
  elements.libraryEnabled.checked = workspace.library_enabled;
  elements.summaryEnabled.checked = workspace.summary_enabled;
  elements.recentChaptersEnabled.checked = workspace.recent_chapters_enabled;
  elements.charactersEnabled.checked = workspace.characters_enabled;
  elements.factsEnabled.checked = workspace.facts_enabled;
  elements.globalSummary.value = workspace.global_summary || "";
  elements.analysisTokenNote.textContent = `当前只处理《${workspace.filename}》。每完成一个分片立即保存，可停止后从断点继续。`;
  elements.resumeAnalysis.hidden = workspace.latest_job?.status !== "paused";
  renderMaterialInspector();
  renderMaterialReviewItems();
  for (const select of [elements.analysisStart, elements.analysisEnd]) {
    select.replaceChildren();
    for (const chapter of workspace.chapters) {
      const option = document.createElement("option");
      option.value = chapter.position;
      option.textContent = `${chapter.position}. ${chapter.title}`;
      select.append(option);
    }
  }
  if (workspace.chapters.length) {
    const firstPending = workspace.chapters.find((chapter) => chapter.status !== "completed") || workspace.chapters[0];
    elements.analysisStart.value = firstPending.position;
    elements.analysisEnd.value = workspace.chapters.at(-1).position;
  }

  renderMaterialBudgetEditor();
  elements.chapterList.className = workspace.chapters.length ? "workspace-list" : "workspace-list empty-list";
  elements.chapterList.innerHTML = workspace.chapters.length ? workspace.chapters.map((chapter) => `
    <details class="workspace-card chapter-card" data-chapter-id="${chapter.id}">
      <summary><span class="workspace-card-title">${escapeText(chapter.title)}</span>
      <span class="workspace-card-meta">${chapter.character_count.toLocaleString()} 字 · ${chapter.chunk_count} 片</span>
      <span class="status-pill is-${chapter.status}">${chapterStatusLabel(chapter.status)}</span></summary>
      <div class="workspace-card-body">
        <textarea class="workspace-editor chapter-summary-editor" rows="7">${escapeText(chapter.summary_text || "")}</textarea>
        ${chapter.error_message ? `<p class="settings-note">${escapeText(chapter.error_message)}</p>` : ""}
        <div class="workspace-actions"><button class="secondary-button summarize-chapter" type="button">${chapter.status === "completed" ? "重新总结" : "总结/续行本章"}</button>
        <button class="secondary-button save-chapter-summary" type="button">保存摘要</button><button class="danger-button delete-chapter" type="button">删除章节</button></div>
      </div></details>`).join("") : "还没有章节";
  elements.chapterList.querySelectorAll(".chapter-card").forEach((card) => {
    const chapterId = card.dataset.chapterId;
    card.querySelector(".save-chapter-summary").addEventListener("click", async () => {
      try {
        const updated = await api.updateChapter(chapterId, { edited_summary: card.querySelector("textarea").value });
        const index = state.workspace.chapters.findIndex((item) => item.id === chapterId);
        if (index >= 0) state.workspace.chapters[index] = updated;
        showToast("章节摘要已保存");
      } catch (error) { showToast(errorMessage(error), "error"); }
    });
    card.querySelector(".summarize-chapter").addEventListener("click", () => runProjectSummary([chapterId], true));
    card.querySelector(".delete-chapter").addEventListener("click", async () => {
      if (!window.confirm("删除这个章节吗？当前 TXT 的派生总览、人物卡和事实会清空。")) return;
      try { state.workspace = await api.deleteChapter(chapterId); renderProject(); scheduleContextUsage(); }
      catch (error) { showToast(errorMessage(error), "error"); }
    });
  });

  elements.characterList.className = workspace.characters.length ? "workspace-list" : "workspace-list empty-list";
  elements.characterList.innerHTML = workspace.characters.length ? workspace.characters.map((character) => `
    <details class="workspace-card character-card" data-character-id="${character.id}"><summary>
    <span class="workspace-card-title">${escapeText(character.name)}</span><span class="workspace-card-meta">${escapeText((character.aliases || []).join("、"))}</span>
    <label class="compact-toggle"><span>${character.enabled ? "已启用" : "未启用"}</span><input type="checkbox" ${character.enabled ? "checked" : ""}/><i></i></label></summary>
    <div class="workspace-card-body"><textarea class="workspace-editor character-card-editor" rows="10">${escapeText(character.prompt_text || "")}</textarea>
    <div class="workspace-actions"><button class="danger-button delete-character" type="button">删除</button><button class="secondary-button save-character" type="button">保存</button></div></div></details>`).join("") : "总结完成后会在这里生成人物卡";
  elements.characterList.querySelectorAll(".character-card").forEach((card) => {
    const id = card.dataset.characterId;
    const toggle = card.querySelector("input");
    toggle.addEventListener("click", (event) => event.stopPropagation());
    toggle.addEventListener("change", async () => {
      try { const updated = await api.updateCharacter(id, { enabled: toggle.checked }); Object.assign(state.workspace.characters.find((x) => x.id === id), updated); scheduleContextUsage(); }
      catch (error) { toggle.checked = !toggle.checked; showToast(errorMessage(error), "error"); }
    });
    card.querySelector(".save-character").addEventListener("click", async () => {
      try { await api.updateCharacter(id, { prompt_text: card.querySelector("textarea").value }); showToast("人物卡已保存"); scheduleContextUsage(); }
      catch (error) { showToast(errorMessage(error), "error"); }
    });
    card.querySelector(".delete-character").addEventListener("click", async () => {
      if (!window.confirm("删除这张人物卡吗？")) return;
      try { state.workspace = await api.deleteCharacter(id); renderProject(); scheduleContextUsage(); }
      catch (error) { showToast(errorMessage(error), "error"); }
    });
  });

  elements.factList.className = workspace.facts.length ? "workspace-list" : "workspace-list empty-list";
  elements.factList.innerHTML = workspace.facts.length ? workspace.facts.map((fact) => `
    <div class="workspace-card fact-card" data-fact-id="${fact.id}"><div class="workspace-card-body">
    <p><b>[${escapeText(fact.fact_type)}]</b> ${escapeText(fact.subject)} ${escapeText(fact.predicate)} ${escapeText(fact.object)}</p>
    <p class="settings-note">状态：${escapeText(fact.state || fact.status)} · 首次：${escapeText(fact.first_chapter || "未知")} · 最近更新：${escapeText(fact.last_chapter || "未知")}</p>
    <div class="workspace-actions"><button class="secondary-button resolve-fact" type="button" ${fact.status === "resolved" ? "disabled" : ""}>${fact.status === "resolved" ? "已回收" : "标记已回收"}</button><button class="danger-button delete-fact" type="button">删除</button></div>
    </div></div>`).join("") : "总结章节后会在这里生成结构化事实";
  elements.factList.querySelectorAll(".fact-card").forEach((card) => {
    const id = card.dataset.factId;
    card.querySelector(".resolve-fact").addEventListener("click", async () => {
      try {
        await api.updateFact(id, { status: "resolved" });
        state.workspace = await api.getDocumentWorkspace(workspace.id);
        renderProject();
        scheduleContextUsage();
      } catch (error) { showToast(errorMessage(error), "error"); }
    });
    card.querySelector(".delete-fact").addEventListener("click", async () => {
      if (!window.confirm("删除这条事实吗？")) return;
      try {
        state.workspace = await api.deleteFact(id);
        renderProject();
        scheduleContextUsage();
      } catch (error) { showToast(errorMessage(error), "error"); }
    });
  });
}

async function selectDocument(documentId) {
  if (!documentId) return;
  state.workspace = await api.getDocumentWorkspace(documentId);
  state.materialReviewItems = [];
  state.materialReviewsLoaded = false;
  state.materialOverview = null;
  state.materialInspectorLoaded = false;
  state.materialBudgetProfile = null;
  state.materialBudgetLoaded = false;
  if (state.conversation?.document_id !== documentId) {
    state.conversation = await api.updateConversation(state.conversation.id, { document_id: documentId });
  }
  renderProject();
  scheduleContextUsage();
}

async function saveDocumentSetting(field, value) {
  if (!state.workspace) return;
  try {
    state.workspace = await api.updateDocument(state.workspace.id, { [field]: value });
    renderProject();
    scheduleContextUsage();
  } catch (error) {
    renderProject();
    showToast(errorMessage(error), "error");
  }
}

async function previewInjectedPrompt() {
  if (!state.conversation || !state.workspace) return;
  try {
    const result = await api.promptPreview(
      state.conversation.id,
      elements.composerInput.value.trim(),
    );
    const labels = {
      system_prompt: "系统提示词",
      pinned_context: "固定创作资料",
      style_guide: "词汇风格",
      style_lexicon: "词表白名单 / 优先用词",
      project_summary: "前文总览",
      recent_chapters: "最近章节结构摘要",
      characters: "人物卡",
      facts: "相关结构化事实",
      outline: "已选大纲",
    };
    const fixedEntries = [
      ["system_prompt", result.system_prompt],
      ["pinned_context", result.pinned_context],
      ["style_guide", result.style_guide],
      ["style_lexicon", result.style_lexicon],
      ...Object.entries(result.sources),
    ];
    const sections = fixedEntries
      .filter(([, value]) => String(value || "").trim())
      .map(([key, value]) => `## ${labels[key] || key}\n\n${value}`);
    elements.promptPreviewContent.value = sections.length
      ? sections.join("\n\n---\n\n")
      : "当前没有固定提示词、小说资料或大纲会注入。";
    elements.promptPreviewBox.hidden = false;
    elements.promptPreviewBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    showToast(errorMessage(error), "error");
  }
}

async function loadProject(preferredDocumentId = null) {
  state.project = await api.getProject("default");
  const selectedId = preferredDocumentId || state.conversation?.document_id || state.workspace?.id || state.project.documents[0]?.id;
  state.workspace = selectedId && state.project.documents.some((item) => item.id === selectedId)
    ? await api.getDocumentWorkspace(selectedId) : null;
  state.materialReviewItems = [];
  state.materialReviewsLoaded = false;
  state.materialOverview = null;
  state.materialInspectorLoaded = false;
  state.materialBudgetProfile = null;
  state.materialBudgetLoaded = false;
  renderProject();
}

async function openProject() {
  closeSettings();
  closeOutline();
  elements.projectBackdrop.hidden = false;
  elements.projectPanel.hidden = false;
  syncBodyLock();
  closeMobileSidebar();
  try {
    await loadProject();
  } catch (error) {
    showToast(errorMessage(error), "error");
  }
}

async function saveProjectSummary() {
  if (!state.workspace) return;
  try {
    state.workspace = await api.updateDocument(state.workspace.id, {
      global_summary: elements.globalSummary.value,
      summary_enabled: elements.summaryEnabled.checked,
    });
    renderProject();
    showToast("前文总览已保存");
    scheduleContextUsage();
  } catch (error) {
    showToast(errorMessage(error), "error");
  }
}

function exportProjectTxt() {
  if (!state.workspace || !state.workspace.chapters.length) {
    showToast("资料库里还没有可导出的章节", "error");
    return;
  }
  const link = document.createElement("a");
  link.href = `/api/documents/${state.workspace.id}/export.txt`;
  link.click();
}

function formatMaterialPackageReport(report) {
  const packageInfo = report.package || {};
  const checks = report.checks || {};
  const target = report.target || {};
  const diffPreview = report.diff_preview || {};
  const layerCounts = packageInfo.material_layer_counts || {};
  const scope = report.scope || {};
  const scopedLayerCounts = packageInfo.scoped_material_layer_counts || {};
  const layerLines = ["observations", "timeline", "characters", "reviews", "auxiliary", "budget"]
    .map((layer) => {
      const scopedText = scope.enabled ? `，范围内 ${Number(scopedLayerCounts[layer] || 0)}` : "";
      return `- ${materialLayerLabel(layer)}：${Number(layerCounts[layer] || 0)}${scopedText}`;
    })
    .join("\n");
  const diffLines = ["observations", "timeline", "characters", "reviews", "auxiliary", "budget"]
    .map((layer) => {
      const preview = diffPreview.layers?.[layer];
      if (!preview) return "";
      const samples = Array.isArray(preview.samples) && preview.samples.length
        ? `；样例：${preview.samples.slice(0, 3).map(formatMaterialDiffSample).join("、")}`
        : "";
      return `- ${materialLayerLabel(layer)}：传入 ${preview.incoming || 0} / 新增 ${preview.added || 0} / 更新 ${preview.updated || 0} / 相同 ${preview.unchanged || 0} / 本地独有 ${preview.local_only || 0}${samples}`;
    })
    .filter(Boolean)
    .join("\n");
  const matchingDocuments = Array.isArray(target.matching_documents) ? target.matching_documents : [];
  const matchingText = matchingDocuments.length
    ? matchingDocuments
      .slice(0, 5)
      .map((document) => document.filename || document.id)
      .join("、") + (matchingDocuments.length > 5 ? ` 等 ${matchingDocuments.length} 个` : "")
    : "未发现相同原文";
  const lines = [
    `文件：${packageInfo.filename || "未命名分析包"}`,
    `模式：${target.mode === "pure_new_file" ? "纯新文件导入" : "匹配现有文档"}`,
    `schema：${checks.schema || "未知"}`,
    `包内原文 hash：${packageInfo.source_document_hash || "未知"}（${checks.package_source_document_hash || "未检查"}）`,
    `章节数：${packageInfo.chapter_count ?? 0}（${checks.chapter_count || "未检查"}）`,
    `chunk 数：${packageInfo.chunk_count ?? 0}`,
    `资料层：\n${layerLines}`,
    `可安全导入记录：${checks.safe_records ?? 0}`,
    `需确认记录：${checks.review_records ?? 0}`,
    `拒绝记录：${checks.rejected_records ?? 0}`,
  ];
  if (scope.enabled) {
    lines.push(`章节范围：${scope.chapter_start || 1}-${scope.chapter_end || "末尾"}（匹配 ${scope.matched_chapter_count || 0} 章）`);
  }
  if (target.mode === "pure_new_file") {
    lines.push(`匹配本地 TXT：${matchingText}`);
  } else {
    lines.push(
      `目标 TXT：${target.filename || target.document_id || "未选择"}${
        target.chapter_count == null ? "" : `（${target.chapter_count} 章）`
      }`,
      `目标原文 hash：${target.source_document_hash || "未知"}（${checks.source_document_hash || "未检查"}）`,
    );
    if (diffLines) {
      lines.push(`逐条对照：\n${diffLines}`);
    }
  }
  if (Array.isArray(report.actions) && report.actions.length) {
    lines.push("", ...report.actions);
  }
  return lines.join("\n");
}

function selectedMaterialLayers() {
  return [...document.querySelectorAll('input[name="material-layer"]:checked')]
    .map((input) => input.value);
}

function selectedMaterialScope() {
  const rawStart = elements.materialScopeStart?.value.trim() || "";
  const rawEnd = elements.materialScopeEnd?.value.trim() || "";
  if (!rawStart && !rawEnd) return null;
  const chapterStart = rawStart ? Number(rawStart) : null;
  const chapterEnd = rawEnd ? Number(rawEnd) : null;
  if ((chapterStart !== null && (!Number.isInteger(chapterStart) || chapterStart < 1))
    || (chapterEnd !== null && (!Number.isInteger(chapterEnd) || chapterEnd < 1))) {
    return { error: "章节范围必须是正整数" };
  }
  if (chapterStart !== null && chapterEnd !== null && chapterEnd < chapterStart) {
    return { error: "结束章节不能小于起始章节" };
  }
  return {
    chapterStart,
    chapterEnd,
  };
}

function materialScopeLabel(scope) {
  if (!scope) return "";
  return `${scope.chapterStart || 1}-${scope.chapterEnd || "末尾"}`;
}

function materialLayerLabel(layer) {
  return {
    observations: "语义观察",
    timeline: "时间线",
    characters: "人物 / 关系",
    reviews: "确认队列",
    auxiliary: "辅助账本",
    budget: "预算配置",
  }[layer] || layer;
}

function formatMaterialDiffSample(sample) {
  const status = {
    added: "+",
    updated: "~",
    unchanged: "=",
  }[sample.status] || "?";
  return `${status}${sample.label || sample.id || sample.file || "记录"}`;
}

function formatMaterialOverview(overview) {
  const timelineNodes = overview.timeline?.nodes || [];
  const timelineEvents = overview.timeline?.events || [];
  const characters = overview.characters || [];
  const relationships = overview.relationships || [];
  const auxiliaryRecords = overview.auxiliary_records || [];
  const reviewItems = overview.review_items || [];
  return [
    "实验资料系统已重建",
    `时间线节点：${timelineNodes.length}`,
    `时间线事件：${timelineEvents.length}`,
    `人物实体：${characters.length}`,
    `关系边：${relationships.length}`,
    `辅助账本：${auxiliaryRecords.length}`,
    `待确认项：${reviewItems.filter((item) => item.status === "pending").length}`,
  ].join("\n");
}

function formatMaterialPromptPlan(plan) {
  const lines = [
    `提示词预算：${plan.total_tokens} / ${plan.max_tokens} tokens`,
    "",
    ...plan.sections.map((section) => {
      const state = section.included ? "加入" : `跳过${section.reason ? `：${section.reason}` : ""}`;
      return `${state} · ${section.label} · ${section.tokens}/${section.budget} tokens`;
    }),
  ];
  if (plan.trimmed?.length) {
    lines.push("", "裁剪：", ...plan.trimmed.map((item) => `- ${item.key}：${item.reason}`));
  }
  return lines.join("\n");
}

const materialBudgetLabels = {
  project_summary: "前文总览",
  current_timeline_node: "当前时间线节点",
  recent_chapter_summaries: "最近章节摘要",
  timeline_events: "时间线事件",
  character_snapshots: "人物当前快照",
  relationships: "人物关系",
  auxiliary_records: "地点 / 物件 / 悬念",
  facts: "结构化事实",
  outline: "下一章大纲",
};

function renderMaterialBudgetEditor() {
  if (!state.materialBudgetLoaded || !state.materialBudgetProfile) {
    elements.materialBudgetEditor.hidden = true;
    elements.materialBudgetEditor.textContent = "";
    return;
  }
  const config = state.materialBudgetProfile.config || {};
  elements.materialBudgetEditor.hidden = false;
  elements.materialBudgetEditor.innerHTML = `
    <div class="section-heading-row material-budget-heading">
      <h3>提示词预算设置</h3>
      <span class="muted-badge">${escapeText(state.materialBudgetProfile.name || "默认预算")}</span>
    </div>
    <div class="material-budget-grid">
      ${Object.entries(materialBudgetLabels).map(([key, label]) => `
        <label class="material-budget-field">
          <span>${escapeText(label)}</span>
          <input type="number" min="0" max="50000" step="100" data-budget-key="${escapeText(key)}" value="${Number(config[key] ?? 0)}" />
        </label>
      `).join("")}
    </div>
    <div class="workspace-actions">
      <button id="save-material-budget" class="secondary-button" type="button">保存预算</button>
    </div>
  `;
  elements.materialBudgetEditor.querySelector("#save-material-budget").addEventListener("click", saveMaterialBudgetProfile);
}

function compactList(items, fallback = "无") {
  const values = (items || []).map((item) => String(item || "").trim()).filter(Boolean);
  return values.length ? values.join("、") : fallback;
}

function splitMaterialList(text) {
  return String(text || "")
    .split(/[、，,\n;；|]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function materialAuxiliaryTypeLabel(type) {
  return {
    location: "地点",
    object: "物件",
    unresolved: "悬念",
  }[type] || type || "辅助";
}

function materialAuxiliaryTypeOptions(selected) {
  return [
    ["location", "地点"],
    ["object", "物件"],
    ["unresolved", "悬念"],
  ].map(([value, label]) => `<option value="${value}" ${selected === value ? "selected" : ""}>${label}</option>`).join("");
}

function renderMaterialInspector() {
  if (!state.materialInspectorLoaded) {
    elements.materialInspector.hidden = true;
    elements.materialInspector.textContent = "";
    return;
  }
  const overview = state.materialOverview || {};
  const timelineNodes = overview.timeline?.nodes || [];
  const timelineEvents = overview.timeline?.events || [];
  const characters = overview.characters || [];
  const relationships = overview.relationships || [];
  const auxiliaryRecords = overview.auxiliary_records || [];
  const reviewItems = overview.review_items || [];
  const pendingCount = reviewItems.filter((item) => item.status === "pending").length;
  elements.materialInspector.hidden = false;
  elements.materialInspector.innerHTML = `
    <div class="section-heading-row material-inspector-heading">
      <h3>实验资料视图</h3>
      <span class="muted-badge">${timelineEvents.length} 事件 · ${characters.length} 人物 · ${relationships.length} 关系 · ${auxiliaryRecords.length} 辅助</span>
    </div>
    <div class="material-inspector-grid">
      <section class="material-inspector-column">
        <div class="material-inspector-title">时间线</div>
        <div class="material-inspector-list">
          <article class="material-inspector-item material-node-create">
            <label class="material-inspector-field">
              <span>新节点</span>
              <input class="material-new-node-title" type="text" placeholder="卷 / 阶段 / 故事弧" />
            </label>
            <label class="material-inspector-field">
              <span>类型</span>
              <select class="material-new-node-type">
                ${[
                  ["stage", "阶段"],
                  ["volume", "卷"],
                  ["arc", "故事弧"],
                  ["chapter_group", "章节组"],
                  ["scene", "场景"],
                ].map(([value, label]) => `<option value="${value}">${label}</option>`).join("")}
              </select>
            </label>
            <label class="material-inspector-field">
              <span>摘要</span>
              <textarea class="material-new-node-summary" rows="2"></textarea>
            </label>
            <div class="material-inspector-actions"><button class="secondary-button create-material-node" type="button">新建节点</button></div>
          </article>
          ${timelineNodes.length ? timelineNodes.map((node) => `
            <article class="material-inspector-item material-node-item" data-node-id="${escapeText(node.id)}">
              <label class="material-inspector-field">
                <span>节点标题</span>
                <input class="material-node-title" type="text" value="${escapeText(node.title || node.node_type || "节点")}" />
              </label>
              <label class="material-inspector-field">
                <span>节点摘要</span>
                <textarea class="material-node-summary" rows="3">${escapeText(node.summary || "")}</textarea>
              </label>
              <label class="material-inspector-check">
                <input class="material-node-enabled" type="checkbox" ${node.enabled ? "checked" : ""} />
                <span>启用</span>
              </label>
              <small>${escapeText(node.node_type || "node")} · ${node.manually_edited ? "人工编辑" : "自动生成"}</small>
              <div class="material-inspector-actions">
                <button class="secondary-button save-material-node" type="button">保存节点</button>
                <button class="danger-button delete-material-node" type="button">删除</button>
              </div>
            </article>
          `).join("") : '<div class="empty-list">暂无时间线节点</div>'}
          <article class="material-inspector-item material-event-create">
            <label class="material-inspector-field">
              <span>新事件</span>
              <input class="material-new-event-title" type="text" placeholder="关键事件" />
            </label>
            <label class="material-inspector-field">
              <span>类型</span>
              <input class="material-new-event-type" type="text" value="event" />
            </label>
            <label class="material-inspector-field">
              <span>状态</span>
              <select class="material-new-event-status">
                ${["active", "resolved", "disabled"].map((status) => `<option value="${status}">${status}</option>`).join("")}
              </select>
            </label>
            <label class="material-inspector-field">
              <span>描述</span>
              <textarea class="material-new-event-description" rows="2"></textarea>
            </label>
            <div class="material-inspector-actions"><button class="secondary-button create-material-event" type="button">新建事件</button></div>
          </article>
          ${timelineEvents.length ? timelineEvents.map((event) => `
            <article class="material-inspector-item material-event-item" data-event-id="${escapeText(event.id)}">
              <label class="material-inspector-field">
                <span>标题</span>
                <input class="material-event-title" type="text" value="${escapeText(event.title || event.event_type || "事件")}" />
              </label>
              <label class="material-inspector-field">
                <span>描述</span>
                <textarea class="material-event-description" rows="3">${escapeText(event.description || "")}</textarea>
              </label>
              <label class="material-inspector-field">
                <span>状态</span>
                <select class="material-event-status">
                  ${["active", "resolved", "disabled"].map((status) => `<option value="${status}" ${event.status === status ? "selected" : ""}>${status}</option>`).join("")}
                </select>
              </label>
              <small>${escapeText(event.event_type || "event")} · ${escapeText(compactList(event.participants))}</small>
              <div class="material-inspector-actions">
                <button class="secondary-button save-material-event" type="button">保存</button>
                <button class="danger-button delete-material-event" type="button">删除</button>
              </div>
            </article>
          `).join("") : '<div class="empty-list">暂无时间线事件</div>'}
        </div>
        <small class="material-inspector-footnote">节点 ${timelineNodes.length}</small>
      </section>
      <section class="material-inspector-column">
        <div class="material-inspector-title">人物</div>
        <div class="material-inspector-list">
          <article class="material-inspector-item material-character-create">
            <label class="material-inspector-field">
              <span>新人物</span>
              <input class="material-new-character-name" type="text" placeholder="人物名" />
            </label>
            <label class="material-inspector-field">
              <span>别名</span>
              <input class="material-new-character-aliases" type="text" placeholder="别名 / 称谓" />
            </label>
            <label class="material-inspector-field">
              <span>身份 / 当前档案</span>
              <textarea class="material-new-character-identity" rows="2"></textarea>
            </label>
            <label class="material-inspector-check">
              <input class="material-new-character-enabled" type="checkbox" checked />
              <span>启用</span>
            </label>
            <div class="material-inspector-actions"><button class="secondary-button create-material-character" type="button">新建人物</button></div>
          </article>
          ${characters.length ? characters.map((character) => {
            const profiles = character.profiles || [];
            const facts = character.facts || [];
            const events = character.events || [];
            const profile = profiles[0] || {};
            return `
              <article class="material-inspector-item material-character-item" data-character-id="${escapeText(character.id)}">
                <label class="material-inspector-field">
                  <span>人物名</span>
                  <input class="material-character-name" type="text" value="${escapeText(character.canonical_name)}" />
                </label>
                <label class="material-inspector-field">
                  <span>身份 / 当前档案</span>
                  <textarea class="material-character-identity" rows="3">${escapeText(profile.identity || profile.behavior_pattern || "")}</textarea>
                </label>
                <label class="material-inspector-check">
                  <input class="material-character-enabled" type="checkbox" ${character.enabled ? "checked" : ""} />
                  <span>启用</span>
                </label>
                <small>${character.enabled ? "启用" : "停用"} · ${character.manually_confirmed ? "已确认" : "未确认"} · ${escapeText(compactList((character.aliases || []).map((alias) => alias.alias)))}</small>
                <div class="material-profile-list">
                  <small>阶段档案 ${profiles.length}</small>
                  ${profiles.map((profileItem) => `
                    <div class="material-profile-row" data-profile-id="${escapeText(profileItem.id)}">
                      <label class="material-inspector-field">
                        <span>阶段标题</span>
                        <input class="material-profile-title" type="text" value="${escapeText(profileItem.title || "阶段档案")}" />
                      </label>
                      <label class="material-inspector-field">
                        <span>阶段身份</span>
                        <textarea class="material-profile-identity" rows="2">${escapeText(profileItem.identity || profileItem.behavior_pattern || "")}</textarea>
                      </label>
                      <label class="material-inspector-check">
                        <input class="material-profile-enabled" type="checkbox" ${profileItem.enabled ? "checked" : ""} />
                        <span>启用</span>
                      </label>
                      <div class="material-inspector-actions">
                        <button class="secondary-button save-material-profile" type="button">保存阶段</button>
                        <button class="danger-button delete-material-profile" type="button">删除阶段</button>
                      </div>
                    </div>
                  `).join("")}
                  <div class="material-profile-row material-profile-create">
                    <label class="material-inspector-field">
                      <span>新增阶段</span>
                      <input class="material-new-profile-title" type="text" placeholder="阶段标题" />
                    </label>
                    <label class="material-inspector-field">
                      <span>阶段身份</span>
                      <textarea class="material-new-profile-identity" rows="2"></textarea>
                    </label>
                    <div class="material-inspector-actions">
                      <button class="secondary-button create-material-profile" type="button">新建阶段</button>
                    </div>
                  </div>
                </div>
                <div class="material-profile-list">
                  <small>人物事实 ${facts.length}</small>
                  ${facts.map((fact) => `
                    <div class="material-profile-row" data-character-fact-id="${escapeText(fact.id)}">
                      <label class="material-inspector-field">
                        <span>事实字段</span>
                        <input class="material-character-fact-field" type="text" value="${escapeText(fact.field || "")}" />
                      </label>
                      <label class="material-inspector-field">
                        <span>事实内容</span>
                        <textarea class="material-character-fact-value" rows="2">${escapeText(fact.value || "")}</textarea>
                      </label>
                      <label class="material-inspector-field">
                        <span>可信度</span>
                        <input class="material-character-fact-certainty" type="number" min="0" max="1" step="0.01" value="${Number(fact.certainty ?? 1).toFixed(2)}" />
                      </label>
                      <div class="material-inspector-actions">
                        <button class="secondary-button save-material-character-fact" type="button">保存事实</button>
                        <button class="danger-button delete-material-character-fact" type="button">删除事实</button>
                      </div>
                    </div>
                  `).join("")}
                  <div class="material-profile-row material-character-fact-create">
                    <label class="material-inspector-field">
                      <span>新增事实</span>
                      <input class="material-new-character-fact-field" type="text" placeholder="身份 / 位置 / 能力 / 状态" />
                    </label>
                    <label class="material-inspector-field">
                      <span>事实内容</span>
                      <textarea class="material-new-character-fact-value" rows="2"></textarea>
                    </label>
                    <div class="material-inspector-actions">
                      <button class="secondary-button create-material-character-fact" type="button">新建事实</button>
                    </div>
                  </div>
                </div>
                <div class="material-profile-list">
                  <small>经历事件 ${events.length}</small>
                  ${events.map((event) => `
                    <div class="material-profile-row" data-character-event-id="${escapeText(event.id)}">
                      <label class="material-inspector-field">
                        <span>经历类型</span>
                        <input class="material-character-event-type" type="text" value="${escapeText(event.event_type || "event")}" />
                      </label>
                      <label class="material-inspector-field">
                        <span>经历内容</span>
                        <textarea class="material-character-event-value" rows="2">${escapeText(event.value || "")}</textarea>
                      </label>
                      <label class="material-inspector-field">
                        <span>顺序</span>
                        <input class="material-character-event-sequence" type="number" min="0" step="1" value="${Number(event.sequence ?? 0)}" />
                      </label>
                      <div class="material-inspector-actions">
                        <button class="secondary-button save-material-character-event" type="button">保存经历</button>
                        <button class="danger-button delete-material-character-event" type="button">删除经历</button>
                      </div>
                    </div>
                  `).join("")}
                  <div class="material-profile-row material-character-event-create">
                    <label class="material-inspector-field">
                      <span>新增经历</span>
                      <input class="material-new-character-event-type" type="text" value="event" />
                    </label>
                    <label class="material-inspector-field">
                      <span>经历内容</span>
                      <textarea class="material-new-character-event-value" rows="2"></textarea>
                    </label>
                    <div class="material-inspector-actions">
                      <button class="secondary-button create-material-character-event" type="button">新建经历</button>
                    </div>
                  </div>
                </div>
                <label class="material-inspector-field">
                  <span>新增别名</span>
                  <input class="material-character-alias" type="text" placeholder="别名 / 称谓" />
                </label>
                <label class="material-inspector-field">
                  <span>合并到</span>
                  <select class="material-character-merge-target">
                    <option value="">选择目标人物</option>
                    ${characters
                      .filter((target) => target.id !== character.id)
                      .map((target) => `<option value="${escapeText(target.id)}">${escapeText(target.canonical_name)}</option>`)
                      .join("")}
                  </select>
                </label>
                <label class="material-inspector-field">
                  <span>拆分为</span>
                  <input class="material-character-split-name" type="text" placeholder="新人物名" />
                </label>
                <label class="material-inspector-field">
                  <span>拆分别名</span>
                  <input class="material-character-split-aliases" type="text" placeholder="填写当前人物已有别名，可用顿号分隔" />
                </label>
                <div class="material-inspector-actions">
                  <button class="secondary-button save-material-character" type="button">保存</button>
                  <button class="secondary-button add-material-alias" type="button">加别名</button>
                  <button class="danger-button merge-material-character" type="button">合并</button>
                  <button class="secondary-button split-material-character" type="button">拆分</button>
                  <button class="danger-button delete-material-character" type="button">删除</button>
                </div>
              </article>
            `;
          }).join("") : '<div class="empty-list">暂无人物实体</div>'}
        </div>
      </section>
      <section class="material-inspector-column">
        <div class="material-inspector-title">辅助账本</div>
        <div class="material-inspector-list">
          <article class="material-inspector-item material-auxiliary-create">
            <label class="material-inspector-field">
              <span>类型</span>
              <select class="material-new-auxiliary-type">${materialAuxiliaryTypeOptions("location")}</select>
            </label>
            <label class="material-inspector-field">
              <span>名称</span>
              <input class="material-new-auxiliary-name" type="text" placeholder="地点 / 物件 / 悬念名" />
            </label>
            <label class="material-inspector-field">
              <span>摘要</span>
              <textarea class="material-new-auxiliary-summary" rows="2"></textarea>
            </label>
            <label class="material-inspector-field">
              <span>状态</span>
              <select class="material-new-auxiliary-status">
                ${["active", "resolved", "disabled"].map((status) => `<option value="${status}">${status}</option>`).join("")}
              </select>
            </label>
            <div class="material-inspector-actions"><button class="secondary-button create-material-auxiliary" type="button">新建账本</button></div>
          </article>
          ${auxiliaryRecords.length ? auxiliaryRecords.map((record) => `
            <article class="material-inspector-item material-auxiliary-item" data-auxiliary-id="${escapeText(record.id)}">
              <label class="material-inspector-field">
                <span>类型</span>
                <select class="material-auxiliary-type">${materialAuxiliaryTypeOptions(record.record_type)}</select>
              </label>
              <label class="material-inspector-field">
                <span>名称</span>
                <input class="material-auxiliary-name" type="text" value="${escapeText(record.name || "")}" />
              </label>
              <label class="material-inspector-field">
                <span>摘要</span>
                <textarea class="material-auxiliary-summary" rows="2">${escapeText(record.summary || "")}</textarea>
              </label>
              <label class="material-inspector-field">
                <span>状态</span>
                <select class="material-auxiliary-status">
                  ${["active", "resolved", "disabled"].map((status) => `<option value="${status}" ${record.status === status ? "selected" : ""}>${status}</option>`).join("")}
                </select>
              </label>
              <small>${escapeText(materialAuxiliaryTypeLabel(record.record_type))} · ${escapeText(record.status || "active")} · ${Number(record.confidence ?? 0).toFixed(2)}</small>
              <div class="material-inspector-actions">
                <button class="secondary-button save-material-auxiliary" type="button">保存账本</button>
                <button class="danger-button delete-material-auxiliary" type="button">删除</button>
              </div>
            </article>
          `).join("") : '<div class="empty-list">暂无辅助账本</div>'}
        </div>
        <small class="material-inspector-footnote">辅助 ${auxiliaryRecords.length}</small>
        <div class="material-inspector-title material-inspector-title-spaced">关系</div>
        <div class="material-inspector-list">
          <article class="material-inspector-item material-relationship-create">
            <label class="material-inspector-field">
              <span>源人物</span>
              <select class="material-new-relationship-source">
                <option value="">选择人物</option>
                ${characters.map((character) => `<option value="${escapeText(character.id)}">${escapeText(character.canonical_name)}</option>`).join("")}
              </select>
            </label>
            <label class="material-inspector-field">
              <span>目标人物</span>
              <select class="material-new-relationship-target">
                <option value="">选择人物</option>
                ${characters.map((character) => `<option value="${escapeText(character.id)}">${escapeText(character.canonical_name)}</option>`).join("")}
              </select>
            </label>
            <label class="material-inspector-field">
              <span>关系</span>
              <input class="material-new-relationship-type" type="text" value="related" />
            </label>
            <label class="material-inspector-field">
              <span>状态</span>
              <select class="material-new-relationship-status">
                ${["active", "resolved", "disabled"].map((status) => `<option value="${status}">${status}</option>`).join("")}
              </select>
            </label>
            <label class="material-inspector-field">
              <span>强度</span>
              <input class="material-new-relationship-strength" type="number" min="0" max="1" step="0.01" value="0.50" />
            </label>
            <div class="material-inspector-actions"><button class="secondary-button create-material-relationship" type="button">新建关系</button></div>
          </article>
          ${relationships.length ? relationships.map((relationship) => {
            const relationshipEvents = relationship.events || [];
            return `
              <article class="material-inspector-item material-relationship-item" data-relationship-id="${escapeText(relationship.id)}">
                <b>${escapeText(relationship.source_name)} -> ${escapeText(relationship.target_name)}</b>
                <label class="material-inspector-field">
                  <span>关系</span>
                  <input class="material-relationship-type" type="text" value="${escapeText(relationship.relation_type || "related")}" />
                </label>
                <label class="material-inspector-field">
                  <span>状态</span>
                  <select class="material-relationship-status">
                    ${["active", "resolved", "disabled"].map((status) => `<option value="${status}" ${relationship.status === status ? "selected" : ""}>${status}</option>`).join("")}
                  </select>
                </label>
                <label class="material-inspector-field">
                  <span>强度</span>
                  <input class="material-relationship-strength" type="number" min="0" max="1" step="0.01" value="${Number(relationship.strength ?? 0).toFixed(2)}" />
                </label>
                <small>${escapeText(relationship.status || "active")} · 强度 ${Number(relationship.strength ?? 0).toFixed(2)} · 事件 ${relationshipEvents.length}</small>
                <div class="material-profile-list">
                  <small>关系事件 ${relationshipEvents.length}</small>
                  ${relationshipEvents.map((event) => `
                    <div class="material-profile-row" data-relationship-event-id="${escapeText(event.id)}">
                      <label class="material-inspector-field">
                        <span>事件类型</span>
                        <input class="material-relationship-event-type" type="text" value="${escapeText(event.event_type || "manual")}" />
                      </label>
                      <label class="material-inspector-field">
                        <span>事件描述</span>
                        <textarea class="material-relationship-event-description" rows="2">${escapeText(event.description || "")}</textarea>
                      </label>
                      <label class="material-inspector-field">
                        <span>强度变化</span>
                        <input class="material-relationship-event-strength-delta" type="number" min="-1" max="1" step="0.01" value="${Number(event.strength_delta ?? 0).toFixed(2)}" />
                      </label>
                      <div class="material-inspector-actions">
                        <button class="secondary-button save-material-relationship-event" type="button">保存事件</button>
                        <button class="danger-button delete-material-relationship-event" type="button">删除事件</button>
                      </div>
                    </div>
                  `).join("")}
                  <div class="material-profile-row material-relationship-event-create">
                    <label class="material-inspector-field">
                      <span>新增事件</span>
                      <input class="material-new-relationship-event-type" type="text" value="manual" />
                    </label>
                    <label class="material-inspector-field">
                      <span>事件描述</span>
                      <textarea class="material-new-relationship-event-description" rows="2"></textarea>
                    </label>
                    <div class="material-inspector-actions">
                      <button class="secondary-button create-material-relationship-event" type="button">新建事件</button>
                    </div>
                  </div>
                </div>
                <div class="material-inspector-actions">
                  <button class="secondary-button save-material-relationship" type="button">保存</button>
                  <button class="danger-button delete-material-relationship" type="button">删除</button>
                </div>
              </article>
            `;
          }).join("") : '<div class="empty-list">暂无关系边</div>'}
        </div>
        <small class="material-inspector-footnote">待确认 ${pendingCount}</small>
      </section>
    </div>
  `;
  elements.materialInspector.querySelector(".create-material-node")?.addEventListener("click", () => createMaterialTimelineNode());
  elements.materialInspector.querySelectorAll(".save-material-node").forEach((button) => {
    button.addEventListener("click", () => saveMaterialTimelineNode(button.closest(".material-inspector-item")));
  });
  elements.materialInspector.querySelectorAll(".delete-material-node").forEach((button) => {
    button.addEventListener("click", () => deleteMaterialTimelineNode(button.closest(".material-inspector-item")));
  });
  elements.materialInspector.querySelector(".create-material-event")?.addEventListener("click", () => createMaterialTimelineEvent());
  elements.materialInspector.querySelectorAll(".save-material-event").forEach((button) => {
    button.addEventListener("click", () => saveMaterialTimelineEvent(button.closest(".material-inspector-item")));
  });
  elements.materialInspector.querySelectorAll(".delete-material-event").forEach((button) => {
    button.addEventListener("click", () => deleteMaterialTimelineEvent(button.closest(".material-inspector-item")));
  });
  elements.materialInspector.querySelector(".create-material-character")?.addEventListener("click", () => createMaterialCharacter());
  elements.materialInspector.querySelectorAll(".save-material-character").forEach((button) => {
    button.addEventListener("click", () => saveMaterialCharacter(button.closest(".material-inspector-item")));
  });
  elements.materialInspector.querySelectorAll(".add-material-alias").forEach((button) => {
    button.addEventListener("click", () => addMaterialCharacterAlias(button.closest(".material-inspector-item")));
  });
  elements.materialInspector.querySelectorAll(".merge-material-character").forEach((button) => {
    button.addEventListener("click", () => mergeMaterialCharacter(button.closest(".material-inspector-item")));
  });
  elements.materialInspector.querySelectorAll(".split-material-character").forEach((button) => {
    button.addEventListener("click", () => splitMaterialCharacter(button.closest(".material-inspector-item")));
  });
  elements.materialInspector.querySelectorAll(".delete-material-character").forEach((button) => {
    button.addEventListener("click", () => deleteMaterialCharacter(button.closest(".material-inspector-item")));
  });
  elements.materialInspector.querySelectorAll(".create-material-profile").forEach((button) => {
    button.addEventListener("click", () => createMaterialCharacterProfile(button.closest(".material-inspector-item")));
  });
  elements.materialInspector.querySelectorAll(".save-material-profile").forEach((button) => {
    button.addEventListener("click", () => saveMaterialCharacterProfile(button.closest(".material-profile-row")));
  });
  elements.materialInspector.querySelectorAll(".delete-material-profile").forEach((button) => {
    button.addEventListener("click", () => deleteMaterialCharacterProfile(button.closest(".material-profile-row")));
  });
  elements.materialInspector.querySelectorAll(".create-material-character-fact").forEach((button) => {
    button.addEventListener("click", () => createMaterialCharacterFact(button.closest(".material-inspector-item")));
  });
  elements.materialInspector.querySelectorAll(".save-material-character-fact").forEach((button) => {
    button.addEventListener("click", () => saveMaterialCharacterFact(button.closest(".material-profile-row")));
  });
  elements.materialInspector.querySelectorAll(".delete-material-character-fact").forEach((button) => {
    button.addEventListener("click", () => deleteMaterialCharacterFact(button.closest(".material-profile-row")));
  });
  elements.materialInspector.querySelectorAll(".create-material-character-event").forEach((button) => {
    button.addEventListener("click", () => createMaterialCharacterEvent(button.closest(".material-inspector-item")));
  });
  elements.materialInspector.querySelectorAll(".save-material-character-event").forEach((button) => {
    button.addEventListener("click", () => saveMaterialCharacterEvent(button.closest(".material-profile-row")));
  });
  elements.materialInspector.querySelectorAll(".delete-material-character-event").forEach((button) => {
    button.addEventListener("click", () => deleteMaterialCharacterEvent(button.closest(".material-profile-row")));
  });
  elements.materialInspector.querySelector(".create-material-auxiliary")?.addEventListener("click", () => createMaterialAuxiliaryRecord());
  elements.materialInspector.querySelectorAll(".save-material-auxiliary").forEach((button) => {
    button.addEventListener("click", () => saveMaterialAuxiliaryRecord(button.closest(".material-inspector-item")));
  });
  elements.materialInspector.querySelectorAll(".delete-material-auxiliary").forEach((button) => {
    button.addEventListener("click", () => deleteMaterialAuxiliaryRecord(button.closest(".material-inspector-item")));
  });
  elements.materialInspector.querySelector(".create-material-relationship")?.addEventListener("click", () => createMaterialRelationship());
  elements.materialInspector.querySelectorAll(".save-material-relationship").forEach((button) => {
    button.addEventListener("click", () => saveMaterialRelationship(button.closest(".material-inspector-item")));
  });
  elements.materialInspector.querySelectorAll(".delete-material-relationship").forEach((button) => {
    button.addEventListener("click", () => deleteMaterialRelationship(button.closest(".material-inspector-item")));
  });
  elements.materialInspector.querySelectorAll(".create-material-relationship-event").forEach((button) => {
    button.addEventListener("click", () => createMaterialRelationshipEvent(button.closest(".material-inspector-item")));
  });
  elements.materialInspector.querySelectorAll(".save-material-relationship-event").forEach((button) => {
    button.addEventListener("click", () => saveMaterialRelationshipEvent(button.closest(".material-profile-row")));
  });
  elements.materialInspector.querySelectorAll(".delete-material-relationship-event").forEach((button) => {
    button.addEventListener("click", () => deleteMaterialRelationshipEvent(button.closest(".material-profile-row")));
  });
}

function materialReviewStatusLabel(status) {
  return {
    pending: "待确认",
    resolved: "已确认",
    rejected: "已忽略",
  }[status] || status || "未知";
}

function materialReviewTypeLabel(type) {
  return {
    relationship_entity_missing: "关系人物待匹配",
    character_entity_missing: "人物事件待匹配",
    material_import_conflict: "导入字段冲突",
    location_observation: "位置观察",
    ability_observation: "能力观察",
    object_observation: "物件观察",
    unresolved_observation: "悬念线索",
    local: "本地确认项",
  }[type] || type || "确认项";
}

function formatReviewValue(value) {
  if (Array.isArray(value)) return value.map(formatReviewValue).filter(Boolean).join("、");
  if (value && typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value ?? "").trim();
}

function formatMaterialReviewPayload(item) {
  const payload = item.payload || {};
  const keys = [
    "subject", "predicate", "object", "source", "target", "relation_type",
    "character", "name", "event_type", "description", "state", "value",
    "location", "ability", "item", "status", "evidence",
  ];
  const lines = keys
    .filter((key) => payload[key] !== undefined && formatReviewValue(payload[key]))
    .map((key) => `${key}: ${formatReviewValue(payload[key])}`);
  if (lines.length) return lines.join("\n");
  const fallback = Object.entries(payload).slice(0, 12)
    .map(([key, value]) => `${key}: ${formatReviewValue(value)}`)
    .filter((line) => line.split(":").slice(1).join(":").trim());
  return fallback.length ? fallback.join("\n") : "没有附加 payload";
}

function materialReviewSuggestedNames(item) {
  const payload = item.payload || {};
  const rawNames = item.review_type === "relationship_entity_missing"
    ? [payload.source || payload.subject, payload.target || payload.object]
    : [payload.character || payload.name || payload.subject];
  return [...new Set(rawNames.map((value) => String(value || "").trim()).filter(Boolean))];
}

function splitMaterialReviewNames(value) {
  return [...new Set(String(value || "")
    .split(/[、，,;\n|]+/)
    .map((name) => name.trim())
    .filter(Boolean))];
}

function materialReviewCanCreateEntities(item) {
  return ["relationship_entity_missing", "character_entity_missing"].includes(item.review_type);
}

function materialReviewCanApplyImportConflict(item) {
  return item.review_type === "material_import_conflict";
}

function materialReviewCanApplyAuxiliary(item) {
  return [
    "location_observation",
    "ability_observation",
    "object_observation",
    "unresolved_observation",
  ].includes(item.review_type);
}

function renderMaterialReviewItems() {
  if (!state.materialReviewsLoaded) {
    elements.materialReviewList.hidden = true;
    elements.materialReviewList.textContent = "";
    return;
  }
  const items = [...state.materialReviewItems].sort((left, right) => {
    if (left.status === right.status) return String(left.created_at || "").localeCompare(String(right.created_at || ""));
    return left.status === "pending" ? -1 : 1;
  });
  const pendingCount = items.filter((item) => item.status === "pending").length;
  elements.materialReviewList.hidden = false;
  elements.materialReviewList.innerHTML = `
    <div class="section-heading-row material-review-heading">
      <h3>人工确认队列</h3>
      <span class="muted-badge">待确认 ${pendingCount}</span>
    </div>
    ${items.length ? items.map((item) => {
      const suggestedNames = materialReviewSuggestedNames(item);
      const canCreateEntities = item.status === "pending" && materialReviewCanCreateEntities(item);
      const canApplyImportConflict = item.status === "pending" && materialReviewCanApplyImportConflict(item);
      const canApplyAuxiliary = item.status === "pending" && materialReviewCanApplyAuxiliary(item);
      return `
      <details class="workspace-card material-review-card" data-review-id="${escapeText(item.id)}" ${item.status === "pending" ? "open" : ""}>
        <summary>
          <span class="workspace-card-title">${escapeText(item.title || materialReviewTypeLabel(item.review_type))}</span>
          <span class="workspace-card-meta">${escapeText(materialReviewTypeLabel(item.review_type))}</span>
          <span class="status-pill is-${escapeText(item.status || "pending")}">${escapeText(materialReviewStatusLabel(item.status))}</span>
        </summary>
        <div class="workspace-card-body">
          <pre class="material-review-payload">${escapeText(formatMaterialReviewPayload(item))}</pre>
          ${canCreateEntities ? `
            <label class="material-review-resolution">
              <span>人物实体</span>
              <input class="material-review-names" type="text" value="${escapeText(suggestedNames.join("、"))}" />
            </label>
          ` : ""}
          ${item.resolution && Object.keys(item.resolution).length ? `<p class="settings-note">处理记录：${escapeText(formatReviewValue(item.resolution))}</p>` : ""}
          <div class="workspace-actions">
            ${canApplyImportConflict ? '<button class="secondary-button apply-material-import-conflict" type="button">应用包内值</button>' : ""}
            ${canApplyAuxiliary ? '<button class="secondary-button apply-material-auxiliary" type="button">写入资料</button>' : ""}
            <button class="secondary-button resolve-material-review" type="button" ${item.status !== "pending" ? "disabled" : ""}>${canCreateEntities ? "确认并写回" : "确认"}</button>
            <button class="danger-button reject-material-review" type="button" ${item.status !== "pending" ? "disabled" : ""}>忽略</button>
          </div>
        </div>
      </details>`;
    }).join("") : '<div class="empty-list">暂无待确认资料</div>'}
  `;
  elements.materialReviewList.querySelectorAll(".material-review-card").forEach((card) => {
    const itemId = card.dataset.reviewId;
    card.querySelector(".resolve-material-review")?.addEventListener("click", () => updateMaterialReviewItemStatus(itemId, "resolved", card));
    card.querySelector(".apply-material-import-conflict")?.addEventListener("click", () => updateMaterialReviewItemStatus(itemId, "resolved", card, {
      apply: "apply_import_conflict_incoming",
    }));
    card.querySelector(".apply-material-auxiliary")?.addEventListener("click", () => updateMaterialReviewItemStatus(itemId, "resolved", card, {
      apply: "apply_auxiliary_observation",
    }));
    card.querySelector(".reject-material-review")?.addEventListener("click", () => updateMaterialReviewItemStatus(itemId, "rejected"));
  });
}

async function refreshMaterialReviews() {
  if (!state.workspace) {
    showToast("请先选择一个 TXT", "error");
    return;
  }
  elements.refreshMaterialReviews.disabled = true;
  elements.refreshMaterialReviews.textContent = "正在读取…";
  try {
    state.materialReviewItems = await api.listMaterialReviewItems(state.workspace.id);
    state.materialReviewsLoaded = true;
    renderMaterialReviewItems();
    showToast("确认队列已刷新");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    elements.refreshMaterialReviews.disabled = false;
    elements.refreshMaterialReviews.textContent = "确认队列";
  }
}

async function inspectMaterialSystem() {
  if (!state.workspace) {
    showToast("请先选择一个 TXT", "error");
    return;
  }
  elements.inspectMaterialSystem.disabled = true;
  elements.inspectMaterialSystem.textContent = "正在读取…";
  try {
    state.materialOverview = await api.getMaterialOverview(state.workspace.id);
    state.materialInspectorLoaded = true;
    state.materialReviewItems = state.materialOverview.review_items || [];
    state.materialReviewsLoaded = true;
    renderMaterialInspector();
    renderMaterialReviewItems();
    showToast("实验资料视图已刷新");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    elements.inspectMaterialSystem.disabled = false;
    elements.inspectMaterialSystem.textContent = "实验资料视图";
  }
}

async function refreshMaterialOverviewAfterEdit(message) {
  if (!state.workspace) return;
  state.materialOverview = await api.getMaterialOverview(state.workspace.id);
  state.materialInspectorLoaded = true;
  state.materialReviewItems = state.materialOverview.review_items || [];
  state.materialReviewsLoaded = true;
  renderMaterialInspector();
  renderMaterialReviewItems();
  showToast(message);
}

async function createMaterialTimelineEvent() {
  if (!state.workspace) return;
  const panel = elements.materialInspector.querySelector(".material-event-create");
  const title = panel?.querySelector(".material-new-event-title")?.value.trim();
  if (!title) {
    showToast("请填写事件标题", "error");
    return;
  }
  const button = panel.querySelector(".create-material-event");
  button.disabled = true;
  try {
    await api.createMaterialTimelineEvent(state.workspace.id, {
      title,
      event_type: panel.querySelector(".material-new-event-type").value.trim() || "event",
      status: panel.querySelector(".material-new-event-status").value,
      description: panel.querySelector(".material-new-event-description").value.trim(),
    });
    await refreshMaterialOverviewAfterEdit("时间线事件已新建");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function saveMaterialTimelineEvent(card) {
  const eventId = card?.dataset.eventId;
  if (!eventId) return;
  const button = card.querySelector(".save-material-event");
  button.disabled = true;
  try {
    await api.updateMaterialTimelineEvent(eventId, {
      title: card.querySelector(".material-event-title").value.trim(),
      description: card.querySelector(".material-event-description").value.trim(),
      status: card.querySelector(".material-event-status").value,
    });
    await refreshMaterialOverviewAfterEdit("时间线事件已保存");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function deleteMaterialTimelineEvent(card) {
  const eventId = card?.dataset.eventId;
  if (!eventId) return;
  const title = card.querySelector(".material-event-title")?.value.trim() || "这个事件";
  if (!window.confirm(`删除“${title}”吗？`)) return;
  const button = card.querySelector(".delete-material-event");
  button.disabled = true;
  try {
    await api.deleteMaterialTimelineEvent(eventId);
    await refreshMaterialOverviewAfterEdit("时间线事件已删除");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function saveMaterialTimelineNode(card) {
  const nodeId = card?.dataset.nodeId;
  if (!nodeId) return;
  const button = card.querySelector(".save-material-node");
  button.disabled = true;
  try {
    await api.updateMaterialTimelineNode(nodeId, {
      title: card.querySelector(".material-node-title").value.trim(),
      summary: card.querySelector(".material-node-summary").value.trim(),
      enabled: card.querySelector(".material-node-enabled").checked,
    });
    await refreshMaterialOverviewAfterEdit("时间线节点已保存");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function createMaterialTimelineNode() {
  if (!state.workspace) return;
  const panel = elements.materialInspector.querySelector(".material-node-create");
  const title = panel?.querySelector(".material-new-node-title")?.value.trim();
  if (!title) {
    showToast("请填写节点标题", "error");
    return;
  }
  const button = panel.querySelector(".create-material-node");
  button.disabled = true;
  try {
    await api.createMaterialTimelineNode(state.workspace.id, {
      title,
      node_type: panel.querySelector(".material-new-node-type").value,
      summary: panel.querySelector(".material-new-node-summary").value.trim(),
    });
    await refreshMaterialOverviewAfterEdit("时间线节点已新建");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function deleteMaterialTimelineNode(card) {
  const nodeId = card?.dataset.nodeId;
  if (!nodeId) return;
  const title = card.querySelector(".material-node-title")?.value.trim() || "这个节点";
  if (!window.confirm(`删除“${title}”吗？子节点会挂到上一级。`)) return;
  const button = card.querySelector(".delete-material-node");
  button.disabled = true;
  try {
    await api.deleteMaterialTimelineNode(nodeId);
    await refreshMaterialOverviewAfterEdit("时间线节点已删除");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function createMaterialCharacter() {
  if (!state.workspace) return;
  const panel = elements.materialInspector.querySelector(".material-character-create");
  const name = panel?.querySelector(".material-new-character-name")?.value.trim();
  if (!name) {
    showToast("请填写人物名", "error");
    return;
  }
  const button = panel.querySelector(".create-material-character");
  button.disabled = true;
  try {
    await api.createMaterialCharacterEntity(state.workspace.id, {
      canonical_name: name,
      enabled: panel.querySelector(".material-new-character-enabled").checked,
      aliases: splitMaterialList(panel.querySelector(".material-new-character-aliases").value),
      profile: {
        identity: panel.querySelector(".material-new-character-identity").value.trim(),
      },
    });
    await refreshMaterialOverviewAfterEdit("人物已新建");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function saveMaterialCharacter(card) {
  const characterId = card?.dataset.characterId;
  if (!characterId) return;
  const button = card.querySelector(".save-material-character");
  button.disabled = true;
  try {
    await api.updateMaterialCharacterEntity(characterId, {
      canonical_name: card.querySelector(".material-character-name").value.trim(),
      enabled: card.querySelector(".material-character-enabled").checked,
      manually_confirmed: true,
      profile: {
        identity: card.querySelector(".material-character-identity").value.trim(),
      },
    });
    await refreshMaterialOverviewAfterEdit("人物实体已保存");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function deleteMaterialCharacter(card) {
  const characterId = card?.dataset.characterId;
  if (!characterId) return;
  const name = card.querySelector(".material-character-name")?.value.trim() || "这个人物";
  if (!window.confirm(`删除“${name}”吗？有关联关系时会被阻止。`)) return;
  const button = card.querySelector(".delete-material-character");
  button.disabled = true;
  try {
    await api.deleteMaterialCharacterEntity(characterId);
    await refreshMaterialOverviewAfterEdit("人物已删除");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function createMaterialCharacterProfile(card) {
  const characterId = card?.dataset.characterId;
  if (!characterId) return;
  const row = card.querySelector(".material-profile-create");
  const title = row.querySelector(".material-new-profile-title").value.trim() || "阶段档案";
  const button = row.querySelector(".create-material-profile");
  button.disabled = true;
  try {
    await api.createMaterialCharacterProfile(characterId, {
      title,
      identity: row.querySelector(".material-new-profile-identity").value.trim(),
    });
    await refreshMaterialOverviewAfterEdit("人物阶段档案已新建");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function saveMaterialCharacterProfile(row) {
  const profileId = row?.dataset.profileId;
  if (!profileId) return;
  const button = row.querySelector(".save-material-profile");
  button.disabled = true;
  try {
    await api.updateMaterialCharacterProfile(profileId, {
      title: row.querySelector(".material-profile-title").value.trim(),
      identity: row.querySelector(".material-profile-identity").value.trim(),
      enabled: row.querySelector(".material-profile-enabled").checked,
    });
    await refreshMaterialOverviewAfterEdit("人物阶段档案已保存");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function deleteMaterialCharacterProfile(row) {
  const profileId = row?.dataset.profileId;
  if (!profileId) return;
  const title = row.querySelector(".material-profile-title")?.value.trim() || "这个阶段档案";
  if (!window.confirm(`删除“${title}”吗？`)) return;
  const button = row.querySelector(".delete-material-profile");
  button.disabled = true;
  try {
    await api.deleteMaterialCharacterProfile(profileId);
    await refreshMaterialOverviewAfterEdit("人物阶段档案已删除");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function createMaterialCharacterFact(card) {
  const characterId = card?.dataset.characterId;
  if (!characterId) return;
  const row = card.querySelector(".material-character-fact-create");
  const field = row.querySelector(".material-new-character-fact-field").value.trim();
  const value = row.querySelector(".material-new-character-fact-value").value.trim();
  if (!field || !value) {
    showToast("请填写事实字段和内容", "error");
    return;
  }
  const button = row.querySelector(".create-material-character-fact");
  button.disabled = true;
  try {
    await api.createMaterialCharacterFact(characterId, { field, value });
    await refreshMaterialOverviewAfterEdit("人物事实已新建");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function saveMaterialCharacterFact(row) {
  const factId = row?.dataset.characterFactId;
  if (!factId) return;
  const button = row.querySelector(".save-material-character-fact");
  button.disabled = true;
  try {
    await api.updateMaterialCharacterFact(factId, {
      field: row.querySelector(".material-character-fact-field").value.trim(),
      value: row.querySelector(".material-character-fact-value").value.trim(),
      certainty: Number(row.querySelector(".material-character-fact-certainty").value || 1),
    });
    await refreshMaterialOverviewAfterEdit("人物事实已保存");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function deleteMaterialCharacterFact(row) {
  const factId = row?.dataset.characterFactId;
  if (!factId) return;
  const field = row.querySelector(".material-character-fact-field")?.value.trim() || "这条事实";
  if (!window.confirm(`删除“${field}”吗？`)) return;
  const button = row.querySelector(".delete-material-character-fact");
  button.disabled = true;
  try {
    await api.deleteMaterialCharacterFact(factId);
    await refreshMaterialOverviewAfterEdit("人物事实已删除");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function createMaterialCharacterEvent(card) {
  const characterId = card?.dataset.characterId;
  if (!characterId) return;
  const row = card.querySelector(".material-character-event-create");
  const value = row.querySelector(".material-new-character-event-value").value.trim();
  if (!value) {
    showToast("请填写经历内容", "error");
    return;
  }
  const button = row.querySelector(".create-material-character-event");
  button.disabled = true;
  try {
    await api.createMaterialCharacterEvent(characterId, {
      event_type: row.querySelector(".material-new-character-event-type").value.trim() || "event",
      value,
    });
    await refreshMaterialOverviewAfterEdit("人物经历已新建");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function saveMaterialCharacterEvent(row) {
  const eventId = row?.dataset.characterEventId;
  if (!eventId) return;
  const button = row.querySelector(".save-material-character-event");
  button.disabled = true;
  try {
    await api.updateMaterialCharacterEvent(eventId, {
      event_type: row.querySelector(".material-character-event-type").value.trim() || "event",
      value: row.querySelector(".material-character-event-value").value.trim(),
      sequence: Number(row.querySelector(".material-character-event-sequence").value || 0),
    });
    await refreshMaterialOverviewAfterEdit("人物经历已保存");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function deleteMaterialCharacterEvent(row) {
  const eventId = row?.dataset.characterEventId;
  if (!eventId) return;
  const title = row.querySelector(".material-character-event-value")?.value.trim() || "这条经历";
  if (!window.confirm(`删除“${title}”吗？`)) return;
  const button = row.querySelector(".delete-material-character-event");
  button.disabled = true;
  try {
    await api.deleteMaterialCharacterEvent(eventId);
    await refreshMaterialOverviewAfterEdit("人物经历已删除");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function addMaterialCharacterAlias(card) {
  const characterId = card?.dataset.characterId;
  if (!characterId) return;
  const aliasInput = card.querySelector(".material-character-alias");
  const alias = aliasInput.value.trim();
  if (!alias) {
    showToast("请先填写别名", "error");
    return;
  }
  const button = card.querySelector(".add-material-alias");
  button.disabled = true;
  try {
    await api.addMaterialCharacterAlias(characterId, { alias });
    await refreshMaterialOverviewAfterEdit("人物别名已保存");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function mergeMaterialCharacter(card) {
  const characterId = card?.dataset.characterId;
  if (!characterId) return;
  const targetId = card.querySelector(".material-character-merge-target").value;
  if (!targetId) {
    showToast("请先选择合并目标", "error");
    return;
  }
  const currentName = card.querySelector(".material-character-name").value.trim() || "当前人物";
  if (!window.confirm(`把“${currentName}”合并到所选人物吗？相关事件、档案、关系和别名会迁移到目标人物。`)) return;
  const button = card.querySelector(".merge-material-character");
  button.disabled = true;
  try {
    await api.mergeMaterialCharacterEntity(characterId, {
      target_character_id: targetId,
      keep_source_name_as_alias: true,
    });
    await refreshMaterialOverviewAfterEdit("人物已合并");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function splitMaterialCharacter(card) {
  const characterId = card?.dataset.characterId;
  if (!characterId) return;
  const name = card.querySelector(".material-character-split-name").value.trim();
  if (!name) {
    showToast("请填写拆分后的人物名", "error");
    return;
  }
  const aliases = splitMaterialList(card.querySelector(".material-character-split-aliases").value);
  const currentName = card.querySelector(".material-character-name").value.trim() || "当前人物";
  const aliasNote = aliases.length ? `，并移动别名：${aliases.join("、")}` : "";
  if (!window.confirm(`从“${currentName}”拆分出“${name}”${aliasNote}吗？`)) return;
  const button = card.querySelector(".split-material-character");
  button.disabled = true;
  try {
    await api.splitMaterialCharacterEntity(characterId, {
      canonical_name: name,
      aliases,
      copy_current_profile: true,
    });
    await refreshMaterialOverviewAfterEdit("人物已拆分");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function createMaterialAuxiliaryRecord() {
  if (!state.workspace) return;
  const panel = elements.materialInspector.querySelector(".material-auxiliary-create");
  const name = panel.querySelector(".material-new-auxiliary-name").value.trim();
  if (!name) {
    showToast("请填写辅助账本名称", "error");
    return;
  }
  const button = panel.querySelector(".create-material-auxiliary");
  button.disabled = true;
  try {
    await api.createMaterialAuxiliaryRecord(state.workspace.id, {
      record_type: panel.querySelector(".material-new-auxiliary-type").value,
      name,
      summary: panel.querySelector(".material-new-auxiliary-summary").value.trim(),
      status: panel.querySelector(".material-new-auxiliary-status").value,
    });
    await refreshMaterialOverviewAfterEdit("辅助账本已新增");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function saveMaterialAuxiliaryRecord(card) {
  const recordId = card?.dataset.auxiliaryId;
  if (!recordId) return;
  const name = card.querySelector(".material-auxiliary-name").value.trim();
  if (!name) {
    showToast("请填写辅助账本名称", "error");
    return;
  }
  const button = card.querySelector(".save-material-auxiliary");
  button.disabled = true;
  try {
    await api.updateMaterialAuxiliaryRecord(recordId, {
      record_type: card.querySelector(".material-auxiliary-type").value,
      name,
      summary: card.querySelector(".material-auxiliary-summary").value.trim(),
      status: card.querySelector(".material-auxiliary-status").value,
    });
    await refreshMaterialOverviewAfterEdit("辅助账本已保存");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function deleteMaterialAuxiliaryRecord(card) {
  const recordId = card?.dataset.auxiliaryId;
  if (!recordId) return;
  const name = card.querySelector(".material-auxiliary-name").value.trim() || "辅助账本";
  if (!window.confirm(`删除“${name}”吗？`)) return;
  try {
    await api.deleteMaterialAuxiliaryRecord(recordId);
    await refreshMaterialOverviewAfterEdit("辅助账本已删除");
  } catch (error) {
    showToast(errorMessage(error), "error");
  }
}

async function createMaterialRelationship() {
  if (!state.workspace) return;
  const panel = elements.materialInspector.querySelector(".material-relationship-create");
  const sourceId = panel?.querySelector(".material-new-relationship-source")?.value;
  const targetId = panel?.querySelector(".material-new-relationship-target")?.value;
  if (!sourceId || !targetId) {
    showToast("请先选择源人物和目标人物", "error");
    return;
  }
  if (sourceId === targetId) {
    showToast("关系不能指向同一人物", "error");
    return;
  }
  const button = panel.querySelector(".create-material-relationship");
  button.disabled = true;
  try {
    await api.createMaterialRelationship(state.workspace.id, {
      source_character_id: sourceId,
      target_character_id: targetId,
      relation_type: panel.querySelector(".material-new-relationship-type").value.trim() || "related",
      status: panel.querySelector(".material-new-relationship-status").value,
      strength: Number(panel.querySelector(".material-new-relationship-strength").value || 0.5),
    });
    await refreshMaterialOverviewAfterEdit("关系边已新建");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function saveMaterialRelationship(card) {
  const relationshipId = card?.dataset.relationshipId;
  if (!relationshipId) return;
  const button = card.querySelector(".save-material-relationship");
  button.disabled = true;
  try {
    await api.updateMaterialRelationship(relationshipId, {
      relation_type: card.querySelector(".material-relationship-type").value.trim(),
      status: card.querySelector(".material-relationship-status").value,
      strength: Number(card.querySelector(".material-relationship-strength").value || 0),
    });
    await refreshMaterialOverviewAfterEdit("关系边已保存");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function deleteMaterialRelationship(card) {
  const relationshipId = card?.dataset.relationshipId;
  if (!relationshipId) return;
  const title = card.querySelector("b")?.textContent.trim() || "这条关系";
  if (!window.confirm(`删除“${title}”吗？`)) return;
  const button = card.querySelector(".delete-material-relationship");
  button.disabled = true;
  try {
    await api.deleteMaterialRelationship(relationshipId);
    await refreshMaterialOverviewAfterEdit("关系边已删除");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function createMaterialRelationshipEvent(card) {
  const relationshipId = card?.dataset.relationshipId;
  if (!relationshipId) return;
  const row = card.querySelector(".material-relationship-event-create");
  const eventType = row.querySelector(".material-new-relationship-event-type").value.trim() || "manual";
  const description = row.querySelector(".material-new-relationship-event-description").value.trim();
  const button = row.querySelector(".create-material-relationship-event");
  button.disabled = true;
  try {
    await api.createMaterialRelationshipEvent(relationshipId, { event_type: eventType, description });
    await refreshMaterialOverviewAfterEdit("关系事件已新建");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function saveMaterialRelationshipEvent(row) {
  const eventId = row?.dataset.relationshipEventId;
  if (!eventId) return;
  const button = row.querySelector(".save-material-relationship-event");
  button.disabled = true;
  try {
    await api.updateMaterialRelationshipEvent(eventId, {
      event_type: row.querySelector(".material-relationship-event-type").value.trim() || "manual",
      description: row.querySelector(".material-relationship-event-description").value.trim(),
      strength_delta: Number(row.querySelector(".material-relationship-event-strength-delta").value || 0),
    });
    await refreshMaterialOverviewAfterEdit("关系事件已保存");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function deleteMaterialRelationshipEvent(row) {
  const eventId = row?.dataset.relationshipEventId;
  if (!eventId) return;
  const title = row.querySelector(".material-relationship-event-description")?.value.trim() || "这个关系事件";
  if (!window.confirm(`删除“${title}”吗？`)) return;
  const button = row.querySelector(".delete-material-relationship-event");
  button.disabled = true;
  try {
    await api.deleteMaterialRelationshipEvent(eventId);
    await refreshMaterialOverviewAfterEdit("关系事件已删除");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function updateMaterialReviewItemStatus(itemId, status, card = null, extraResolution = {}) {
  if (!itemId) return;
  const item = state.materialReviewItems.find((entry) => entry.id === itemId);
  const action = status === "resolved" ? api.resolveMaterialReviewItem : api.rejectMaterialReviewItem;
  const resolution = {
    source: "workspace_ui",
    action: status,
    handled_at: new Date().toISOString(),
    ...extraResolution,
  };
  if (status === "resolved" && item && materialReviewCanCreateEntities(item)) {
    const names = splitMaterialReviewNames(card?.querySelector(".material-review-names")?.value);
    if (!names.length) {
      showToast("请先填写要确认的人物名", "error");
      return;
    }
    resolution.apply = "create_missing_entities";
    resolution.names = names;
  }
  try {
    const updated = await action(itemId, resolution);
    if (state.materialInspectorLoaded && state.workspace) {
      state.materialOverview = await api.getMaterialOverview(state.workspace.id);
      state.materialReviewItems = state.materialOverview.review_items || [];
      renderMaterialInspector();
    } else {
      const index = state.materialReviewItems.findIndex((item) => item.id === itemId);
      if (index >= 0) {
        state.materialReviewItems.splice(index, 1, updated);
      } else {
        state.materialReviewItems.push(updated);
      }
    }
    state.materialReviewsLoaded = true;
    renderMaterialReviewItems();
    showToast(status === "resolved" ? "确认项已写回" : "确认项已忽略");
  } catch (error) {
    showToast(errorMessage(error), "error");
  }
}

async function exportMaterialPackage() {
  if (!state.workspace) {
    showToast("请先选择一个 TXT", "error");
    return;
  }
  elements.exportMaterialPackage.disabled = true;
  elements.exportMaterialPackage.textContent = "正在导出…";
  try {
    const { blob, filename } = await api.exportMaterialPackage(state.workspace.id);
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
    showToast("分析包已导出");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    elements.exportMaterialPackage.disabled = false;
    elements.exportMaterialPackage.textContent = "导出分析包";
  }
}

async function rebuildMaterialSystem() {
  if (!state.workspace) {
    showToast("请先选择一个 TXT", "error");
    return;
  }
  elements.rebuildMaterialSystem.disabled = true;
  elements.rebuildMaterialSystem.textContent = "正在重建…";
  try {
    const overview = await api.rebuildMaterialSystem(state.workspace.id);
    state.materialOverview = overview;
    state.materialInspectorLoaded = true;
    state.materialReviewItems = overview.review_items || [];
    state.materialReviewsLoaded = true;
    elements.materialPackageReport.textContent = formatMaterialOverview(overview);
    elements.materialPackageReport.hidden = false;
    renderMaterialInspector();
    renderMaterialReviewItems();
    showToast("实验资料已重建");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    elements.rebuildMaterialSystem.disabled = false;
    elements.rebuildMaterialSystem.textContent = "重建实验资料";
  }
}

async function previewMaterialPromptPlan() {
  if (!state.workspace) {
    showToast("请先选择一个 TXT", "error");
    return;
  }
  elements.previewMaterialPlan.disabled = true;
  elements.previewMaterialPlan.textContent = "正在计算…";
  try {
    const plan = await api.materialPromptPlan(state.workspace.id, {
      query_text: elements.composerInput.value.trim(),
      max_tokens: 8000,
    });
    elements.materialPackageReport.textContent = formatMaterialPromptPlan(plan);
    elements.materialPackageReport.hidden = false;
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    elements.previewMaterialPlan.disabled = false;
    elements.previewMaterialPlan.textContent = "提示词预算";
  }
}

async function editMaterialBudget() {
  if (!state.workspace) {
    showToast("请先选择一个 TXT", "error");
    return;
  }
  elements.editMaterialBudget.disabled = true;
  elements.editMaterialBudget.textContent = "正在读取…";
  try {
    state.materialBudgetProfile = await api.getMaterialBudgetProfile(state.workspace.id);
    state.materialBudgetLoaded = true;
    renderMaterialBudgetEditor();
    showToast("预算设置已载入");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    elements.editMaterialBudget.disabled = false;
    elements.editMaterialBudget.textContent = "预算设置";
  }
}

async function saveMaterialBudgetProfile() {
  if (!state.workspace || !state.materialBudgetProfile) return;
  const config = {};
  elements.materialBudgetEditor.querySelectorAll("[data-budget-key]").forEach((input) => {
    config[input.dataset.budgetKey] = Number(input.value || 0);
  });
  const button = elements.materialBudgetEditor.querySelector("#save-material-budget");
  button.disabled = true;
  try {
    state.materialBudgetProfile = await api.updateMaterialBudgetProfile(state.workspace.id, {
      name: state.materialBudgetProfile.name || "默认预算",
      config,
    });
    state.materialBudgetLoaded = true;
    renderMaterialBudgetEditor();
    showToast("提示词预算已保存");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    button.disabled = false;
  }
}

async function importMaterialPackageFile(file) {
  if (!file || !state.project || state.analysisRunning) return;
  const mode = elements.materialPackageMode.value || "create_document";
  const layers = selectedMaterialLayers();
  const scope = selectedMaterialScope();
  if (!layers.length) {
    showToast("请至少选择一个资料层", "error");
    elements.materialPackageFile.value = "";
    return;
  }
  if (scope?.error) {
    showToast(scope.error, "error");
    elements.materialPackageFile.value = "";
    return;
  }
  if (scope && mode !== "merge") {
    showToast("章节范围过滤仅支持合并到当前 TXT", "error");
    elements.materialPackageFile.value = "";
    return;
  }
  const targetDocumentId = mode === "create_document" ? null : state.workspace?.id;
  if (mode !== "create_document" && !targetDocumentId) {
    showToast("请先选择一个目标 TXT", "error");
    elements.materialPackageFile.value = "";
    return;
  }
  elements.importMaterialPackage.disabled = true;
  elements.importMaterialPackage.textContent = "正在校验…";
  try {
    const report = await api.validateMaterialPackage(file, targetDocumentId, scope);
    elements.materialPackageReport.textContent = formatMaterialPackageReport(report);
    elements.materialPackageReport.hidden = false;
    const canImport = mode === "create_document" ? report.can_create_new_document : report.can_import;
    if (!canImport) {
      showToast(mode === "create_document" ? "分析包暂不能作为纯新文件导入" : "分析包与当前 TXT 不匹配", "error");
      return;
    }
    const actionLabel = {
      create_document: "创建为新的本地 TXT",
      merge: "合并到当前 TXT 的实验资料",
      replace_material: "替换当前 TXT 的实验资料",
    }[mode];
    const layerText = layers.map(materialLayerLabel).join("、");
    const scopeText = scope ? `\n章节范围：${materialScopeLabel(scope)}` : "";
    const ok = window.confirm(`${formatMaterialPackageReport(report)}\n\n${actionLabel}？\n导入资料层：${layerText}${scopeText}`);
    if (!ok) return;
    elements.importMaterialPackage.textContent = "正在导入…";
    const imported = await api.importMaterialPackage(state.project.id, file, {
      mode,
      documentId: targetDocumentId,
      layers,
      scope,
    });
    await selectDocument(imported.document_id);
    await loadProject(imported.document_id);
    elements.materialPackageReport.textContent = formatMaterialPackageReport(imported.report);
    elements.materialPackageReport.hidden = false;
    showToast(mode === "create_document" ? "分析包已导入为新 TXT" : "分析包已导入当前 TXT");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    elements.importMaterialPackage.disabled = false;
    elements.importMaterialPackage.textContent = "导入分析包";
    elements.materialPackageFile.value = "";
  }
}

async function clearProjectLibrary() {
  if (!state.project || !state.project.documents.length) return;
  if (!window.confirm("清空全部导入文件、章节、总结和人物卡吗？此操作无法撤销。")) return;
  try {
    state.project = await api.clearProjectLibrary(state.project.id);
    state.workspace = null;
    if (state.conversation) {
      state.conversation = await api.updateConversation(state.conversation.id, { document_id: null });
    }
    renderProject();
    showToast("小说资料库已清空");
    scheduleContextUsage();
  } catch (error) {
    showToast(errorMessage(error), "error");
  }
}

function setAnalysisProgress(text, completed = 0, total = 0) {
  elements.analysisProgress.hidden = false;
  elements.analysisProgressText.textContent = text;
  elements.analysisProgressCount.textContent = total ? `${completed} / ${total}` : "";
  elements.analysisProgressBar.style.width = `${total ? Math.round((completed / total) * 100) : 8}%`;
}

function setDetailedAnalysisProgress(text, ratio, countText = "") {
  elements.analysisProgress.hidden = false;
  elements.analysisProgressText.textContent = text;
  elements.analysisProgressCount.textContent = countText;
  elements.analysisProgressBar.style.width = `${Math.max(2, Math.min(100, Math.round(ratio * 100)))}%`;
}

function analysisPhaseLabel(phase) {
  return {
    chapter: "章节分片总结",
    facts: "提取结构化事实",
    chapter_merge: "合并本章摘要",
    increment: "增量章节总结",
    project_summary: "合并全书总览",
    characters: "整理人物卡",
  }[phase] || "资料整理";
}

async function runProjectSummary(chapterIds = null, regenerate = false, resumeJobId = null) {
  if (!state.project || !state.workspace || state.analysisRunning || state.generating || state.outlineGenerating) return;
  if (state.runtime?.status !== "ready") {
    showToast("请先启动本地模型", "error");
    return;
  }
  const documentId = state.workspace.id;
  let startPosition = Number(elements.analysisStart.value || state.workspace.chapters[0]?.position || 1);
  let endPosition = Number(elements.analysisEnd.value || state.workspace.chapters.at(-1)?.position || startPosition);
  if (chapterIds?.length) {
    const selected = state.workspace.chapters.filter((chapter) => chapterIds.includes(chapter.id));
    if (selected.length) {
      startPosition = Math.min(...selected.map((chapter) => chapter.position));
      endPosition = Math.max(...selected.map((chapter) => chapter.position));
    }
  }
  if (!resumeJobId && startPosition > endPosition) {
    showToast("起始章节不能晚于结束章节", "error");
    return;
  }
  state.analysisRunning = true;
  elements.summarizeProject.textContent = "停止总结";
  elements.documentSelect.disabled = true;
  updateSendButton();
  let total = 0;
  try {
    await stream(`/api/projects/${state.project.id}/summarize`, {
      document_id: documentId,
      chapter_ids: chapterIds,
      start_position: startPosition,
      end_position: endPosition,
      resume_job_id: resumeJobId,
      regenerate,
      max_tokens: Math.max(8192, currentGenerationSettings().max_tokens),
    }, {
      onEvent: async (event, data) => {
        if (event === "job_started") {
          total = data.total;
          setAnalysisProgress(total ? "正在逐章分析并保存断点…" : "正在整理全书资料…", 0, total);
        } else if (event === "chapter_started") {
          setAnalysisProgress(`正在总结：${data.title}`, data.index - 1, data.total);
        } else if (event === "chapter_completed") {
          const index = state.workspace?.id === documentId
            ? state.workspace.chapters.findIndex((item) => item.id === data.chapter.id)
            : -1;
          if (index >= 0) state.workspace.chapters[index] = data.chapter;
          setAnalysisProgress(`已完成：${data.chapter.title}`, data.index, data.total);
          renderProject();
        } else if (event === "analysis_progress") {
          const finished = data.stage.endsWith("completed");
          const itemFraction = data.total
            ? (finished || data.stage === "chunk_resumed" ? data.index : Math.max(0, data.index - 1)) / data.total
            : 0;
          if (["chapter", "facts", "chapter_merge"].includes(data.phase)) {
            const chapterFraction = ((data.chapter_index - 1) + itemFraction) / Math.max(1, data.chapter_total);
            const stageText = data.phase === "chapter_merge"
              ? "合并本章摘要"
              : data.phase === "facts"
                ? `提取事实 ${data.index}/${data.total}`
                : data.stage === "chunk_resumed"
                  ? `读取断点 ${data.index}/${data.total}`
                  : `分析分片 ${data.index}/${data.total}`;
            setDetailedAnalysisProgress(
              `${data.title} · ${stageText}`,
              chapterFraction * 0.85,
              `第 ${data.chapter_index}/${data.chapter_total} 章`,
            );
          } else if (data.phase === "project_summary") {
            setDetailedAnalysisProgress(
              `${analysisPhaseLabel(data.phase)} · ${data.index}/${data.total}`,
              0.85 + itemFraction * 0.1,
              "总览阶段",
            );
          } else if (data.phase === "characters") {
            setDetailedAnalysisProgress(
              `${analysisPhaseLabel(data.phase)} · ${data.index}/${data.total}`,
              0.95 + itemFraction * 0.05,
              "人物阶段",
            );
          }
        } else if (event === "analysis_heartbeat") {
          const title = data.title ? `${data.title} · ` : "";
          elements.analysisProgressText.textContent = `${title}${analysisPhaseLabel(data.phase)}仍在运行（${data.elapsed_seconds} 秒）`;
        } else if (event === "chapter_error") {
          setAnalysisProgress(`本章失败：${data.message}`, data.index, data.total);
          showToast(data.message, "error");
        } else if (event === "project_summary_started") {
          setAnalysisProgress("正在合并前文总览…", total, total);
        } else if (event === "project_summary_completed") {
          if (state.workspace?.id === documentId) state.workspace.global_summary = data.global_summary;
          elements.globalSummary.value = data.global_summary;
          setAnalysisProgress("正在拆解核心人物…", total, total);
        } else if (event === "characters_completed") {
          if (state.workspace?.id === documentId) state.workspace.characters = data.characters;
          renderProject();
        } else if (["done", "cancelled"].includes(event)) {
          state.workspace = data.workspace;
          renderProject();
          setAnalysisProgress(event === "done" ? "摘要、人物卡与结构化事实已完成" : "已暂停，断点已保存", total, total);
        } else if (event === "error") {
          showToast(data.message || "总结失败", "error");
        }
      },
    });
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    state.analysisRunning = false;
    elements.summarizeProject.textContent = "总结全部章节";
    try {
      await loadProject(documentId);
    } catch {
      // Keep the last locally rendered state when the refresh itself fails.
    }
    elements.documentSelect.disabled = false;
    updateSendButton();
    scheduleContextUsage();
  }
}

async function toggleProjectSummary() {
  if (state.analysisRunning) {
    elements.summarizeProject.textContent = "正在停止…";
    try {
      await api.stop();
    } catch (error) {
      showToast(errorMessage(error), "error");
    }
    return;
  }
  await runProjectSummary();
}

async function importTxtFile(file) {
  if (!file || !state.project || state.analysisRunning) return;
  elements.importTxt.disabled = true;
  elements.importTxt.textContent = "正在导入…";
  try {
    const imported = await api.importTxt(state.project.id, file);
    await selectDocument(imported.document.id);
    await loadProject(imported.document.id);
    showToast(`已导入 ${imported.chapters.length} 章；请选择起止章节后开始总结`);
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    elements.importTxt.disabled = false;
    elements.importTxt.textContent = "选择 TXT 文件";
    elements.txtFile.value = "";
  }
}

async function openIncrement(candidate) {
  if (!candidate?.content || candidate.status !== "completed") return;
  if (state.generating || state.analysisRunning || state.outlineGenerating) {
    showToast("请先等待当前任务结束", "error");
    return;
  }
  try {
    if (!state.project) await loadProject();
    if (!state.workspace) {
      showToast("请先在小说资料库导入并选择一个 TXT", "error");
      return;
    }
    state.incrementCandidate = candidate;
    elements.incrementTarget.replaceChildren();
    for (const chapter of state.workspace.chapters) {
      const option = document.createElement("option");
      option.value = chapter.id;
      option.textContent = `追加到：${chapter.title}`;
      elements.incrementTarget.append(option);
    }
    const newOption = document.createElement("option");
    newOption.value = "__new__";
    newOption.textContent = "新建章节";
    elements.incrementTarget.append(newOption);
    elements.incrementTarget.value = state.workspace.chapters.at(-1)?.id || "__new__";
    elements.incrementTitleField.hidden = elements.incrementTarget.value !== "__new__";
    elements.incrementChapterTitle.value = "";
    elements.incrementContent.value = candidate.content;
    elements.incrementSummarizeNow.checked = false;
    elements.confirmIncrement.textContent = "仅加入章节";
    elements.incrementStatus.textContent = `目标小说：《${state.workspace.filename}》。默认只保存正文，稍后可批量总结所有待处理章节。`;
    elements.incrementBackdrop.hidden = false;
    elements.incrementDialog.hidden = false;
    syncBodyLock();
  } catch (error) {
    showToast(errorMessage(error), "error");
  }
}

async function confirmIncrement() {
  const candidate = state.incrementCandidate;
  if (!candidate || state.incrementRunning) return;
  const isNew = elements.incrementTarget.value === "__new__";
  const summarizeNow = elements.incrementSummarizeNow.checked;
  const documentId = state.workspace?.id;
  if (!documentId) {
    showToast("请先选择目标 TXT 小说", "error");
    return;
  }
  const title = elements.incrementChapterTitle.value.trim();
  if (isNew && !title) {
    showToast("请填写新章节标题", "error");
    elements.incrementChapterTitle.focus();
    return;
  }
  state.incrementRunning = true;
  state.analysisRunning = summarizeNow;
  elements.confirmIncrement.disabled = true;
  elements.confirmIncrement.textContent = summarizeNow ? "正在加入并总结…" : "正在加入…";
  elements.incrementStatus.textContent = "正文即将写入章节…";
  updateSendButton();
  let completed = false;
  try {
    const payload = {
      content: candidate.content,
      chapter_id: isNew ? null : elements.incrementTarget.value,
      document_id: documentId,
      title: isNew ? title : null,
      source_candidate_id: candidate.id,
      max_tokens: Math.max(8192, currentGenerationSettings().max_tokens),
      summarize_now: summarizeNow,
    };
    if (!summarizeNow) {
      const data = await request(`/api/projects/${state.project.id}/append`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      state.workspace = data.workspace;
      state.appendedCandidateIds.add(candidate.id);
      completed = true;
      elements.incrementStatus.textContent = "正文已保存为待总结内容";
    } else {
      await stream(`/api/projects/${state.project.id}/append`, payload, {
      onEvent: async (event, data) => {
        if (event === "append_saved") {
          elements.incrementStatus.textContent = `正文已写入《${data.chapter.title}》，正在增量总结…`;
        } else if (event === "analysis_progress") {
          const action = data.stage.startsWith("merge")
            ? "正在合并结果"
            : `${data.stage.endsWith("completed") ? "已完成" : "正在处理"} ${data.index}/${data.total}`;
          const subphase = data.stage.startsWith("character")
            ? "人物信息"
            : data.stage.startsWith("summary") ? "情节摘要" : analysisPhaseLabel(data.phase);
          elements.incrementStatus.textContent = `${subphase} · ${action}`;
        } else if (event === "analysis_heartbeat") {
          elements.incrementStatus.textContent = `${analysisPhaseLabel(data.phase)}仍在运行（${data.elapsed_seconds} 秒），正文已经安全保存。`;
        } else if (event === "chapter_completed") {
          elements.incrementStatus.textContent = "章节摘要已更新，正在合并全书总览…";
        } else if (event === "project_summary_completed") {
          elements.incrementStatus.textContent = "全书总览已更新，正在整理人物卡…";
        } else if (event === "characters_completed") {
          elements.incrementStatus.textContent = `人物卡已更新（${data.characters.length} 人）…`;
        } else if (event === "done") {
          state.workspace = data.workspace;
          completed = true;
          state.appendedCandidateIds.add(candidate.id);
          elements.incrementStatus.textContent = "增量资料更新完成";
        } else if (event === "cancelled") {
          state.workspace = data.workspace;
          elements.incrementStatus.textContent = "正文已保存；总结已停止，可在资料库重新总结本章。";
        } else if (event === "error") {
          if (data.workspace) state.workspace = data.workspace;
          elements.incrementStatus.textContent = data.message || "增量总结失败";
          showToast(data.message || "增量总结失败", "error");
        }
      },
      });
    }
  } catch (error) {
    showToast(errorMessage(error), "error");
    elements.incrementStatus.textContent = errorMessage(error);
  } finally {
    state.incrementRunning = false;
    state.analysisRunning = false;
    elements.confirmIncrement.disabled = false;
    elements.confirmIncrement.textContent = elements.incrementSummarizeNow.checked ? "加入并立即总结" : "仅加入章节";
    try {
      await loadProject(documentId);
    } catch {
      // The content is already durable even if this refresh fails.
    }
    renderMessages({ keepScroll: false });
    updateSendButton();
    scheduleContextUsage();
    if (completed) {
      closeIncrement();
      showToast(summarizeNow ? "正文与小说资料已更新" : "正文已加入；可稍后批量总结");
    }
  }
}

function outlineCandidates() {
  return [
    ...(state.outline?.candidates || []).map((candidate) => ({ ...candidate, persisted: true })),
    ...state.outlineDrafts,
  ].filter((candidate) => candidate.status !== "failed");
}

function viewedOutlineCandidate() {
  const candidates = outlineCandidates();
  return candidates.find((item) => item.id === state.outlineViewedCandidateId)
    || candidates.find((item) => item.id === state.outline?.selected_candidate_id)
    || candidates.at(-1)
    || null;
}

function renderOutline({ preserveInstruction = false } = {}) {
  const outline = state.outline;
  if (!outline) return;
  if (!preserveInstruction) elements.outlineInstruction.value = outline.instruction || "请规划紧接当前进度的下一章。";
  const candidates = outlineCandidates();
  let candidate = viewedOutlineCandidate();
  if (candidate) state.outlineViewedCandidateId = candidate.id;
  const index = candidate ? candidates.findIndex((item) => item.id === candidate.id) : -1;
  elements.outlineCounter.textContent = `${index >= 0 ? index + 1 : 0} / ${candidates.length}`;
  elements.outlinePrev.disabled = index <= 0 || state.outlineGenerating;
  elements.outlineNext.disabled = index < 0 || index >= candidates.length - 1 || state.outlineGenerating;
  elements.outlineContent.value = candidate ? candidate.edited_content || candidate.content || "" : "";
  elements.outlineContent.disabled = !candidate || state.outlineGenerating;
  elements.outlineState.textContent = !candidate
    ? "尚未生成"
    : candidate.status === "streaming"
      ? "正在抽取"
      : candidate.persisted === false
        ? "临时 · 未保存"
      : candidate.id === outline.selected_candidate_id
        ? "已选用"
        : candidate.status === "cancelled" ? "未完成" : "待选用";
  elements.saveOutline.disabled = !candidate || state.outlineGenerating;
  elements.selectOutline.disabled = !candidate || candidate.status !== "completed" || candidate.id === outline.selected_candidate_id || state.outlineGenerating;
  elements.selectOutline.textContent = candidate?.id === outline.selected_candidate_id ? "已选用此版本" : "选用此版本";
  elements.outlineEnabled.checked = outline.enabled;
  elements.outlineEnabled.disabled = !outline.selected_candidate_id || state.outlineGenerating;
  elements.outlineTokenNote.textContent = `本次大纲最多输出 ${currentGenerationSettings().max_tokens} tokens，跟随当前对话的创作设置。未保存草稿不会写入数据库。`;
  elements.deleteOutlineCandidate.disabled = !candidate || state.outlineGenerating;
  elements.clearOutline.disabled = (!candidates.length && !state.previousOutlineId) || state.outlineGenerating;
  elements.newOutline.disabled = state.outlineGenerating;
  elements.rerollOutline.disabled = state.outlineGenerating;
  elements.rerollOutline.textContent = state.outlineGenerating ? "正在抽取…" : "再抽一版";
}

async function loadOutline() {
  if (!state.conversation) return;
  const stored = await api.getOutline(state.conversation.id);
  state.outline = stored || {
    id: null,
    conversation_id: state.conversation.id,
    instruction: "请规划紧接当前进度的下一章。",
    selected_candidate_id: null,
    enabled: false,
    candidates: [],
  };
  state.outlineDrafts = [];
  state.previousOutlineId = null;
  state.outlineViewedCandidateId = state.outline.selected_candidate_id || state.outline.candidates.at(-1)?.id || null;
  renderOutline();
}

async function openOutline() {
  if (!state.conversation) return;
  closeSettings();
  closeProject();
  elements.outlineBackdrop.hidden = false;
  elements.outlinePanel.hidden = false;
  syncBodyLock();
  closeMobileSidebar();
  try {
    if (!state.outline || state.outline.conversation_id !== state.conversation.id) {
      await loadOutline();
    } else {
      renderOutline();
    }
  } catch (error) {
    showToast(errorMessage(error), "error");
  }
}

function switchOutlineCandidate(direction) {
  const candidates = outlineCandidates();
  const candidate = viewedOutlineCandidate();
  const index = candidates.findIndex((item) => item.id === candidate?.id);
  const next = candidates[index + direction];
  if (!next) return;
  state.outlineViewedCandidateId = next.id;
  renderOutline({ preserveInstruction: true });
}

async function generateOutline(newGroup = false) {
  if (!state.conversation || state.outlineGenerating || state.generating || state.analysisRunning) return;
  if (state.runtime?.status !== "ready") {
    showToast("请先启动本地模型", "error");
    return;
  }
  const instruction = elements.outlineInstruction.value.trim();
  if (!instruction) {
    showToast("请先写下本章要求", "error");
    return;
  }
  state.outlineGenerating = true;
  state.outlineStreamController = new AbortController();
  renderOutline({ preserveInstruction: true });
  updateSendButton();
  let activeCandidateId = null;
  const path = `/api/conversations/${state.conversation.id}/outline/generate${newGroup ? "?new_group=true" : ""}`;
  if (newGroup) {
    state.previousOutlineId = state.outline?.id || state.previousOutlineId;
    state.outlineDrafts = [];
    state.outline = {
      id: null,
      conversation_id: state.conversation.id,
      instruction,
      selected_candidate_id: null,
      enabled: false,
      candidates: [],
    };
  }
  try {
    await stream(path, { instruction, settings: currentGenerationSettings() }, {
      signal: state.outlineStreamController.signal,
      onEvent: async (event, data) => {
        if (event === "outline_preview_created") {
          activeCandidateId = data.candidate.id;
          state.outlineDrafts.push(data.candidate);
          state.outlineViewedCandidateId = activeCandidateId;
          state.contextStats = {
            input_tokens: data.prompt_tokens,
            context_size: data.context_size,
            reserved_output_tokens: data.max_tokens + 384,
          };
          renderContextUsage();
          renderOutline({ preserveInstruction: true });
        } else if (event === "content_delta") {
          const candidate = state.outlineDrafts.find((item) => item.id === activeCandidateId);
          if (candidate) candidate.content += data.text;
          if (state.outlineViewedCandidateId === activeCandidateId) elements.outlineContent.value += data.text;
        } else if (["done", "cancelled"].includes(event)) {
          const index = state.outlineDrafts.findIndex((item) => item.id === activeCandidateId);
          if (index >= 0) state.outlineDrafts[index] = data.candidate;
          state.outlineViewedCandidateId = data.candidate.id;
          renderOutline({ preserveInstruction: true });
        } else if (event === "error") {
          renderOutline({ preserveInstruction: true });
          showToast(data.message || "大纲生成失败", "error");
        }
      },
    });
  } catch (error) {
    if (error?.name !== "AbortError") showToast(errorMessage(error), "error");
  } finally {
    state.outlineGenerating = false;
    state.outlineStreamController = null;
    renderOutline({ preserveInstruction: true });
    updateSendButton();
    scheduleContextUsage();
  }
}

async function saveOutlineCandidate() {
  const candidate = viewedOutlineCandidate();
  if (!candidate) return;
  try {
    if (candidate.persisted === false) {
      state.outline = await api.saveOutlineCandidate(state.conversation.id, {
        outline_id: state.outline.id,
        instruction: elements.outlineInstruction.value.trim(),
        content: elements.outlineContent.value,
        select: false,
        settings: currentGenerationSettings(),
      });
      state.outlineDrafts = state.outlineDrafts.filter((item) => item.id !== candidate.id);
      state.previousOutlineId = null;
      state.outlineViewedCandidateId = state.outline.candidates.at(-1)?.id || null;
    } else {
      state.outline = await api.editOutlineCandidate(candidate.id, elements.outlineContent.value);
      state.outlineViewedCandidateId = candidate.id;
    }
    renderOutline({ preserveInstruction: true });
    showToast("大纲修改已保存");
    scheduleContextUsage();
  } catch (error) {
    showToast(errorMessage(error), "error");
  }
}

async function selectOutlineCandidate() {
  const candidate = viewedOutlineCandidate();
  if (!candidate || !state.outline) return;
  try {
    if (candidate.persisted === false) {
      state.outline = await api.saveOutlineCandidate(state.conversation.id, {
        outline_id: state.outline.id,
        instruction: elements.outlineInstruction.value.trim(),
        content: elements.outlineContent.value,
        select: true,
        settings: currentGenerationSettings(),
      });
      state.outlineDrafts = state.outlineDrafts.filter((item) => item.id !== candidate.id);
      state.previousOutlineId = null;
      state.outlineViewedCandidateId = state.outline.selected_candidate_id;
    } else {
      if (elements.outlineContent.value !== (candidate.edited_content || candidate.content || "")) {
        state.outline = await api.editOutlineCandidate(candidate.id, elements.outlineContent.value);
      }
      state.outline = await api.selectOutline(state.outline.id, candidate.id);
      state.outlineViewedCandidateId = candidate.id;
    }
    renderOutline({ preserveInstruction: true });
    showToast("已选用这个大纲版本");
    scheduleContextUsage();
  } catch (error) {
    showToast(errorMessage(error), "error");
  }
}

async function deleteCurrentOutlineCandidate() {
  const candidate = viewedOutlineCandidate();
  if (!candidate || state.outlineGenerating) return;
  if (!window.confirm("删除当前这版大纲吗？此操作无法撤销。")) return;
  try {
    if (candidate.persisted === false) {
      state.outlineDrafts = state.outlineDrafts.filter((item) => item.id !== candidate.id);
    } else {
      state.outline = await api.deleteOutlineCandidate(candidate.id);
    }
    const remaining = outlineCandidates();
    state.outlineViewedCandidateId = remaining.at(-1)?.id || null;
    renderOutline({ preserveInstruction: true });
    scheduleContextUsage();
  } catch (error) {
    showToast(errorMessage(error), "error");
  }
}

async function clearOutlineCandidates() {
  if ((!outlineCandidates().length && !state.previousOutlineId) || state.outlineGenerating) return;
  if (!window.confirm("清空当前对话的全部大纲版本吗？已保存和临时版本都会删除。")) return;
  try {
    state.outlineDrafts = [];
    const storedOutlineId = state.outline?.id || state.previousOutlineId;
    if (storedOutlineId) await api.deleteOutline(storedOutlineId);
    state.outline = {
      id: null,
      conversation_id: state.conversation.id,
      instruction: elements.outlineInstruction.value.trim() || "请规划紧接当前进度的下一章。",
      selected_candidate_id: null,
      enabled: false,
      candidates: [],
    };
    state.outlineViewedCandidateId = null;
    state.previousOutlineId = null;
    renderOutline({ preserveInstruction: true });
    showToast("大纲已清空");
    scheduleContextUsage();
  } catch (error) {
    showToast(errorMessage(error), "error");
  }
}

async function toggleOutlineEnabled() {
  if (!state.outline) return;
  const enabled = elements.outlineEnabled.checked;
  try {
    state.outline = await api.updateOutline(state.outline.id, enabled);
    renderOutline({ preserveInstruction: true });
    showToast(enabled ? "正文生成时会使用已选大纲" : "正文生成时不再使用大纲");
    scheduleContextUsage();
  } catch (error) {
    elements.outlineEnabled.checked = !enabled;
    showToast(errorMessage(error), "error");
  }
}

function groupLabel(isoDate) {
  const date = new Date(isoDate);
  const now = new Date();
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const days = Math.round((startToday - startDate) / 86400000);
  if (days <= 0) return "今天";
  if (days === 1) return "昨天";
  if (days < 7) return "过去 7 天";
  return "更早";
}

function renderConversationList() {
  const fragment = document.createDocumentFragment();
  let lastGroup = "";
  for (const conversation of state.conversations) {
    const group = groupLabel(conversation.updated_at);
    if (group !== lastGroup) {
      const label = document.createElement("div");
      label.className = "conversation-group-label";
      label.textContent = group;
      fragment.append(label);
      lastGroup = group;
    }
    const item = document.createElement("div");
    item.className = `conversation-item ${conversation.id === state.conversation?.id ? "is-active" : ""}`;
    item.dataset.id = conversation.id;
    item.innerHTML = `<span class="conversation-item-title"></span><button class="conversation-delete" type="button" aria-label="删除对话">${icons.trash}</button>`;
    item.querySelector(".conversation-item-title").textContent = conversation.title;
    item.addEventListener("click", (event) => {
      if (event.target.closest(".conversation-delete")) return;
      loadConversation(conversation.id);
      closeMobileSidebar();
    });
    item.querySelector(".conversation-delete").addEventListener("click", () => deleteConversation(conversation));
    fragment.append(item);
  }
  elements.conversationList.replaceChildren(fragment);
}

function getViewedCandidate(exchange) {
  const candidateId = state.viewedCandidates.get(exchange.id) || exchange.selected_candidate_id;
  return exchange.candidates.find((candidate) => candidate.id === candidateId) || exchange.candidates.at(-1) || null;
}

function visibleCandidateIndex(exchange, candidate) {
  const candidates = exchange.candidates.filter((item) => item.status !== "failed");
  const index = candidates.findIndex((item) => item.id === candidate?.id);
  if (index >= 0) return { candidates, index };
  return { candidates: exchange.candidates, index: Math.max(0, exchange.candidates.indexOf(candidate)) };
}

function generationMeta(candidate) {
  const parts = [];
  if (candidate.completion_tokens) parts.push(`${candidate.completion_tokens} tokens`);
  if (candidate.duration_ms) parts.push(`${(candidate.duration_ms / 1000).toFixed(1)} 秒`);
  return parts.join(" · ");
}

function renderAssistantContent(candidate) {
  if (!candidate) return '<p class="empty-generation">尚未生成内容</p>';
  if (candidate.status === "failed") {
    return `<div class="generation-error">${escapeText(candidate.error_message || "本次生成失败，可以重新尝试")}</div>`;
  }
  const reasoning = candidate.reasoning_content
    ? `<details class="reasoning-block"><summary>查看思考过程</summary><div class="reasoning-content">${escapeText(candidate.reasoning_content)}</div></details>`
    : "";
  const content = renderMarkdown(candidate.content);
  const caret = candidate.status === "streaming" ? '<span class="streaming-caret" aria-label="正在生成"></span>' : "";
  const cancelled = candidate.status === "cancelled" ? '<span class="candidate-state">未完成 · 已停止</span>' : "";
  const autoContinue = candidate.status === "streaming" && candidate.auto_continue
    ? `<span class="candidate-state">正在自动续写第 ${candidate.auto_continue.attempt} 轮 · 已输出约 ${candidate.auto_continue.completion_tokens} / ${candidate.auto_continue.target_completion_tokens} tokens</span>`
    : "";
  return `${reasoning}<div class="assistant-content">${content}${caret}</div>${autoContinue}${cancelled}`;
}

function renderActions(exchange, candidate) {
  const { candidates, index } = visibleCandidateIndex(exchange, candidate);
  const selected = candidate?.id === exchange.selected_candidate_id;
  const selectable = candidate?.status === "completed";
  const generatingThis = candidate?.status === "streaming";
  if (generatingThis) {
    return `<div class="message-actions"><button class="action-button stop-generation" type="button">停止生成</button></div>`;
  }
  return `
    <div class="message-actions" data-exchange-id="${exchange.id}" data-candidate-id="${candidate?.id || ""}">
      <button class="action-button candidate-prev" type="button" aria-label="上一版" title="上一版" ${index <= 0 ? "disabled" : ""}>${icons.left}</button>
      <span class="candidate-counter">${candidates.length ? index + 1 : 0} / ${candidates.length}</span>
      <button class="action-button candidate-next" type="button" aria-label="下一版" title="下一版" ${index >= candidates.length - 1 ? "disabled" : ""}>${icons.right}</button>
      <button class="action-button select-candidate" type="button" ${!selectable || selected ? "disabled" : ""}>
        ${icons.check}<span>${selected ? "已选用" : "选用此版本"}</span>
      </button>
      <button class="action-button regenerate" type="button">${icons.refresh}<span>重新生成</span></button>
      <button class="action-button copy-message" type="button" aria-label="复制" title="复制">${icons.copy}</button>
      <button class="action-button add-to-library" type="button" ${!selectable || state.appendedCandidateIds.has(candidate?.id) ? "disabled" : ""}>
        <span>${state.appendedCandidateIds.has(candidate?.id) ? "已加入章节" : "加入章节"}</span>
      </button>
      <span class="generation-meta">${generationMeta(candidate || {})}</span>
    </div>`;
}

function renderMessages({ keepScroll = true } = {}) {
  const conversation = state.conversation;
  const nearBottom = elements.chatScroll.scrollHeight - elements.chatScroll.scrollTop - elements.chatScroll.clientHeight < 130;
  elements.conversationTitle.textContent = conversation?.title || "新对话";
  const exchanges = conversation?.exchanges || [];
  elements.welcome.hidden = exchanges.length > 0;
  elements.messages.hidden = exchanges.length === 0;
  const html = exchanges
    .map((exchange) => {
      const candidate = getViewedCandidate(exchange);
      const previewing = Boolean(candidate && exchange.selected_candidate_id && candidate.id !== exchange.selected_candidate_id);
      return `<article class="exchange" data-exchange-id="${exchange.id}">
        <div class="user-message">${escapeText(exchange.user_content)}</div>
        <div class="assistant-message">
          <div class="assistant-label"><span class="assistant-avatar">墨</span><span>Novel-factory</span></div>
          ${renderAssistantContent(candidate)}
          ${previewing ? '<span class="candidate-state">正在预览，尚未选用</span>' : ""}
          ${renderActions(exchange, candidate)}
        </div>
      </article>`;
    })
    .join("");
  elements.messages.innerHTML = html;
  bindMessageActions();
  updatePreviewNotice();
  if (keepScroll && (nearBottom || state.shouldFollowStream)) {
    requestAnimationFrame(() => {
      elements.chatScroll.scrollTop = elements.chatScroll.scrollHeight;
    });
  }
}

function updatePreviewNotice() {
  const exchange = state.conversation?.exchanges?.at(-1);
  const candidate = exchange ? getViewedCandidate(exchange) : null;
  elements.previewNotice.hidden = !(
    exchange && candidate && exchange.selected_candidate_id && candidate.id !== exchange.selected_candidate_id
  );
}

function bindMessageActions() {
  elements.messages.querySelectorAll(".message-actions").forEach((actions) => {
    const exchange = state.conversation.exchanges.find((item) => item.id === actions.dataset.exchangeId);
    if (!exchange) return;
    const candidate = exchange.candidates.find((item) => item.id === actions.dataset.candidateId);
    actions.querySelector(".candidate-prev")?.addEventListener("click", () => switchCandidate(exchange, candidate, -1));
    actions.querySelector(".candidate-next")?.addEventListener("click", () => switchCandidate(exchange, candidate, 1));
    actions.querySelector(".select-candidate")?.addEventListener("click", () => selectCandidate(exchange, candidate));
    actions.querySelector(".regenerate")?.addEventListener("click", () => regenerate(exchange));
    actions.querySelector(".copy-message")?.addEventListener("click", (event) => copyCandidate(candidate, event.currentTarget));
    actions.querySelector(".add-to-library")?.addEventListener("click", () => openIncrement(candidate));
    actions.querySelector(".stop-generation")?.addEventListener("click", stopGeneration);
  });
  elements.messages.querySelectorAll(".copy-code").forEach((button) => {
    button.addEventListener("click", async () => {
      const code = button.closest(".code-block")?.querySelector("code")?.textContent || "";
      await navigator.clipboard.writeText(code);
      button.textContent = "已复制";
      window.setTimeout(() => (button.textContent = "复制"), 1200);
    });
  });
}

function switchCandidate(exchange, candidate, direction) {
  const { candidates, index } = visibleCandidateIndex(exchange, candidate);
  const next = candidates[index + direction];
  if (!next) return;
  state.viewedCandidates.set(exchange.id, next.id);
  renderMessages({ keepScroll: false });
}

async function selectCandidate(exchange, candidate) {
  if (!candidate || candidate.status !== "completed") return;
  try {
    const updated = await api.selectCandidate(exchange.id, candidate.id);
    replaceExchange(updated);
    renderMessages({ keepScroll: false });
    showToast("已选用这个版本，后续会以它为上下文");
  } catch (error) {
    if (error instanceof ApiError && error.code === "BRANCH_REQUIRED") {
      const shouldBranch = window.confirm("这条回复后面已经有内容。要从当前版本创建一条新分支吗？原对话不会改变。");
      if (!shouldBranch) return;
      try {
        const branch = await api.branch(exchange.id, candidate.id);
        await refreshConversationList();
        await loadConversation(branch.id);
        showToast("已创建新分支");
      } catch (branchError) {
        showToast(errorMessage(branchError), "error");
      }
      return;
    }
    showToast(errorMessage(error), "error");
  }
}

async function copyCandidate(candidate, button) {
  if (!candidate?.content) return;
  await navigator.clipboard.writeText(candidate.content);
  const label = button.getAttribute("title");
  button.setAttribute("title", "已复制");
  showToast("已复制到剪贴板");
  window.setTimeout(() => button.setAttribute("title", label || "复制"), 1200);
}

function replaceExchange(updated) {
  const index = state.conversation.exchanges.findIndex((item) => item.id === updated.id);
  if (index >= 0) state.conversation.exchanges[index] = updated;
}

function currentGenerationSettings() {
  return state.conversation?.generation_settings || {
    temperature: 0.9,
    top_p: 0.95,
    max_tokens: 1600,
    min_completion_tokens: 2000,
    repeat_penalty: 1.08,
    seed: null,
  };
}

function addOrUpdateStreamingCandidate(data) {
  let exchange = state.conversation.exchanges.find((item) => item.id === data.exchange_id);
  if (!exchange) {
    exchange = {
      id: data.exchange_id,
      conversation_id: state.conversation.id,
      position: state.conversation.exchanges.length + 1,
      user_content: data.user_content,
      selected_candidate_id: null,
      candidates: [],
      created_at: new Date().toISOString(),
    };
    state.conversation.exchanges.push(exchange);
  }
  const existing = exchange.candidates.findIndex((item) => item.id === data.candidate.id);
  if (existing >= 0) exchange.candidates[existing] = data.candidate;
  else exchange.candidates.push(data.candidate);
  state.activeCandidateId = data.candidate.id;
  state.viewedCandidates.set(exchange.id, data.candidate.id);
  if (data.prompt_tokens != null) {
    state.contextStats = {
      input_tokens: data.prompt_tokens,
      context_size: data.context_size || state.runtime?.context_size || 32768,
      reserved_output_tokens: currentGenerationSettings().max_tokens + 384,
    };
    renderContextUsage();
  }
  if (data.trimmed_exchange_count > 0) {
    showToast(`上下文较长，本次未发送最早的 ${data.trimmed_exchange_count} 轮对话`);
  }
}

function updateActiveCandidate(delta) {
  for (const exchange of state.conversation.exchanges) {
    const candidate = exchange.candidates.find((item) => item.id === state.activeCandidateId);
    if (!candidate) continue;
    if (delta.content) candidate.content += delta.content;
    if (delta.reasoning) candidate.reasoning_content += delta.reasoning;
    if (delta.autoContinue) candidate.auto_continue = delta.autoContinue;
    return;
  }
}

async function runStream(path, body) {
  state.generating = true;
  state.shouldFollowStream = true;
  state.streamController = new AbortController();
  updateSendButton();
  try {
    await stream(path, body, {
      signal: state.streamController.signal,
      onEvent: async (event, data) => {
        if (event === "candidate_created") {
          addOrUpdateStreamingCandidate(data);
          renderMessages();
        } else if (event === "content_delta") {
          updateActiveCandidate({ content: data.text });
          renderMessages();
        } else if (event === "reasoning_delta") {
          updateActiveCandidate({ reasoning: data.text });
          renderMessages();
        } else if (event === "auto_continue_started") {
          updateActiveCandidate({ autoContinue: data });
          renderMessages();
          showToast(`正文偏短，正在自动续写第 ${data.attempt} 轮`);
        } else if (["done", "cancelled"].includes(event)) {
          replaceExchange(data.exchange);
          state.viewedCandidates.set(data.exchange.id, data.candidate_id);
          renderMessages();
        } else if (event === "error") {
          if (data.exchange) replaceExchange(data.exchange);
          renderMessages();
          showToast(data.message || "生成失败", "error");
        }
      },
    });
  } catch (error) {
    if (error?.name !== "AbortError") showToast(errorMessage(error), "error");
  } finally {
    state.generating = false;
    state.activeCandidateId = null;
    state.streamController = null;
    updateSendButton();
    await refreshConversationList();
    scheduleContextUsage();
  }
}

async function sendMessage() {
  if (state.generating) return stopGeneration();
  const content = elements.composerInput.value.trim();
  if (!content || !state.conversation) return;
  const lastExchange = state.conversation.exchanges.at(-1);
  if (lastExchange) {
    const viewed = getViewedCandidate(lastExchange);
    if (viewed && lastExchange.selected_candidate_id && viewed.id !== lastExchange.selected_candidate_id) {
      const usePreview = window.confirm("当前正在预览另一个候选版本。点击“确定”会选用它并继续；点击“取消”将保持原版本继续。");
      if (usePreview) {
        await selectCandidate(lastExchange, viewed);
        if (state.conversation.id !== lastExchange.conversation_id) return;
      }
    }
  }
  elements.composerInput.value = "";
  localStorage.removeItem(`llm4chat-draft:${state.conversation.id}`);
  autoResizeComposer();
  await runStream(`/api/conversations/${state.conversation.id}/generate`, {
    content,
    settings: currentGenerationSettings(),
  });
}

async function regenerate(exchange) {
  if (state.generating) {
    showToast("请先停止当前生成");
    return;
  }
  await runStream(`/api/exchanges/${exchange.id}/regenerate`, {
    settings: currentGenerationSettings(),
  });
}

async function stopGeneration() {
  if (!state.generating) return;
  try {
    await api.stop();
  } catch {
    // Aborting the browser stream still causes the server to cancel upstream.
  }
  window.setTimeout(() => state.streamController?.abort(), 250);
}

async function refreshConversationList() {
  try {
    const result = await api.listConversations();
    state.conversations = result.items;
    if (state.conversation) {
      const currentSummary = state.conversations.find((item) => item.id === state.conversation.id);
      if (currentSummary) {
        state.conversation.title = currentSummary.title;
        elements.conversationTitle.textContent = currentSummary.title;
      }
    }
    renderConversationList();
  } catch (error) {
    showToast(errorMessage(error), "error");
  }
}

async function loadConversation(id, { enforceWindowIsolation = true } = {}) {
  if (state.generating) {
    showToast("请先停止当前生成");
    return false;
  }
  if (enforceWindowIsolation && id !== state.conversation?.id && await conversationOpenElsewhere(id)) {
    showToast("这个对话已在另一个窗口打开。为避免上下文串线，请在本窗口新建或选择其他对话。", "error");
    return false;
  }
  try {
    state.conversation = await api.getConversation(id);
    state.project = null;
    state.workspace = null;
    state.pendingConversationId = id;
    sessionStorage.setItem(TAB_CONVERSATION_KEY, id);
    announceTabConversation();
    state.outline = null;
    state.outlineDrafts = [];
    state.previousOutlineId = null;
    state.outlineViewedCandidateId = null;
    state.contextStats = null;
    state.viewedCandidates.clear();
    for (const exchange of state.conversation.exchanges) {
      if (exchange.selected_candidate_id) state.viewedCandidates.set(exchange.id, exchange.selected_candidate_id);
    }
    renderConversationList();
    renderMessages();
    restoreDraft();
    scheduleContextUsage();
    return true;
  } catch (error) {
    state.pendingConversationId = state.conversation?.id || null;
    showToast(errorMessage(error), "error");
    return false;
  }
}

async function createConversation() {
  if (state.generating) {
    showToast("请先停止当前生成");
    return;
  }
  try {
    const conversation = await api.createConversation();
    await refreshConversationList();
    await loadConversation(conversation.id);
    elements.composerInput.focus();
    closeMobileSidebar();
  } catch (error) {
    showToast(errorMessage(error), "error");
  }
}

async function deleteConversation(conversation) {
  if (state.generating) {
    showToast("请先停止当前生成");
    return;
  }
  if (!window.confirm(`确定删除“${conversation.title}”吗？这条对话会从列表中移除。`)) return;
  try {
    await api.deleteConversation(conversation.id);
    localStorage.removeItem(`llm4chat-draft:${conversation.id}`);
    await refreshConversationList();
    if (state.conversation?.id === conversation.id) {
      if (state.conversations.length) await loadConversation(state.conversations[0].id);
      else await createConversation();
    }
  } catch (error) {
    showToast(errorMessage(error), "error");
  }
}

async function renameConversation() {
  if (!state.conversation) return;
  const title = window.prompt("对话名称", state.conversation.title)?.trim();
  if (!title || title === state.conversation.title) return;
  try {
    state.conversation = await api.updateConversation(state.conversation.id, { title });
    await refreshConversationList();
    renderMessages({ keepScroll: false });
  } catch (error) {
    showToast(errorMessage(error), "error");
  }
}

function openSettings() {
  if (!state.conversation) return;
  closeProject();
  closeOutline();
  const settings = currentGenerationSettings();
  elements.temperature.value = settings.temperature;
  elements.temperatureValue.value = Number(settings.temperature).toFixed(2);
  elements.topP.value = settings.top_p;
  elements.topPValue.value = Number(settings.top_p).toFixed(2);
  elements.maxTokens.value = settings.max_tokens;
  elements.minCompletionTokens.value = settings.min_completion_tokens ?? 2000;
  elements.repeatPenalty.value = settings.repeat_penalty;
  elements.randomSeed.checked = settings.seed == null;
  elements.seed.value = settings.seed ?? 42;
  elements.seedField.hidden = elements.randomSeed.checked;
  elements.systemPrompt.value = state.conversation.system_prompt;
  elements.pinnedContext.value = state.conversation.pinned_context;
  elements.styleGuide.value = state.conversation.style_guide || "";
  elements.styleLexicon.value = state.conversation.style_lexicon || "";
  elements.settingsSaveState.textContent = "";
  document.querySelectorAll("[data-style-template]").forEach((button) => {
    button.classList.remove("is-active");
  });
  elements.settingsBackdrop.hidden = false;
  elements.settingsPanel.hidden = false;
  document.querySelectorAll("[data-context-size]").forEach((button) => {
    button.classList.toggle("is-active", Number(button.dataset.contextSize) === Number(state.runtime?.context_size));
  });
  syncBodyLock();
}

function closeSettings() {
  elements.settingsBackdrop.hidden = true;
  elements.settingsPanel.hidden = true;
  syncBodyLock();
}

function setPreset(name) {
  const preset = presets[name];
  if (!preset) return;
  elements.temperature.value = preset.temperature;
  elements.topP.value = preset.top_p;
  elements.repeatPenalty.value = preset.repeat_penalty;
  elements.temperatureValue.value = preset.temperature.toFixed(2);
  elements.topPValue.value = preset.top_p.toFixed(2);
  elements.presetLabel.textContent = preset.label;
  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.preset === name);
  });
}

function appendTemplateText(current, addition) {
  const existing = String(current || "").trim();
  const next = String(addition || "").trim();
  if (!next || existing.includes(next)) return existing;
  return existing ? `${existing}\n\n${next}` : next;
}

function applyStyleTemplate(name) {
  const template = styleTemplates[name];
  if (!template) return;
  elements.styleGuide.value = appendTemplateText(elements.styleGuide.value, template.guide);
  elements.styleLexicon.value = appendTemplateText(elements.styleLexicon.value, template.lexicon);
  elements.settingsSaveState.textContent = `${template.label}已加入`;
  document.querySelectorAll("[data-style-template]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.styleTemplate === name);
  });
}

async function saveSettings() {
  if (!state.conversation) return;
  const generationSettings = {
    temperature: Number(elements.temperature.value),
    top_p: Number(elements.topP.value),
    max_tokens: Number(elements.maxTokens.value),
    min_completion_tokens: Number(elements.minCompletionTokens.value),
    repeat_penalty: Number(elements.repeatPenalty.value),
    seed: elements.randomSeed.checked ? null : Number(elements.seed.value),
  };
  try {
    state.conversation = await api.updateConversation(state.conversation.id, {
      system_prompt: elements.systemPrompt.value,
      pinned_context: elements.pinnedContext.value,
      style_guide: elements.styleGuide.value,
      style_lexicon: elements.styleLexicon.value,
      generation_settings: generationSettings,
    });
    elements.settingsSaveState.textContent = "已保存";
    showToast("创作设置已保存");
    scheduleContextUsage();
    window.setTimeout(closeSettings, 350);
  } catch (error) {
    elements.settingsSaveState.textContent = errorMessage(error);
  }
}

function exportConversation(format) {
  if (!state.conversation) return;
  const url = `/api/conversations/${state.conversation.id}/export?format=${format}${format === "markdown" ? "&include_all=true" : ""}`;
  const link = document.createElement("a");
  link.href = url;
  link.click();
}

async function pollRuntime({ announce = false } = {}) {
  try {
    state.runtime = await api.runtime();
    const status = state.runtime.status;
    elements.runtimeStatus.className = `runtime-status ${status === "ready" ? "is-ready" : status === "error" ? "is-error" : "is-loading"}`;
    elements.runtimeStatusText.textContent = status === "ready" ? "本地模型已就绪" : state.runtime.message;
    elements.runtimeStatus.title = `${state.runtime.model_name}\n${state.runtime.message}`;
    document.querySelectorAll("[data-context-size]").forEach((button) => {
      button.classList.toggle("is-active", Number(button.dataset.contextSize) === Number(state.runtime.context_size));
    });
    renderContextUsage();
    if (announce && status === "ready") showToast("本地模型已就绪");
  } catch {
    state.runtime = { status: "error", message: "应用服务连接失败" };
    elements.runtimeStatus.className = "runtime-status is-error";
    elements.runtimeStatusText.textContent = "服务连接失败";
  }
  updateSendButton();
}

async function changeContextSize(contextSize) {
  if (Number(state.runtime?.context_size) === contextSize) return;
  if (state.generating || state.analysisRunning || state.outlineGenerating) {
    showToast("请先等待当前任务结束", "error");
    return;
  }
  const label = contextSize === 65536 ? "64K" : "32K";
  elements.runtimeStatus.className = "runtime-status is-loading";
  elements.runtimeStatusText.textContent = `正在切换到 ${label}`;
  document.querySelectorAll("[data-context-size]").forEach((button) => (button.disabled = true));
  try {
    state.runtime = await api.changeContext(contextSize);
    state.contextStats = null;
    await pollRuntime();
    await updateContextUsage();
    showToast(`上下文窗口已切换为 ${label}`);
  } catch (error) {
    showToast(errorMessage(error), "error");
    await pollRuntime();
  } finally {
    document.querySelectorAll("[data-context-size]").forEach((button) => (button.disabled = false));
  }
}

async function handleRuntimeClick() {
  if (state.runtime?.status === "ready" || state.runtime?.status === "loading") {
    showToast(state.runtime.message || "模型正在运行");
    return;
  }
  elements.runtimeStatus.className = "runtime-status is-loading";
  elements.runtimeStatusText.textContent = "正在启动模型";
  try {
    await api.startRuntime();
    await pollRuntime({ announce: true });
  } catch (error) {
    showToast(errorMessage(error), "error");
    await pollRuntime();
  }
}

function bindStaticEvents() {
  document.querySelector("#new-chat").addEventListener("click", createConversation);
  document.querySelector("#brand-button").addEventListener("click", createConversation);
  document.querySelector("#collapse-sidebar").addEventListener("click", () => elements.app.classList.add("sidebar-collapsed"));
  document.querySelector("#expand-sidebar").addEventListener("click", () => elements.app.classList.remove("sidebar-collapsed"));
  document.querySelector("#open-sidebar").addEventListener("click", () => elements.app.classList.add("mobile-sidebar-open"));
  elements.sidebarOverlay.addEventListener("click", closeMobileSidebar);
  elements.conversationTitle.addEventListener("click", renameConversation);
  elements.runtimeStatus.addEventListener("click", handleRuntimeClick);
  elements.contextUsage.addEventListener("click", openSettings);
  document.querySelector("#open-project").addEventListener("click", openProject);
  document.querySelector("#close-project").addEventListener("click", closeProject);
  elements.projectBackdrop.addEventListener("click", closeProject);
  elements.importTxt.addEventListener("click", () => elements.txtFile.click());
  elements.txtFile.addEventListener("change", () => importTxtFile(elements.txtFile.files?.[0]));
  elements.exportMaterialPackage.addEventListener("click", exportMaterialPackage);
  elements.importMaterialPackage.addEventListener("click", () => elements.materialPackageFile.click());
  elements.materialPackageFile.addEventListener("change", () => importMaterialPackageFile(elements.materialPackageFile.files?.[0]));
  elements.rebuildMaterialSystem.addEventListener("click", rebuildMaterialSystem);
  elements.previewMaterialPlan.addEventListener("click", previewMaterialPromptPlan);
  elements.editMaterialBudget.addEventListener("click", editMaterialBudget);
  elements.refreshMaterialReviews.addEventListener("click", refreshMaterialReviews);
  elements.inspectMaterialSystem.addEventListener("click", inspectMaterialSystem);
  elements.documentSelect.addEventListener("change", () => selectDocument(elements.documentSelect.value));
  document.querySelector("#save-global-summary").addEventListener("click", saveProjectSummary);
  elements.libraryEnabled.addEventListener("change", () => saveDocumentSetting("library_enabled", elements.libraryEnabled.checked));
  elements.summaryEnabled.addEventListener("change", () => saveDocumentSetting("summary_enabled", elements.summaryEnabled.checked));
  document.querySelectorAll(".workspace-module .compact-toggle").forEach((toggle) => {
    toggle.addEventListener("click", (event) => event.stopPropagation());
  });
  elements.recentChaptersEnabled.addEventListener("change", () => saveDocumentSetting("recent_chapters_enabled", elements.recentChaptersEnabled.checked));
  elements.charactersEnabled.addEventListener("change", () => saveDocumentSetting("characters_enabled", elements.charactersEnabled.checked));
  elements.factsEnabled.addEventListener("change", () => saveDocumentSetting("facts_enabled", elements.factsEnabled.checked));
  elements.summarizeProject.addEventListener("click", toggleProjectSummary);
  elements.resumeAnalysis.addEventListener("click", () => {
    const jobId = state.workspace?.latest_job?.status === "paused" ? state.workspace.latest_job.id : null;
    if (jobId) runProjectSummary(null, false, jobId);
  });
  document.querySelector("#export-project-txt").addEventListener("click", exportProjectTxt);
  elements.previewPrompt.addEventListener("click", previewInjectedPrompt);
  document.querySelector("#clear-project-library").addEventListener("click", clearProjectLibrary);
  document.querySelector("#open-outline").addEventListener("click", openOutline);
  document.querySelector("#close-outline").addEventListener("click", closeOutline);
  elements.outlineBackdrop.addEventListener("click", closeOutline);
  elements.outlinePrev.addEventListener("click", () => switchOutlineCandidate(-1));
  elements.outlineNext.addEventListener("click", () => switchOutlineCandidate(1));
  elements.newOutline.addEventListener("click", () => generateOutline(true));
  elements.rerollOutline.addEventListener("click", () => generateOutline(false));
  elements.saveOutline.addEventListener("click", saveOutlineCandidate);
  elements.selectOutline.addEventListener("click", selectOutlineCandidate);
  elements.outlineEnabled.addEventListener("change", toggleOutlineEnabled);
  elements.deleteOutlineCandidate.addEventListener("click", deleteCurrentOutlineCandidate);
  elements.clearOutline.addEventListener("click", clearOutlineCandidates);
  document.querySelector("#close-increment").addEventListener("click", closeIncrement);
  document.querySelector("#cancel-increment").addEventListener("click", closeIncrement);
  elements.incrementBackdrop.addEventListener("click", closeIncrement);
  elements.incrementTarget.addEventListener("change", () => {
    elements.incrementTitleField.hidden = elements.incrementTarget.value !== "__new__";
  });
  elements.incrementSummarizeNow.addEventListener("change", () => {
    elements.confirmIncrement.textContent = elements.incrementSummarizeNow.checked ? "加入并立即总结" : "仅加入章节";
    elements.incrementStatus.textContent = elements.incrementSummarizeNow.checked
      ? "正文会先安全写入，再更新本章摘要、总览、人物卡和结构化事实。"
      : "只保存正文并标记为待总结；可稍后在资料库批量处理。";
  });
  elements.confirmIncrement.addEventListener("click", confirmIncrement);
  ["#open-settings-sidebar", "#open-settings-top", "#composer-settings"].forEach((selector) => {
    document.querySelector(selector).addEventListener("click", openSettings);
  });
  document.querySelector("#close-settings").addEventListener("click", closeSettings);
  elements.settingsBackdrop.addEventListener("click", closeSettings);
  document.querySelector("#save-settings").addEventListener("click", saveSettings);
  document.querySelector("#export-markdown").addEventListener("click", () => exportConversation("markdown"));
  document.querySelector("#export-json").addEventListener("click", () => exportConversation("json"));
  elements.randomSeed.addEventListener("change", () => (elements.seedField.hidden = elements.randomSeed.checked));
  elements.temperature.addEventListener("input", () => (elements.temperatureValue.value = Number(elements.temperature.value).toFixed(2)));
  elements.topP.addEventListener("input", () => (elements.topPValue.value = Number(elements.topP.value).toFixed(2)));
  document.querySelectorAll("[data-preset]").forEach((button) => button.addEventListener("click", () => setPreset(button.dataset.preset)));
  document.querySelectorAll("[data-style-template]").forEach((button) => {
    button.addEventListener("click", () => applyStyleTemplate(button.dataset.styleTemplate));
  });
  document.querySelectorAll("[data-theme]").forEach((button) => button.addEventListener("click", () => setTheme(button.dataset.theme)));
  document.querySelectorAll("[data-context-size]").forEach((button) => {
    button.addEventListener("click", () => changeContextSize(Number(button.dataset.contextSize)));
  });
  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      elements.composerInput.value = button.dataset.prompt;
      autoResizeComposer();
      elements.composerInput.focus();
    });
  });
  elements.composerInput.addEventListener("input", () => {
    autoResizeComposer();
    saveDraft();
    scheduleContextUsage();
  });
  elements.composerInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendMessage();
    }
  });
  elements.composerForm.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage();
  });
  elements.chatScroll.addEventListener("scroll", () => {
    const distance = elements.chatScroll.scrollHeight - elements.chatScroll.scrollTop - elements.chatScroll.clientHeight;
    state.shouldFollowStream = distance < 130;
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      if (!elements.settingsPanel.hidden) closeSettings();
      if (!elements.projectPanel.hidden) closeProject();
      if (!elements.outlinePanel.hidden) closeOutline();
      if (!elements.incrementDialog.hidden) closeIncrement();
      closeMobileSidebar();
    }
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if ((localStorage.getItem("llm4chat-theme") || "system") === "system") setTheme("system");
  });
}

async function initialize() {
  setTheme(localStorage.getItem("llm4chat-theme") || "system");
  initializeWindowIsolation();
  bindStaticEvents();
  await Promise.all([refreshConversationList(), pollRuntime()]);
  const preferredId = sessionStorage.getItem(TAB_CONVERSATION_KEY);
  const preferredExists = state.conversations.some((item) => item.id === preferredId);
  if (preferredExists && !(await conversationOpenElsewhere(preferredId))) {
    await loadConversation(preferredId, { enforceWindowIsolation: false });
  } else {
    await createConversation();
  }
  window.setInterval(pollRuntime, 2500);
}

initialize();
