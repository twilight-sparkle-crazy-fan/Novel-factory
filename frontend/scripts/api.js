export class ApiError extends Error {
  constructor(message, code = "UNKNOWN_ERROR", status = 0, detail = "") {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.detail = detail;
  }
}

async function parseError(response) {
  let body = {};
  try {
    body = await response.json();
  } catch {
    body = {};
  }
  const data = body.error || body.detail || {};
  if (typeof data === "string") {
    return new ApiError(data, "HTTP_ERROR", response.status);
  }
  return new ApiError(
    data.message || `请求失败（${response.status}）`,
    data.code || "HTTP_ERROR",
    response.status,
    data.detail || "",
  );
}

export async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    throw await parseError(response);
  }
  if (response.status === 204) return null;
  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("application/json") ? response.json() : response.text();
}

function parseEventBlock(block) {
  let event = "message";
  const dataLines = [];
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (!dataLines.length) return null;
  const raw = dataLines.join("\n");
  try {
    return { event, data: JSON.parse(raw) };
  } catch {
    return { event, data: { text: raw } };
  }
}

export async function stream(path, body, { signal, onEvent }) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw await parseError(response);
  if (!response.body) throw new ApiError("浏览器不支持流式响应", "STREAM_UNAVAILABLE");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      const parsed = parseEventBlock(block);
      if (parsed) await onEvent(parsed.event, parsed.data);
    }
    if (done) break;
  }
  if (buffer.trim()) {
    const parsed = parseEventBlock(buffer);
    if (parsed) await onEvent(parsed.event, parsed.data);
  }
}

export const api = {
  runtime: () => request("/api/runtime"),
  startRuntime: () => request("/api/runtime/start", { method: "POST" }),
  changeContext: (contextSize) =>
    request("/api/runtime/context", {
      method: "POST",
      body: JSON.stringify({ context_size: contextSize }),
    }),
  listConversations: () => request("/api/conversations"),
  createConversation: (title = "新对话") =>
    request("/api/conversations", { method: "POST", body: JSON.stringify({ title }) }),
  getConversation: (id) => request(`/api/conversations/${id}`),
  updateConversation: (id, changes) =>
    request(`/api/conversations/${id}`, { method: "PATCH", body: JSON.stringify(changes) }),
  deleteConversation: (id) => request(`/api/conversations/${id}`, { method: "DELETE" }),
  selectCandidate: (exchangeId, candidateId) =>
    request(`/api/exchanges/${exchangeId}/selection`, {
      method: "PUT",
      body: JSON.stringify({ candidate_id: candidateId }),
    }),
  branch: (exchangeId, candidateId) =>
    request(`/api/exchanges/${exchangeId}/branch`, {
      method: "POST",
      body: JSON.stringify({ candidate_id: candidateId }),
    }),
  countContext: (conversationId, content = "") =>
    request(`/api/conversations/${conversationId}/context-count`, {
      method: "POST",
      body: JSON.stringify({ content }),
    }),
  getProject: (id = "default") => request(`/api/projects/${id}`),
  getDocumentWorkspace: (id) => request(`/api/documents/${id}/workspace`),
  updateDocument: (id, changes) =>
    request(`/api/documents/${id}`, { method: "PATCH", body: JSON.stringify(changes) }),
  updateProject: (id, changes) =>
    request(`/api/projects/${id}`, { method: "PATCH", body: JSON.stringify(changes) }),
  importTxt: async (id, file) => {
    const response = await fetch(`/api/projects/${id}/import-txt`, {
      method: "POST",
      headers: { "X-Filename": encodeURIComponent(file.name) },
      body: await file.arrayBuffer(),
    });
    if (!response.ok) throw await parseError(response);
    return response.json();
  },
  exportMaterialPackage: async (documentId) => {
    const response = await fetch(`/api/experimental/material-system/documents/${documentId}/package`);
    if (!response.ok) throw await parseError(response);
    const disposition = response.headers.get("content-disposition") || "";
    const match = disposition.match(/filename\*=UTF-8''([^;]+)/);
    const filename = match ? decodeURIComponent(match[1]) : "project-analysis.llm4pkg";
    return { blob: await response.blob(), filename };
  },
  validateMaterialPackage: async (file, documentId = null, scope = null) => {
    const params = new URLSearchParams();
    if (documentId) params.set("document_id", documentId);
    if (scope?.chapterStart) params.set("chapter_start", scope.chapterStart);
    if (scope?.chapterEnd) params.set("chapter_end", scope.chapterEnd);
    const suffix = params.toString() ? `?${params.toString()}` : "";
    const path = `/api/experimental/material-system/packages/validate${suffix}`;
    const response = await fetch(path, {
      method: "POST",
      body: await file.arrayBuffer(),
    });
    if (!response.ok) throw await parseError(response);
    return response.json();
  },
  importMaterialPackage: async (projectId, file, { mode = "create_document", documentId = null, layers = [], scope = null } = {}) => {
    const params = new URLSearchParams({ project_id: projectId, mode });
    if (documentId) params.set("document_id", documentId);
    if (layers.length) params.set("material_layers", layers.join(","));
    if (scope?.chapterStart) params.set("chapter_start", scope.chapterStart);
    if (scope?.chapterEnd) params.set("chapter_end", scope.chapterEnd);
    const response = await fetch(`/api/experimental/material-system/packages/import?${params.toString()}`, {
      method: "POST",
      body: await file.arrayBuffer(),
    });
    if (!response.ok) throw await parseError(response);
    return response.json();
  },
  getMaterialOverview: (documentId) =>
    request(`/api/experimental/material-system/documents/${documentId}/overview`),
  rebuildMaterialSystem: (documentId) =>
    request(`/api/experimental/material-system/documents/${documentId}/rebuild`, { method: "POST" }),
  createMaterialTimelineNode: (documentId, payload) =>
    request(`/api/experimental/material-system/documents/${documentId}/timeline/nodes`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  createMaterialTimelineEvent: (documentId, payload) =>
    request(`/api/experimental/material-system/documents/${documentId}/timeline/events`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updateMaterialTimelineEvent: (eventId, payload) =>
    request(`/api/experimental/material-system/timeline-events/${eventId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  deleteMaterialTimelineEvent: (eventId) =>
    request(`/api/experimental/material-system/timeline-events/${eventId}`, { method: "DELETE" }),
  updateMaterialTimelineNode: (nodeId, payload) =>
    request(`/api/experimental/material-system/timeline-nodes/${nodeId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  deleteMaterialTimelineNode: (nodeId) =>
    request(`/api/experimental/material-system/timeline-nodes/${nodeId}`, { method: "DELETE" }),
  createMaterialCharacterEntity: (documentId, payload) =>
    request(`/api/experimental/material-system/documents/${documentId}/characters/entities`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updateMaterialCharacterEntity: (characterId, payload) =>
    request(`/api/experimental/material-system/characters/entities/${characterId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  deleteMaterialCharacterEntity: (characterId) =>
    request(`/api/experimental/material-system/characters/entities/${characterId}`, { method: "DELETE" }),
  createMaterialCharacterProfile: (characterId, payload) =>
    request(`/api/experimental/material-system/characters/entities/${characterId}/profiles`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updateMaterialCharacterProfile: (profileId, payload) =>
    request(`/api/experimental/material-system/characters/profiles/${profileId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  deleteMaterialCharacterProfile: (profileId) =>
    request(`/api/experimental/material-system/characters/profiles/${profileId}`, { method: "DELETE" }),
  createMaterialCharacterEvent: (characterId, payload) =>
    request(`/api/experimental/material-system/characters/entities/${characterId}/events`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updateMaterialCharacterEvent: (eventId, payload) =>
    request(`/api/experimental/material-system/characters/events/${eventId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  deleteMaterialCharacterEvent: (eventId) =>
    request(`/api/experimental/material-system/characters/events/${eventId}`, { method: "DELETE" }),
  addMaterialCharacterAlias: (characterId, payload) =>
    request(`/api/experimental/material-system/characters/entities/${characterId}/aliases`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  mergeMaterialCharacterEntity: (characterId, payload) =>
    request(`/api/experimental/material-system/characters/entities/${characterId}/merge`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  createMaterialRelationship: (documentId, payload) =>
    request(`/api/experimental/material-system/documents/${documentId}/relationships`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updateMaterialRelationship: (relationshipId, payload) =>
    request(`/api/experimental/material-system/relationships/${relationshipId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  deleteMaterialRelationship: (relationshipId) =>
    request(`/api/experimental/material-system/relationships/${relationshipId}`, { method: "DELETE" }),
  getMaterialBudgetProfile: (documentId) =>
    request(`/api/experimental/material-system/documents/${documentId}/prompt-budget-profile`),
  updateMaterialBudgetProfile: (documentId, payload) =>
    request(`/api/experimental/material-system/documents/${documentId}/prompt-budget-profile`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  listMaterialReviewItems: (documentId) =>
    request(`/api/experimental/material-system/documents/${documentId}/review-items`),
  resolveMaterialReviewItem: (itemId, payload = {}) =>
    request(`/api/experimental/material-system/review-items/${itemId}/resolve`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  rejectMaterialReviewItem: (itemId, payload = {}) =>
    request(`/api/experimental/material-system/review-items/${itemId}/reject`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  materialPromptPlan: (documentId, payload) =>
    request(`/api/experimental/material-system/documents/${documentId}/prompt-plan`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getChapter: (id) => request(`/api/chapters/${id}`),
  updateChapter: (id, changes) =>
    request(`/api/chapters/${id}`, { method: "PATCH", body: JSON.stringify(changes) }),
  updateCharacter: (id, changes) =>
    request(`/api/characters/${id}`, { method: "PATCH", body: JSON.stringify(changes) }),
  deleteDocument: (id) => request(`/api/documents/${id}`, { method: "DELETE" }),
  deleteChapter: (id) => request(`/api/chapters/${id}`, { method: "DELETE" }),
  deleteCharacter: (id) => request(`/api/characters/${id}`, { method: "DELETE" }),
  updateFact: (id, changes) =>
    request(`/api/facts/${id}`, { method: "PATCH", body: JSON.stringify(changes) }),
  deleteFact: (id) => request(`/api/facts/${id}`, { method: "DELETE" }),
  clearProjectLibrary: (id) => request(`/api/projects/${id}/library`, { method: "DELETE" }),
  getOutline: (conversationId) => request(`/api/conversations/${conversationId}/outline`),
  promptPreview: (conversationId, query = "") =>
    request(`/api/conversations/${conversationId}/prompt-preview?query=${encodeURIComponent(query)}`),
  saveOutlineCandidate: (conversationId, payload) =>
    request(`/api/conversations/${conversationId}/outline/candidates`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updateOutline: (outlineId, enabled) =>
    request(`/api/outlines/${outlineId}`, {
      method: "PATCH",
      body: JSON.stringify({ enabled }),
    }),
  selectOutline: (outlineId, candidateId) =>
    request(`/api/outlines/${outlineId}/selection`, {
      method: "PUT",
      body: JSON.stringify({ candidate_id: candidateId }),
    }),
  editOutlineCandidate: (candidateId, content) =>
    request(`/api/outline-candidates/${candidateId}`, {
      method: "PATCH",
      body: JSON.stringify({ content }),
    }),
  deleteOutlineCandidate: (candidateId) =>
    request(`/api/outline-candidates/${candidateId}`, { method: "DELETE" }),
  deleteOutline: (outlineId) => request(`/api/outlines/${outlineId}`, { method: "DELETE" }),
  stop: () => request("/api/generation/stop", { method: "POST" }),
};
