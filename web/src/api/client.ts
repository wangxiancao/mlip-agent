/**
 * REST + SSE 客户端（PLAN §2.6）。
 *
 * SSE 用 fetch + ReadableStream 手工解析：POST 响应即事件流，
 * 帧以空行分隔，只处理 `data:` 行（JSON），忽略 `:hb` 等注释行。
 */

import type { AguiEvent } from "./agui";

const BASE = "/api";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, String(detail));
  }
  return (await res.json()) as T;
}

/* ---------- 类型（对应 app/state/schemas.py） ---------- */

export interface SessionInfo {
  session_id: string;
  created_at: string;
  state_version: number;
}

export interface JobInfo {
  job_id: string;
  session_id: string;
  job_type: string | null;
  artifacts: string[];
  has_output: boolean;
}

export interface ChatMessage {
  role: "user" | "assistant" | "tool";
  kind: "text" | "tool_call" | "tool_return";
  text: string | null;
  tool_name: string | null;
}

export interface SessionState {
  session_id: string;
  state_version: number;
  status: "idle" | "completed" | "suspended" | "failed" | "cancelled";
  stale_state: boolean;
  title: string | null;
  pending_approvals: {
    tool_call_id: string;
    tool_name: string;
    args: string | null;
  }[];
  jobs: JobInfo[];
  messages: ChatMessage[];
  last_run: {
    status: string;
    output: string | null;
    failure: Record<string, unknown> | null;
  } | null;
}

export interface ModelInfo {
  name: string;
  label: string;
  path: string;
  backend: string;
  status: string;
  notes: string;
}

export interface ResumeDecision {
  approve_all: boolean;
  answers: Record<string, string>;
  denials: Record<string, string>;
}

/* ---------- REST ---------- */

export function createSession(title?: string): Promise<SessionInfo> {
  return request("POST", "/sessions", title ? { title } : {});
}

export function getSessionState(sid: string): Promise<SessionState> {
  return request("GET", `/sessions/${encodeURIComponent(sid)}/state`);
}

export function getJob(jid: string): Promise<JobInfo> {
  return request("GET", `/jobs/${encodeURIComponent(jid)}`);
}

export function getModels(): Promise<ModelInfo[]> {
  return request("GET", "/models");
}

export function getHealth(): Promise<{ ok: boolean; llm_configured: boolean }> {
  return request("GET", "/health");
}

export function artifactUrl(jid: string, name: string): string {
  return `${BASE}/jobs/${encodeURIComponent(jid)}/artifacts/${encodeURIComponent(name)}`;
}

/**
 * 从作业工件列表挑出可用于 3D 渲染的结构文件（StructureView 数据源）：
 * 优先 structure.xyz（write_maple_inp 落盘的起始结构），其次其他 .xyz，
 * 最后 .extxyz 轨迹（traj.extxyz，渲染首帧）。无可渲染工件时返回 null。
 */
export function pickStructureArtifact(artifacts: string[]): string | null {
  const lower = artifacts.map((n) => n.toLowerCase());
  const plain = artifacts.filter((_, i) => lower[i].endsWith(".xyz") && !lower[i].endsWith(".extxyz"));
  if (plain.includes("structure.xyz")) return "structure.xyz";
  if (plain.length > 0) return plain[0];
  const ext = artifacts.filter((_, i) => lower[i].endsWith(".extxyz"));
  return ext.length > 0 ? ext[0] : null;
}

/* ---------- SSE ---------- */

export interface SseStreamHandle {
  /** 等待流结束；流过程出错会 reject。 */
  done: Promise<void>;
  abort: () => void;
}

/**
 * 解析 SSE 字节流：按空行切帧，帧内拼接多行 `data:`，忽略注释行（`:` 开头）。
 * 该函数独立导出，便于在无 fetch 环境下单测。
 */
export async function parseSseStream(
  body: ReadableStream<Uint8Array>,
  onEvent: (event: AguiEvent) => void,
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const flushFrame = (frame: string) => {
    const dataLines: string[] = [];
    for (const line of frame.split("\n")) {
      if (line.startsWith(":")) return; // 注释行（:hb 心跳）
      if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).replace(/^ /, ""));
      }
    }
    if (dataLines.length === 0) return;
    const text = dataLines.join("\n");
    if (!text.trim()) return;
    onEvent(JSON.parse(text) as AguiEvent);
  };

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      flushFrame(frame);
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) flushFrame(buffer);
}

async function streamSse(
  path: string,
  body: unknown,
  onEvent: (event: AguiEvent) => void,
  signal?: AbortSignal,
): Promise<SseStreamHandle> {
  const controller = new AbortController();
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener("abort", () => controller.abort(), { once: true });
  }
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    signal: controller.signal,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, String(detail));
  }
  if (!res.body) throw new ApiError(0, "响应无流式 body");

  const done = parseSseStream(res.body, onEvent).catch((err: unknown) => {
    if (controller.signal.aborted) return;
    throw err;
  });
  return { done, abort: () => controller.abort() };
}

/** POST /sessions/{sid}/messages → SSE。 */
export function sendMessage(
  sid: string,
  content: string,
  onEvent: (event: AguiEvent) => void,
  signal?: AbortSignal,
): Promise<SseStreamHandle> {
  return streamSse(
    `/sessions/${encodeURIComponent(sid)}/messages`,
    { content },
    onEvent,
    signal,
  );
}

/** POST /sessions/{sid}/resume → SSE。 */
export function resumeSession(
  sid: string,
  decision: ResumeDecision,
  onEvent: (event: AguiEvent) => void,
  signal?: AbortSignal,
): Promise<SseStreamHandle> {
  return streamSse(
    `/sessions/${encodeURIComponent(sid)}/resume`,
    decision,
    onEvent,
    signal,
  );
}
