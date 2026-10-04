import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ApiError,
  createSession,
  getHealth,
  getModels,
  getSessionState,
  type JobInfo,
  type ModelInfo,
  type SessionState,
} from "./api/client";
import {
  createRunState,
  type PendingApproval,
  type RunUiState,
} from "./api/agui";
import ChatPanel from "./components/ChatPanel";
import ResultPanel from "./components/ResultPanel";

const SESSION_KEY = "mlip_agent_session_id";

/** GET /state 快照 → RunUiState（历史消息渲染为静态条目）。 */
function stateToRunState(state: SessionState, base: RunUiState): RunUiState {
  const items: RunUiState["items"] = [];
  let textIdx = 0;
  for (const m of state.messages) {
    if (m.kind === "text" && m.text) {
      items.push({
        kind: "text",
        id: `hist-${textIdx++}`,
        role: m.role,
        text: m.text,
        done: true,
      });
    } else if (m.kind === "tool_call") {
      items.push({
        kind: "tool",
        id: `hist-tc-${textIdx++}`,
        toolCallId: `hist-tc-${textIdx}`,
        toolName: m.tool_name ?? "tool",
        argsText: m.text ?? "",
        argsDone: true,
        result: null,
      });
    }
    // tool_return 的完整内容不存于快照，跳过（计算结果走 artifacts / 重新解析）
  }
  return {
    ...base,
    items,
    status: state.status === "idle" ? "idle" : state.status,
    approvals: state.pending_approvals,
    output: state.last_run?.output ?? null,
    failure: state.last_run?.failure
      ? String(state.last_run.failure.message ?? "运行失败")
      : null,
  };
}

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [title, setTitle] = useState<string | null>(null);
  const [stale, setStale] = useState(false);
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [llmConfigured, setLlmConfigured] = useState<boolean | null>(null);
  const [initError, setInitError] = useState<string | null>(null);
  const [runState, setRunState] = useState<RunUiState>(createRunState);

  // 会话引导：localStorage 里有则恢复，否则（或 404/过期）新建
  useEffect(() => {
    (async () => {
      try {
        const saved = localStorage.getItem(SESSION_KEY);
        if (saved) {
          try {
            const st = await getSessionState(saved);
            setSessionId(st.session_id);
            setTitle(st.title);
            setStale(st.stale_state);
            setJobs(st.jobs);
            setRunState((prev) => stateToRunState(st, prev));
            return;
          } catch (e) {
            if (!(e instanceof ApiError && e.status === 404)) throw e;
            localStorage.removeItem(SESSION_KEY);
          }
        }
        const info = await createSession();
        localStorage.setItem(SESSION_KEY, info.session_id);
        setSessionId(info.session_id);
      } catch (e) {
        setInitError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  useEffect(() => {
    getHealth()
      .then((h) => setLlmConfigured(h.llm_configured))
      .catch(() => setLlmConfigured(null));
    getModels()
      .then(setModels)
      .catch(() => setModels([]));
  }, []);

  /** run 结束后刷新会话快照（作业列表、挂起态）。 */
  const refreshState = useCallback(async () => {
    if (!sessionId) return;
    try {
      const st = await getSessionState(sessionId);
      setJobs(st.jobs);
      setStale(st.stale_state);
      if (st.status === "suspended" && st.pending_approvals.length > 0) {
        setRunState((prev) => ({
          ...prev,
          status: "suspended",
          approvals: st.pending_approvals,
        }));
      }
    } catch {
      /* 忽略刷新失败，下一轮再试 */
    }
  }, [sessionId]);

  const newSession = useCallback(async () => {
    const info = await createSession();
    localStorage.setItem(SESSION_KEY, info.session_id);
    setSessionId(info.session_id);
    setTitle(null);
    setStale(false);
    setJobs([]);
    setRunState(createRunState());
  }, []);

  const onApprovalsConsumed = useCallback((_a: PendingApproval[]) => {
    setRunState((prev) => ({ ...prev, approvals: [] }));
  }, []);

  const layout = useMemo(
    () => ({ ready: sessionId !== null }),
    [sessionId],
  );

  if (initError) {
    return (
      <div className="app-fatal">
        <h1>MLIP Agent</h1>
        <p>无法连接后端（{initError}）。请确认 uvicorn 已在 8000 端口运行。</p>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>MLIP Agent</h1>
          <span className="subtitle">机器学习势计算助手 {title ? `· ${title}` : ""}</span>
        </div>
        <div className="header-right">
          {llmConfigured === false && (
            <span className="badge warn">LLM 端点未配置</span>
          )}
          {models.length > 0 && (
            <span className="badge">{models.length} 个模型可用</span>
          )}
          <button onClick={newSession}>新会话</button>
        </div>
      </header>

      {stale && (
        <div className="banner warn">
          会话状态版本过旧，已弃用旧对话记录（作业产物保留）。如内容缺失请开新会话。
        </div>
      )}

      {layout.ready ? (
        <main className="app-main">
          <ChatPanel
            sessionId={sessionId!}
            runState={runState}
            setRunState={setRunState}
            onRunSettled={refreshState}
            onApprovalsConsumed={onApprovalsConsumed}
          />
          <ResultPanel runState={runState} jobs={jobs} />
        </main>
      ) : (
        <div className="loading">正在初始化会话…</div>
      )}
    </div>
  );
}
