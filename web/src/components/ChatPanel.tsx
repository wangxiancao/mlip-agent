import { useCallback, useRef, useState } from "react";
import {
  ApiError,
  resumeSession,
  sendMessage,
  type ResumeDecision,
} from "../api/client";
import { reduceEvent, type RunUiState } from "../api/agui";
import MessageList from "./MessageList";
import ApprovalDialog from "./ApprovalDialog";

interface Props {
  sessionId: string;
  runState: RunUiState;
  setRunState: React.Dispatch<React.SetStateAction<RunUiState>>;
  onRunSettled: () => void;
  onApprovalsConsumed: (approvals: RunUiState["approvals"]) => void;
}

const STATUS_LABEL: Record<RunUiState["status"], string> = {
  idle: "空闲",
  running: "运行中",
  completed: "已完成",
  suspended: "待审批",
  failed: "失败",
  cancelled: "已取消",
};

export default function ChatPanel({
  sessionId,
  runState,
  setRunState,
  onRunSettled,
  onApprovalsConsumed,
}: Props) {
  const [input, setInput] = useState("");
  const [sendError, setSendError] = useState<string | null>(null);
  const abortRef = useRef<(() => void) | null>(null);

  const running = runState.status === "running";
  const suspended = runState.status === "suspended";

  const startStream = useCallback(
    (
      start: (
        onEvent: (e: Parameters<typeof reduceEvent>[1]) => void,
      ) => Promise<{ done: Promise<void>; abort: () => void }>,
    ) => {
      setSendError(null);
      void (async () => {
        try {
          const handle = await start((event) =>
            setRunState((prev) => reduceEvent(prev, event)),
          );
          abortRef.current = handle.abort;
          await handle.done;
        } catch (e) {
          if (e instanceof ApiError && e.status === 409) {
            setSendError(`冲突：${e.message}（会话挂起或有 run 正在进行）`);
          } else if (e instanceof Error && e.name !== "AbortError") {
            setSendError(e.message);
            setRunState((prev) =>
              prev.status === "running"
                ? { ...prev, status: "failed", failure: e.message }
                : prev,
            );
          }
        } finally {
          abortRef.current = null;
          onRunSettled();
        }
      })();
    },
    [setRunState, onRunSettled],
  );

  const send = useCallback(() => {
    const content = input.trim();
    if (!content || running || suspended) return;
    setInput("");
    // 用户消息立即上屏
    setRunState((prev) => ({
      ...prev,
      items: [
        ...prev.items,
        {
          kind: "text",
          id: `user-${Date.now()}`,
          role: "user",
          text: content,
          done: true,
        },
      ],
    }));
    startStream((onEvent) => sendMessage(sessionId, content, onEvent));
  }, [input, running, suspended, sessionId, setRunState, startStream]);

  const resume = useCallback(
    (decision: ResumeDecision) => {
      onApprovalsConsumed(runState.approvals);
      startStream((onEvent) => resumeSession(sessionId, decision, onEvent));
    },
    [sessionId, startStream, onApprovalsConsumed, runState.approvals],
  );

  return (
    <section className="chat-panel">
      <div className="chat-status">
        会话 <code>{sessionId.slice(0, 8)}</code> · 状态：
        <span
          className={`status status-${runState.status}`}
          data-testid="run-status"
          data-status={runState.status}
        >
          {STATUS_LABEL[runState.status]}
        </span>
        {running && (
          <button className="link" onClick={() => abortRef.current?.()}>
            断开流（计算后台继续）
          </button>
        )}
      </div>

      <MessageList items={runState.items} />

      {runState.failure && (
        <div className="banner error">运行失败：{runState.failure}</div>
      )}
      {sendError && <div className="banner error">{sendError}</div>}

      {suspended && runState.approvals.length > 0 && (
        <ApprovalDialog approvals={runState.approvals} onDecision={resume} />
      )}

      <div className="chat-input">
        <textarea
          data-testid="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) send();
          }}
          placeholder={
            suspended
              ? "会话待审批，请先处理上方审批卡"
              : "描述计算任务，例如：用 ANI-2x 优化水分子并算频率（Ctrl+Enter 发送）"
          }
          rows={3}
          disabled={running || suspended}
        />
        <button
          className="primary"
          data-testid="send-button"
          onClick={send}
          disabled={running || suspended || !input.trim()}
        >
          发送
        </button>
      </div>
    </section>
  );
}
