import { useState } from "react";
import type { PendingApproval } from "../api/agui";
import type { ResumeDecision } from "../api/client";

interface Props {
  approvals: PendingApproval[];
  onDecision: (decision: ResumeDecision) => void;
}

function isAskUser(toolName: string): boolean {
  return toolName.endsWith("ask_user_question");
}

function prettyArgs(args: string | null): string {
  if (!args) return "（无参数）";
  try {
    return JSON.stringify(JSON.parse(args), null, 2);
  } catch {
    return args;
  }
}

/** 从 ask_user_question 的 args 里提取问题文本用于展示。 */
function questionText(args: string | null): string | null {
  if (!args) return null;
  try {
    const obj = JSON.parse(args) as Record<string, unknown>;
    return (obj.question as string) ?? (obj.prompt as string) ?? null;
  } catch {
    return null;
  }
}

/**
 * 审批对话框（PLAN §2.6）：run 挂起时展示待审批工具调用。
 * - 普通审批工具（run_maple/run_nequip）：批准 / 拒绝（可填理由）。
 * - ask_user_question：渲染输入框，答复经 answers[tool_call_id] 注入。
 * 提交后走 POST /resume（approve_all=true，未显式拒绝的一律批准）。
 */
export default function ApprovalDialog({ approvals, onDecision }: Props) {
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [denials, setDenials] = useState<Record<string, string>>({});
  const [denyReasons, setDenyReasons] = useState<Record<string, string>>({});

  const toggleDeny = (id: string) => {
    setDenials((prev) => {
      const next = { ...prev };
      if (id in next) delete next[id];
      else next[id] = "";
      return next;
    });
  };

  const submit = () => {
    const finalDenials: Record<string, string> = {};
    for (const id of Object.keys(denials)) {
      finalDenials[id] = denyReasons[id] || "用户拒绝";
    }
    const finalAnswers: Record<string, string> = {};
    for (const a of approvals) {
      if (isAskUser(a.tool_name) && !(a.tool_call_id in finalDenials)) {
        finalAnswers[a.tool_call_id] = answers[a.tool_call_id] ?? "";
      }
    }
    onDecision({
      approve_all: true,
      answers: finalAnswers,
      denials: finalDenials,
    });
  };

  return (
    <div className="approval-dialog" data-testid="approval-dialog">
      <h3>⚠ 运行已挂起，等待你的确认</h3>
      {approvals.map((a) => {
        const denied = a.tool_call_id in denials;
        const ask = isAskUser(a.tool_name);
        const question = ask ? questionText(a.args) : null;
        return (
          <div
            key={a.tool_call_id}
            className={`approval-item ${denied ? "denied" : ""}`}
            data-testid="approval-item"
            data-tool-name={a.tool_name}
          >            <div className="approval-head">
              <code>{a.tool_name}</code>
              <label>
                <input
                  type="checkbox"
                  checked={denied}
                  onChange={() => toggleDeny(a.tool_call_id)}
                />
                拒绝此项
              </label>
            </div>
            {ask && question && <p className="approval-question">{question}</p>}
            <details>
              <summary>调用参数</summary>
              <pre>{prettyArgs(a.args)}</pre>
            </details>
            {ask && !denied && (
              <input
                className="answer-input"
                data-testid="answer-input"
                type="text"
                placeholder="输入你的答复（如：电荷 0，自旋多重度 1，用 ANI-2x）"
                value={answers[a.tool_call_id] ?? ""}
                onChange={(e) =>
                  setAnswers((prev) => ({ ...prev, [a.tool_call_id]: e.target.value }))
                }
              />
            )}
            {denied && (
              <input
                className="answer-input"
                type="text"
                placeholder="拒绝理由（可选）"
                value={denyReasons[a.tool_call_id] ?? ""}
                onChange={(e) =>
                  setDenyReasons((prev) => ({
                    ...prev,
                    [a.tool_call_id]: e.target.value,
                  }))
                }
              />
            )}
          </div>
        );
      })}
      <button className="primary" data-testid="approval-submit" onClick={submit}>
        提交决定（未拒绝的一律批准）
      </button>
    </div>
  );
}
