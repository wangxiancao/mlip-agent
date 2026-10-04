import { useMemo, useState } from "react";
import {
  HARTREE_TO_EV,
  parseToolResult,
  type ToolItem,
} from "../api/agui";

interface Props {
  item: ToolItem;
}

/** 美化 args：尝试 JSON.parse 后缩进展示，失败则原文。 */
function pretty(text: string): string {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

/** 从 run_maple / parse_maple_out 结果里抽一行摘要。 */
function summarize(result: string): string | null {
  const payload = parseToolResult(result);
  if (!payload) return null;
  if (!payload.ok) return payload.error ? `错误：${payload.error}` : "工具返回失败";
  const d = payload.data ?? {};
  const parsed = (d.parsed ?? d) as Record<string, unknown>;
  const parts: string[] = [];
  if (typeof d.job_id === "string") parts.push(`作业 ${d.job_id}`);
  if (typeof parsed.job_type === "string") parts.push(`类型 ${parsed.job_type}`);
  if (typeof parsed.energy_hartree === "number") {
    parts.push(
      `E = ${parsed.energy_hartree.toFixed(8)} Ha (${(parsed.energy_hartree * HARTREE_TO_EV).toFixed(4)} eV)`,
    );
  } else if (typeof d.energy === "number") {
    parts.push(`E = ${d.energy} ${(d.units as Record<string, string>)?.energy ?? "eV"}`);
  }
  if (parsed.converged === true) parts.push("已收敛");
  if (parsed.converged === false) parts.push("未收敛");
  if (typeof parsed.n_imaginary === "number") parts.push(`虚频 ${parsed.n_imaginary} 个`);
  return parts.length ? parts.join(" · ") : null;
}

export default function ToolCallCard({ item }: Props) {
  const [open, setOpen] = useState(false);
  const running = !item.argsDone;
  const waiting = item.argsDone && item.result === null;
  const summary = useMemo(
    () => (item.result ? summarize(item.result) : null),
    [item.result],
  );

  return (
    <div className={`tool-card ${running || waiting ? "active" : ""}`}>
      <button className="tool-head" onClick={() => setOpen((v) => !v)}>
        <span className={`dot ${running || waiting ? "running" : item.result !== null ? "done" : ""}`} />
        <code className="tool-name">{item.toolName}</code>
        <span className="tool-state">
          {running ? "参数生成中…" : waiting ? "执行中…" : "完成"}
        </span>
        {summary && <span className="tool-summary">{summary}</span>}
        <span className="chevron">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <div className="tool-detail">
          <h4>调用参数</h4>
          <pre>{item.argsText ? pretty(item.argsText) : "（无）"}</pre>
          {item.result !== null && (
            <>
              <h4>返回结果</h4>
              <pre>{pretty(item.result)}</pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}
