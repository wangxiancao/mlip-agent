import { useEffect, useRef } from "react";
import type { TimelineItem } from "../api/agui";
import ToolCallCard from "./ToolCallCard";

interface Props {
  items: TimelineItem[];
}

export default function MessageList({ items }: Props) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [items]);

  return (
    <div className="message-list">
      {items.length === 0 && (
        <div className="empty-hint">
          还没有消息。可以用自然语言描述计算任务，例如：
          <ul>
            <li>用 ANI-2x 计算水分子的单点能</li>
            <li>优化乙醇结构，然后做频率分析</li>
            <li>扫描水分子一个 O-H 键从 0.9 到 1.1 Å，5 个点</li>
          </ul>
        </div>
      )}
      {items.map((item) => {
        if (item.kind === "text") {
          const cls =
            item.role === "user" ? "msg user" : item.role === "tool" ? "msg tool" : "msg assistant";
          return (
            <div key={item.id} className={cls}>
              <div className="msg-role">
                {item.role === "user" ? "你" : item.role === "tool" ? "工具" : "助手"}
              </div>
              <div className="msg-body">
                {item.text}
                {!item.done && <span className="cursor">▍</span>}
              </div>
            </div>
          );
        }
        if (item.kind === "tool") {
          return <ToolCallCard key={item.id} item={item} />;
        }
        // subagent
        return (
          <div key={item.id} className="msg subagent">
            <span className={`dot ${item.done ? "done" : "running"}`} />
            子代理 <code>{item.name}</code> {item.done ? "已完成" : "运行中…"}
          </div>
        );
      })}
      <div ref={endRef} />
    </div>
  );
}
