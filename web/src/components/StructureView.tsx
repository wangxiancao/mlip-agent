import { useEffect, useRef, useState } from "react";
import type { GLViewer } from "3dmol";

interface Props {
  /** xyz / extxyz 工件 URL（GET 返回文本）。 */
  url: string;
  title?: string;
}

/**
 * 3Dmol.js 分子 3D 结构查看器（M6）。
 * 挂载时抓取 xyz 文本并渲染；多帧轨迹（extxyz）只显示首帧。
 * WebGL 不可用或工件下载失败时降级为错误提示，不影响其余面板。
 */
export default function StructureView({ url, title }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let viewer: GLViewer | null = null;
    setError(null);
    (async () => {
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`工件下载失败（HTTP ${res.status}）`);
        const text = await res.text();
        // 按需加载 3dmol（约 700KB），保持主包精简
        const { createViewer } = await import("3dmol");
        if (cancelled || !hostRef.current) return;
        viewer = createViewer(hostRef.current, { backgroundColor: "white" });
        const model = viewer.addModel(text, "xyz");
        if (model.getNumFrames() > 1) await model.setFrame(0);
        viewer.setStyle({}, { stick: { radius: 0.12 }, sphere: { scale: 0.25 } });
        viewer.zoomTo();
        viewer.render();
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      }
    })();
    return () => {
      cancelled = true;
      viewer?.clear();
      hostRef.current?.replaceChildren();
    };
  }, [url]);

  return (
    <div className="structure-view">
      {title && <div className="structure-title">{title}</div>}
      {error ? (
        <p className="warn-text small">3D 结构渲染失败：{error}</p>
      ) : (
        <div ref={hostRef} className="structure-canvas" />
      )}
    </div>
  );
}
