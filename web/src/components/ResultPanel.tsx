import { useMemo, useState } from "react";
import {
  artifactUrl,
  pickStructureArtifact,
  type JobInfo,
} from "../api/client";
import {
  extractJobSummaries,
  HARTREE_TO_EV,
  HARTREE_TO_KCAL,
  type RunUiState,
} from "../api/agui";
import ConvergenceChart from "./ConvergenceChart";
import StructureView from "./StructureView";

interface Props {
  runState: RunUiState;
  jobs: JobInfo[];
}

const JOB_TYPE_LABEL: Record<string, string> = {
  sp: "单点能",
  opt: "几何优化",
  freq: "频率分析",
  scan: "扫描",
  md: "分子动力学",
};

export default function ResultPanel({ runState, jobs }: Props) {
  const [open3d, setOpen3d] = useState<Record<string, boolean>>({});
  const summaries = useMemo(
    () => extractJobSummaries(runState.items),
    [runState.items],
  );

  // 工具结果里有 job_id 的作业 + 快照里的 jobs 取并集
  const knownJobs = useMemo(() => {
    const map = new Map<string, JobInfo>();
    for (const j of jobs) map.set(j.job_id, j);
    for (const s of summaries) {
      if (s.jobId && !map.has(s.jobId)) {
        map.set(s.jobId, {
          job_id: s.jobId,
          session_id: "",
          job_type: s.jobType ?? null,
          artifacts: s.artifacts,
          has_output: s.artifacts.includes("output.out"),
        });
      }
    }
    return [...map.values()];
  }, [jobs, summaries]);

  return (
    <aside className="result-panel">
      <h2>计算结果</h2>

      {summaries.length === 0 && knownJobs.length === 0 && (
        <p className="empty-hint">
          完成计算后，这里会显示能量表、频率表、收敛曲线和工件下载。
        </p>
      )}

      {summaries.map((s, i) => (
        <div key={`${s.toolCallId}-${i}`} className="result-card">
          <h3>
            {JOB_TYPE_LABEL[s.jobType ?? ""] ?? s.jobType ?? "结果"}
            {s.jobId && <code className="job-id">{s.jobId}</code>}
          </h3>

          {(s.energyHartree != null || s.energyEv != null) && (
            <table className="energy-table">
              <thead>
                <tr>
                  <th>Hartree</th>
                  <th>eV</th>
                  <th>kcal·mol⁻¹</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  {s.energyHartree != null ? (
                    <>
                      <td>{s.energyHartree.toFixed(10)}</td>
                      <td>{(s.energyHartree * HARTREE_TO_EV).toFixed(6)}</td>
                      <td>{(s.energyHartree * HARTREE_TO_KCAL).toFixed(4)}</td>
                    </>
                  ) : (
                    <>
                      <td>{(s.energyEv! / HARTREE_TO_EV).toFixed(10)}</td>
                      <td>{s.energyEv!.toFixed(6)}</td>
                      <td>{((s.energyEv! / HARTREE_TO_EV) * HARTREE_TO_KCAL).toFixed(4)}</td>
                    </>
                  )}
                </tr>
              </tbody>
            </table>
          )}

          {s.converged != null && (
            <p className={s.converged ? "ok-text" : "warn-text"}>
              {s.converged ? "✓ 优化已收敛" : "✗ 优化未收敛"}
            </p>
          )}

          {s.frequenciesCm1.length > 0 && (
            <>
              <h4>振动频率（cm⁻¹）</h4>
              <table className="freq-table">
                <tbody>
                  {s.frequenciesCm1.map((f, j) => (
                    <tr key={j}>
                      <td>ν{j + 1}</td>
                      <td className={f < 0 ? "warn-text" : ""}>{f.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {s.nImaginary != null && (
                <p className={s.nImaginary > 0 ? "warn-text" : "ok-text"}>
                  虚频：{s.nImaginary} 个
                  {s.zpeKcalMol != null && ` · ZPE = ${s.zpeKcalMol.toFixed(2)} kcal·mol⁻¹`}
                </p>
              )}
            </>
          )}

          {s.jobType === "opt" && s.energiesHartree.length > 1 && (
            <ConvergenceChart
              points={s.energiesHartree.map((y, j) => ({ x: j + 1, y }))}
              xLabel="优化步数"
              yLabel="能量 (Hartree)"
            />
          )}

          {s.jobType === "scan" && s.energiesHartree.length > 0 && (
            <ConvergenceChart
              points={s.energiesHartree.map((y, j) => ({
                x: s.scanValuesAngstrom[j] ?? j + 1,
                y,
              }))}
              xLabel="扫描坐标 (Å)"
              yLabel="能量 (Hartree)"
            />
          )}

          {s.jobType === "md" && s.energiesHartree.length > 1 && (
            <ConvergenceChart
              points={s.energiesHartree.map((y, j) => ({ x: j + 1, y }))}
              xLabel="MD 步数"
              yLabel="总能量 (Hartree)"
            />
          )}

          {s.warnings.length > 0 && (
            <p className="warn-text small">解析警告：{s.warnings.join("；")}</p>
          )}
        </div>
      ))}

      {knownJobs.length > 0 && (
        <div className="result-card">
          <h3>作业工件</h3>
          {knownJobs.map((j) => {
            const structArtifact = pickStructureArtifact(j.artifacts);
            const show3d = structArtifact !== null && (open3d[j.job_id] ?? false);
            return (
              <div key={j.job_id} className="job-row">
                <div className="job-head">
                  <code>{j.job_id}</code>
                  {j.job_type && (
                    <span className="badge">{JOB_TYPE_LABEL[j.job_type] ?? j.job_type}</span>
                  )}
                  {structArtifact && (
                    <button
                      className="link-btn"
                      onClick={() =>
                        setOpen3d((m) => ({ ...m, [j.job_id]: !m[j.job_id] }))
                      }
                    >
                      {show3d ? "收起 3D 结构" : "3D 结构"}
                    </button>
                  )}
                </div>
                {show3d && (
                  <StructureView
                    url={artifactUrl(j.job_id, structArtifact)}
                    title={`${j.job_id} · ${structArtifact}`}
                  />
                )}
                <div className="artifact-links">
                  {(j.artifacts.length > 0
                    ? j.artifacts
                    : ["input.inp", "output.out", "traj.extxyz"]
                  ).map((name) => (
                    <a key={name} href={artifactUrl(j.job_id, name)} download>
                      {name}
                    </a>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </aside>
  );
}
