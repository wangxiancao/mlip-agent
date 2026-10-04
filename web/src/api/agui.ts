/**
 * AG-UI 事件类型 + 流式事件 → UI 状态的 reducer。
 *
 * 线格式契约（app/agent/streaming.py + ag-ui-protocol 1.0.0，camelCase 别名）：
 *   RUN_STARTED / RUN_FINISHED          {threadId, runId}
 *   TEXT_MESSAGE_START/CONTENT/END      {messageId, role?, delta?}
 *   TOOL_CALL_START                     {toolCallId, toolCallName}
 *   TOOL_CALL_ARGS                      {toolCallId, delta}   ← args 增量文本
 *   TOOL_CALL_END                       {toolCallId}
 *   TOOL_CALL_RESULT                    {toolCallId, content} ← content 是 ToolResult JSON 字符串
 *   SUBAGENT_STARTED/FINISHED           {subagentRunId, name?, ...}
 *   MLIP_RUN_RESULT（合成终止事件）      {status, output, deferred, failure, ...} snake_case
 */

export interface PendingApproval {
  tool_call_id: string;
  tool_name: string;
  args: string | null;
}

export interface RunResultEvent {
  type: "MLIP_RUN_RESULT";
  thread_id?: string;
  run_id?: string;
  status: "completed" | "suspended" | "failed" | "cancelled";
  output?: string | null;
  state_json?: string | null;
  deferred?: { approvals?: PendingApproval[]; calls?: PendingApproval[] } | null;
  deferred_json?: string | null;
  failure?: { message?: string; [k: string]: unknown } | null;
}

/** 原始 AG-UI 事件（宽松形态；具体字段按 type 取）。 */
export interface AguiEvent {
  type: string;
  [k: string]: unknown;
}

export type RunStatus =
  | "idle"
  | "running"
  | "completed"
  | "suspended"
  | "failed"
  | "cancelled";

export interface TextItem {
  kind: "text";
  id: string;
  role: string;
  text: string;
  done: boolean;
}

export interface ToolItem {
  kind: "tool";
  id: string;
  toolCallId: string;
  toolName: string;
  argsText: string;
  argsDone: boolean;
  /** TOOL_CALL_RESULT 的原始 content（通常是 ToolResult JSON 字符串）。 */
  result: string | null;
}

export interface SubagentItem {
  kind: "subagent";
  id: string;
  name: string;
  done: boolean;
}

export type TimelineItem = TextItem | ToolItem | SubagentItem;

export interface RunUiState {
  status: RunStatus;
  items: TimelineItem[];
  approvals: PendingApproval[];
  /** run 结束时 agent 的最终文本输出（MLIP_RUN_RESULT.output）。 */
  output: string | null;
  failure: string | null;
  runId: string | null;
}

export function createRunState(): RunUiState {
  return {
    status: "idle",
    items: [],
    approvals: [],
    output: null,
    failure: null,
    runId: null,
  };
}

function patchItem(
  items: TimelineItem[],
  id: string,
  patch: Partial<TimelineItem>,
): TimelineItem[] {
  return items.map((it) => (it.id === id ? ({ ...it, ...patch } as TimelineItem) : it));
}

/**
 * 把一个 AG-UI 事件折叠进 UI 状态。返回新对象（不原地修改），供 React setState 使用。
 * 未知事件类型原样忽略，保证向前兼容。
 */
export function reduceEvent(state: RunUiState, event: AguiEvent): RunUiState {
  switch (event.type) {
    case "RUN_STARTED":
      return {
        ...state,
        status: "running",
        runId: (event.runId as string) ?? null,
        approvals: [],
        failure: null,
        output: null,
      };

    case "TEXT_MESSAGE_START": {
      const id = event.messageId as string;
      if (state.items.some((it) => it.id === id)) return state;
      const item: TextItem = {
        kind: "text",
        id,
        role: (event.role as string) ?? "assistant",
        text: "",
        done: false,
      };
      return { ...state, items: [...state.items, item] };
    }

    case "TEXT_MESSAGE_CONTENT": {
      const id = event.messageId as string;
      const delta = (event.delta as string) ?? "";
      const exists = state.items.some((it) => it.id === id);
      const items = exists
        ? patchItem(state.items, id, {})
        : state.items;
      return {
        ...state,
        items: items.map((it) =>
          it.id === id && it.kind === "text"
            ? { ...it, text: it.text + delta }
            : it,
        ),
      };
    }

    case "TEXT_MESSAGE_END":
      return {
        ...state,
        items: patchItem(state.items, event.messageId as string, {
          done: true,
        } as Partial<TimelineItem>),
      };

    case "TOOL_CALL_START": {
      const toolCallId = event.toolCallId as string;
      if (state.items.some((it) => it.id === toolCallId)) return state;
      const item: ToolItem = {
        kind: "tool",
        id: toolCallId,
        toolCallId,
        toolName:
          (event.toolCallName as string) ?? (event.toolName as string) ?? "tool",
        argsText: "",
        argsDone: false,
        result: null,
      };
      return { ...state, items: [...state.items, item] };
    }

    case "TOOL_CALL_ARGS": {
      const id = event.toolCallId as string;
      return {
        ...state,
        items: state.items.map((it) =>
          it.id === id && it.kind === "tool"
            ? { ...it, argsText: it.argsText + ((event.delta as string) ?? "") }
            : it,
        ),
      };
    }

    case "TOOL_CALL_END":
      return {
        ...state,
        items: state.items.map((it) =>
          it.id === (event.toolCallId as string) && it.kind === "tool"
            ? { ...it, argsDone: true }
            : it,
        ),
      };

    case "TOOL_CALL_RESULT":
      return {
        ...state,
        items: state.items.map((it) =>
          it.id === (event.toolCallId as string) && it.kind === "tool"
            ? { ...it, result: (event.content as string) ?? null }
            : it,
        ),
      };

    case "SUBAGENT_STARTED": {
      const id = event.subagentRunId as string;
      if (state.items.some((it) => it.id === id)) return state;
      const item: SubagentItem = {
        kind: "subagent",
        id,
        name: (event.name as string) ?? "subagent",
        done: false,
      };
      return { ...state, items: [...state.items, item] };
    }

    case "SUBAGENT_FINISHED":
      return {
        ...state,
        items: patchItem(state.items, event.subagentRunId as string, {
          done: true,
        } as Partial<TimelineItem>),
      };

    case "RUN_FINISHED":
      // 终态以 MLIP_RUN_RESULT 为准，这里不置状态（suspended 时 observer 也会发 RUN_FINISHED）
      return state;

    case "MLIP_RUN_RESULT": {
      const e = event as unknown as RunResultEvent;
      const failureMsg = e.failure
        ? (e.failure.message as string) ?? JSON.stringify(e.failure)
        : null;
      return {
        ...state,
        status: e.status,
        output: e.output ?? null,
        approvals: e.deferred?.approvals ?? [],
        failure: failureMsg,
      };
    }

    default:
      return state;
  }
}

/* ------------------------------------------------------------------ */
/* 工具结果解析：从 TOOL_CALL_RESULT content 提取结构化数据供结果面板使用 */

export interface ToolResultPayload {
  ok: boolean;
  data?: Record<string, unknown> | null;
  error?: string | null;
}

export function parseToolResult(content: string | null): ToolResultPayload | null {
  if (!content) return null;
  try {
    const obj = JSON.parse(content);
    if (obj && typeof obj === "object" && "ok" in obj) {
      return obj as ToolResultPayload;
    }
    // run_nequip 等可能直接返回 data 字典
    return { ok: true, data: obj as Record<string, unknown> };
  } catch {
    return null;
  }
}

/** 从工具结果 data 中归纳出的作业摘要（MAPLE parsed JobResult / NequIP result.json）。 */
export interface JobSummary {
  toolCallId: string;
  toolName: string;
  jobId?: string;
  jobType?: string;
  /** Hartree（MAPLE 路径）。 */
  energyHartree?: number | null;
  /** eV（NequIP 直算路径）。 */
  energyEv?: number | null;
  /** opt 每步 / scan 每点 / md 每步能量（Hartree）。 */
  energiesHartree: number[];
  scanValuesAngstrom: number[];
  frequenciesCm1: number[];
  nImaginary?: number | null;
  converged?: boolean | null;
  zpeKcalMol?: number | null;
  warnings: string[];
  artifacts: string[];
}

function asNumberArray(v: unknown): number[] {
  return Array.isArray(v) ? v.filter((x): x is number => typeof x === "number") : [];
}

/** 从时间线的工具项中提取所有可展示的作业摘要。 */
export function extractJobSummaries(items: TimelineItem[]): JobSummary[] {
  const out: JobSummary[] = [];
  for (const it of items) {
    if (it.kind !== "tool" || !it.result) continue;
    const payload = parseToolResult(it.result);
    const data = payload?.data;
    if (!data) continue;

    // MAPLE 路径：run_maple 的 data.parsed 是 JobResult dump；parse_maple_out 的 data 即 JobResult
    const parsed =
      (data.parsed as Record<string, unknown> | undefined) ??
      (typeof data.job_type === "string" &&
      ("energy_hartree" in data || "frequencies_cm1" in data)
        ? data
        : undefined);
    if (parsed) {
      out.push({
        toolCallId: it.toolCallId,
        toolName: it.toolName,
        jobId: (data.job_id as string) ?? undefined,
        jobType: (parsed.job_type as string) ?? (data.job_type as string),
        energyHartree: (parsed.energy_hartree as number) ?? null,
        energiesHartree: asNumberArray(parsed.energies_hartree),
        scanValuesAngstrom: asNumberArray(parsed.scan_values_angstrom),
        frequenciesCm1: asNumberArray(parsed.frequencies_cm1),
        nImaginary: (parsed.n_imaginary as number) ?? null,
        converged: (parsed.converged as boolean) ?? null,
        zpeKcalMol: (parsed.zpe_kcal_mol as number) ?? null,
        warnings: Array.isArray(parsed.warnings)
          ? (parsed.warnings as string[])
          : [],
        artifacts: asStringArray(data.artifacts ?? parsed.artifacts),
      });
      continue;
    }

    // NequIP 直算路径：data 即 result.json（energy 单位见 units 字段，标准 ASE eV）
    if (typeof data.energy === "number") {
      const units = (data.units as Record<string, string> | undefined)?.energy;
      const isEv = !units || units.toLowerCase() === "ev";
      out.push({
        toolCallId: it.toolCallId,
        toolName: it.toolName,
        jobId: (data.job_id as string) ?? undefined,
        jobType: (data.job_type as string) ?? "sp",
        energyHartree: isEv ? null : (data.energy as number),
        energyEv: isEv ? (data.energy as number) : null,
        energiesHartree: [],
        scanValuesAngstrom: [],
        frequenciesCm1: [],
        warnings: [],
        artifacts: asStringArray(data.artifacts),
      });
    }
  }
  return out;
}

function asStringArray(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
}

/** 能量单位换算。 */
export const HARTREE_TO_EV = 27.211386245988;
export const HARTREE_TO_KCAL = 627.5094740631;
