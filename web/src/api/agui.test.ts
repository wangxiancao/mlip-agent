import { describe, expect, it } from "vitest";
import {
  createRunState,
  extractJobSummaries,
  reduceEvent,
  type AguiEvent,
  type RunUiState,
} from "./agui";

function runAll(events: AguiEvent[]): RunUiState {
  return events.reduce(reduceEvent, createRunState());
}

describe("reduceEvent", () => {
  it("流式文本：START/CONTENT/END 拼成完整气泡", () => {
    const s = runAll([
      { type: "RUN_STARTED", threadId: "t1", runId: "r1" },
      { type: "TEXT_MESSAGE_START", messageId: "m1", role: "assistant" },
      { type: "TEXT_MESSAGE_CONTENT", messageId: "m1", delta: "好的，" },
      { type: "TEXT_MESSAGE_CONTENT", messageId: "m1", delta: "我来计算。" },
      { type: "TEXT_MESSAGE_END", messageId: "m1" },
      { type: "RUN_FINISHED", threadId: "t1", runId: "r1" },
      {
        type: "MLIP_RUN_RESULT",
        thread_id: "t1",
        run_id: "r1",
        status: "completed",
        output: "好的，我来计算。",
      },
    ]);
    expect(s.status).toBe("completed");
    expect(s.runId).toBe("r1");
    expect(s.items).toHaveLength(1);
    const msg = s.items[0];
    expect(msg.kind).toBe("text");
    if (msg.kind === "text") {
      expect(msg.text).toBe("好的，我来计算。");
      expect(msg.done).toBe(true);
    }
  });

  it("工具调用：START/ARGS/END/RESULT 组装成一张工具卡", () => {
    const s = runAll([
      { type: "RUN_STARTED", threadId: "t", runId: "r" },
      { type: "TOOL_CALL_START", toolCallId: "tc1", toolCallName: "write_maple_inp" },
      { type: "TOOL_CALL_ARGS", toolCallId: "tc1", delta: '{"job_spec":' },
      { type: "TOOL_CALL_ARGS", toolCallId: "tc1", delta: "{}}" },
      { type: "TOOL_CALL_END", toolCallId: "tc1" },
      {
        type: "TOOL_CALL_RESULT",
        messageId: "tc1:result",
        toolCallId: "tc1",
        content: '{"ok":true,"data":{"inp_path":"/tmp/x.inp"}}',
      },
      {
        type: "MLIP_RUN_RESULT",
        status: "completed",
        output: null,
      },
    ]);
    const tool = s.items[0];
    expect(tool.kind).toBe("tool");
    if (tool.kind === "tool") {
      expect(tool.toolName).toBe("write_maple_inp");
      expect(tool.argsText).toBe('{"job_spec":{}}');
      expect(tool.argsDone).toBe(true);
      expect(tool.result).toContain('"ok":true');
    }
  });

  it("suspended：MLIP_RUN_RESULT.deferred.approvals 变成待审批列表", () => {
    const s = runAll([
      { type: "RUN_STARTED", threadId: "t", runId: "r" },
      { type: "TOOL_CALL_START", toolCallId: "tc9", toolCallName: "run_maple" },
      { type: "TOOL_CALL_ARGS", toolCallId: "tc9", delta: '{"inp_path":"a.inp"}' },
      { type: "TOOL_CALL_END", toolCallId: "tc9" },
      { type: "RUN_FINISHED", threadId: "t", runId: "r" },
      {
        type: "MLIP_RUN_RESULT",
        status: "suspended",
        output: null,
        deferred: {
          approvals: [
            {
              tool_call_id: "tc9",
              tool_name: "run_maple",
              args: '{"inp_path":"a.inp"}',
            },
          ],
          calls: [],
        },
      },
    ]);
    expect(s.status).toBe("suspended");
    expect(s.approvals).toHaveLength(1);
    expect(s.approvals[0].tool_name).toBe("run_maple");
  });

  it("未知事件类型被忽略", () => {
    const s0 = createRunState();
    const s1 = reduceEvent(s0, { type: "STATE_SNAPSHOT", snapshot: {} });
    expect(s1).toBe(s0);
  });

  it("RUN_STARTED 清空上一轮 approvals/failure", () => {
    let s = runAll([
      {
        type: "MLIP_RUN_RESULT",
        status: "suspended",
        deferred: {
          approvals: [{ tool_call_id: "x", tool_name: "run_maple", args: null }],
        },
      },
    ]);
    s = reduceEvent(s, { type: "RUN_STARTED", threadId: "t", runId: "r2" });
    expect(s.status).toBe("running");
    expect(s.approvals).toHaveLength(0);
  });
});

describe("extractJobSummaries", () => {
  it("从 run_maple 结果提取能量/频率/收敛曲线数据", () => {
    const content = JSON.stringify({
      ok: true,
      data: {
        job_id: "job-abc",
        job_type: "opt",
        energy_hartree: -76.4,
        converged: true,
        parsed: {
          job_type: "opt",
          energy_hartree: -76.4,
          energies_hartree: [-76.39, -76.4],
          converged: true,
          frequencies_cm1: [1595.0, 3657.0, 3757.0],
          n_imaginary: 0,
          warnings: [],
        },
      },
    });
    const s = runAll([
      { type: "TOOL_CALL_START", toolCallId: "tc1", toolCallName: "run_maple" },
      { type: "TOOL_CALL_END", toolCallId: "tc1" },
      { type: "TOOL_CALL_RESULT", toolCallId: "tc1", content },
    ]);
    const summaries = extractJobSummaries(s.items);
    expect(summaries).toHaveLength(1);
    const j = summaries[0];
    expect(j.jobId).toBe("job-abc");
    expect(j.jobType).toBe("opt");
    expect(j.energyHartree).toBeCloseTo(-76.4);
    expect(j.energiesHartree).toEqual([-76.39, -76.4]);
    expect(j.frequenciesCm1).toHaveLength(3);
    expect(j.converged).toBe(true);
  });

  it("从 run_nequip 结果（eV 单位）提取能量", () => {
    const content = JSON.stringify({
      ok: true,
      data: {
        job_id: "job-neq",
        energy: -13.43729751141953,
        units: { energy: "eV" },
      },
    });
    const s = runAll([
      { type: "TOOL_CALL_START", toolCallId: "tc2", toolCallName: "run_nequip" },
      { type: "TOOL_CALL_END", toolCallId: "tc2" },
      { type: "TOOL_CALL_RESULT", toolCallId: "tc2", content },
    ]);
    const summaries = extractJobSummaries(s.items);
    expect(summaries[0].energyEv).toBeCloseTo(-13.43729751141953);
    expect(summaries[0].energyHartree).toBeNull();
  });

  it("非 JSON 结果不产生摘要", () => {
    const s = runAll([
      { type: "TOOL_CALL_START", toolCallId: "tc3", toolCallName: "x" },
      { type: "TOOL_CALL_END", toolCallId: "tc3" },
      { type: "TOOL_CALL_RESULT", toolCallId: "tc3", content: "plain text" },
    ]);
    expect(extractJobSummaries(s.items)).toHaveLength(0);
  });
});
