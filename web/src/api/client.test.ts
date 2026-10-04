import { describe, expect, it } from "vitest";
import { parseSseStream } from "./client";
import type { AguiEvent } from "./agui";

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c));
      controller.close();
    },
  });
}

async function collect(chunks: string[]): Promise<AguiEvent[]> {
  const events: AguiEvent[] = [];
  await parseSseStream(streamOf(chunks), (e) => events.push(e));
  return events;
}

describe("parseSseStream", () => {
  it("按空行切帧，解析 data: JSON，忽略 :hb 注释", async () => {
    const events = await collect([
      'data: {"type":"RUN_STARTED","threadId":"t","runId":"r"}\n\n',
      ":hb\n\n",
      'data: {"type":"TEXT_MESSAGE_CONTENT","messageId":"m1","delta":"你好"}\n\n',
      ":hb\n:hb\n\n",
      'data: {"type":"MLIP_RUN_RESULT","status":"completed"}\n\n',
    ]);
    expect(events.map((e) => e.type)).toEqual([
      "RUN_STARTED",
      "TEXT_MESSAGE_CONTENT",
      "MLIP_RUN_RESULT",
    ]);
  });

  it("跨 chunk 的帧正确拼接", async () => {
    const full = 'data: {"type":"TEXT_MESSAGE_CONTENT","messageId":"m","delta":"abc"}\n\n';
    const events = await collect([full.slice(0, 17), full.slice(17, 40), full.slice(40)]);
    expect(events).toHaveLength(1);
    expect(events[0].delta).toBe("abc");
  });

  it("末尾无空行的残留帧也会处理", async () => {
    const events = await collect(['data: {"type":"RUN_FINISHED","runId":"r"}']);
    expect(events.map((e) => e.type)).toEqual(["RUN_FINISHED"]);
  });

  it("多行 data: 合并为一帧", async () => {
    const events = await collect([
      'data: {"type":"X",\ndata: "v": 1}\n\n',
    ]);
    expect(events).toHaveLength(1);
    expect((events[0] as Record<string, unknown>).v).toBe(1);
  });

  it("非 JSON 帧抛错（坏数据不静默吞掉）", async () => {
    await expect(collect(["data: not-json\n\n"])).rejects.toThrow();
  });
});

describe("pickStructureArtifact", () => {
  it("优先 structure.xyz", async () => {
    const { pickStructureArtifact } = await import("./client");
    expect(
      pickStructureArtifact(["input.inp", "final.xyz", "structure.xyz"]),
    ).toBe("structure.xyz");
  });

  it("无 structure.xyz 时取其他 .xyz，再次 .extxyz 轨迹", async () => {
    const { pickStructureArtifact } = await import("./client");
    expect(pickStructureArtifact(["input.out", "final.xyz"])).toBe("final.xyz");
    expect(pickStructureArtifact(["result.json", "traj.extxyz"])).toBe("traj.extxyz");
    expect(pickStructureArtifact(["input.inp", "input.out"])).toBeNull();
    expect(pickStructureArtifact([])).toBeNull();
  });
});
