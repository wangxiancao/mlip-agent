/**
 * UI benchmark 浏览器驱动（benchmarks/ui_runner.py 的 Node 侧执行器）。
 *
 * 用 Playwright（系统 Chrome，channel: 'chrome'）驱动真实前端页面完成一条任务：
 *   新开浏览器 context（隔离 localStorage → 每任务独立会话）
 *   → 打开 base-url，等前端自动建会话
 *   → textarea 填 prompt，点发送
 *   → 循环：出现审批卡 → 记录挂起（含 jobs 目录快照）→ 填 auto_answer → 点提交
 *   → 直到 MLIP_RUN_RESULT 进入终态 / 超时
 *
 * SSE 事件经 page.addInitScript 包装 window.fetch 旁路 tee 录制到
 * window.__mlipEvents（不改前端代码，run_record.json 的 events/tool_calls 取自真实事件流）。
 *
 * 用法：node ui_driver.mjs <config.json>
 *   config: { baseUrl, prompt, autoAnswer, dataDir, resultPath, headed, timeoutS }
 * 结果写入 resultPath：{ session_id, events, suspensions, final, error, timed_out }
 * 进程退出码恒 0（错误写入结果 JSON，由 ui_runner.py 判定）。
 */

import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const require = createRequire(new URL("../web/package.json", import.meta.url));
const { chromium } = require("playwright");

const RUN_RESULT = "MLIP_RUN_RESULT";
const POLL_MS = 250;

/** fetch 包装：SSE 响应 tee 一路出来解析 AG-UI 事件存 window.__mlipEvents。 */
const INIT_SCRIPT = `(() => {
  window.__mlipEvents = [];
  const origFetch = window.fetch;
  window.fetch = async (...args) => {
    const res = await origFetch(...args);
    const ct = res.headers.get("content-type") || "";
    if (!ct.includes("text/event-stream") || !res.body) return res;
    const [mine, theirs] = res.body.tee();
    (async () => {
      const reader = mine.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      const flush = (frame) => {
        const lines = [];
        for (const line of frame.split("\\n")) {
          if (line.startsWith(":")) return;
          if (line.startsWith("data:")) lines.push(line.slice(5).replace(/^ /, ""));
        }
        if (!lines.length) return;
        const text = lines.join("\\n");
        if (!text.trim()) return;
        try { window.__mlipEvents.push(JSON.parse(text)); } catch { /* ignore */ }
      };
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buffer.indexOf("\\n\\n")) >= 0) {
          flush(buffer.slice(0, idx));
          buffer = buffer.slice(idx + 2);
        }
      }
      buffer += decoder.decode();
      if (buffer.trim()) flush(buffer);
    })().catch(() => {});
    return new Response(theirs, res);
  };
})();`;

function snapshotJobs(dataDir, sessionId) {
  if (!dataDir || !sessionId) return [];
  const jobsDir = path.join(dataDir, "sessions", sessionId, "jobs");
  if (!fs.existsSync(jobsDir)) return [];
  return fs
    .readdirSync(jobsDir, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name)
    .sort()
    .map((name) => ({
      job_id: name,
      files: fs
        .readdirSync(path.join(jobsDir, name), { withFileTypes: true })
        .filter((f) => f.isFile())
        .map((f) => f.name)
        .sort(),
    }));
}

async function main() {
  const config = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
  const {
    baseUrl,
    prompt,
    autoAnswer = null,
    dataDir = null,
    resultPath,
    headed = false,
    timeoutS = 1800,
  } = config;

  const result = {
    session_id: null,
    events: [],
    suspensions: [],
    final: null,
    error: null,
    timed_out: false,
  };
  const deadline = Date.now() + timeoutS * 1000;

  const browser = await chromium.launch({ channel: "chrome", headless: !headed });
  try {
    const context = await browser.newContext();
    await context.addInitScript(INIT_SCRIPT);
    const page = await context.newPage();
    page.on("pageerror", (err) => console.error(`[pageerror] ${err.message}`));

    await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
    const input = page.getByTestId("chat-input");
    await input.waitFor({ state: "visible", timeout: 30000 });
    // 前端就绪 = 会话已建（localStorage 写入）且输入框可用
    await page.waitForFunction(
      () => !!localStorage.getItem("mlip_agent_session_id"),
      null,
      { timeout: 30000 },
    );
    result.session_id = await page.evaluate(() =>
      localStorage.getItem("mlip_agent_session_id"),
    );
    console.log(`[driver] session=${result.session_id}`);

    await input.fill(prompt);
    await page.getByTestId("send-button").click();
    console.log("[driver] prompt sent");

    let handledSuspensions = 0;
    for (;;) {
      if (Date.now() > deadline) {
        result.timed_out = true;
        result.error = `UI 驱动超时（${timeoutS} s）`;
        break;
      }
      const events = await page.evaluate(() => window.__mlipEvents);
      result.events = events;
      const results = events.filter((e) => e.type === RUN_RESULT);
      const final = results.length ? results[results.length - 1] : null;
      result.final = final;

      if (final && final.status !== "suspended") break;

      const dialog = page.getByTestId("approval-dialog");
      if (
        final &&
        final.status === "suspended" &&
        results.filter((e) => e.status === "suspended").length >
          handledSuspensions &&
        (await dialog.isVisible().catch(() => false))
      ) {
        const deferred = final.deferred || {};
        const suspension = {
          approvals: deferred.approvals || [],
          calls: deferred.calls || [],
          jobs_snapshot: snapshotJobs(dataDir, result.session_id),
        };
        result.suspensions.push(suspension);
        handledSuspensions += 1;
        console.log(
          `[driver] suspension #${handledSuspensions}: approvals=[${suspension.approvals
            .map((a) => a.tool_name)
            .join(",")}]`,
        );
        // ask_user_question 项填 auto_answer（缺省给通用答复）
        const items = page.getByTestId("approval-item");
        const n = await items.count();
        for (let i = 0; i < n; i++) {
          const item = items.nth(i);
          const toolName = (await item.getAttribute("data-tool-name")) || "";
          if (!toolName.endsWith("ask_user_question")) continue;
          const answerInput = item.getByTestId("answer-input");
          if (await answerInput.isVisible().catch(() => false)) {
            await answerInput.fill(
              autoAnswer ?? "按你认为最合适的默认方案继续（电荷 0，自旋多重度 1）。",
            );
          }
        }
        await page.getByTestId("approval-submit").click();
        await dialog.waitFor({ state: "hidden", timeout: 10000 }).catch(() => {});
        continue;
      }
      await page.waitForTimeout(POLL_MS);
    }

    if (!result.timed_out && !result.final) {
      result.error = "事件流中无 MLIP_RUN_RESULT";
    }
    const statusText = await page
      .getByTestId("run-status")
      .getAttribute("data-status")
      .catch(() => null);
    result.ui_status = statusText;
    console.log(
      `[driver] done: final=${result.final ? result.final.status : "none"} ui_status=${statusText} suspensions=${result.suspensions.length}`,
    );
  } catch (err) {
    result.error = `${err.name}: ${err.message}`;
  } finally {
    await browser.close().catch(() => {});
    fs.writeFileSync(resultPath, JSON.stringify(result, null, 2) + "\n");
  }
}

main().catch((err) => {
  console.error(`[driver] fatal: ${err.stack || err}`);
  process.exit(0);
});
