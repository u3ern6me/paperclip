import { describe, expect, it, vi, afterEach } from "vitest";
import type { AdapterExecutionContext } from "@paperclipai/adapter-utils";
import { execute, mapFinalResultForTest, parseSseFramesForTest, resolveSessionKey } from "./execute.js";
import { sessionCodec } from "./index.js";
import { testEnvironment } from "./test.js";

function makeCtx(config: Record<string, unknown>): AdapterExecutionContext {
  return {
    runId: "pc-run-1",
    agent: {
      id: "agent-1",
      companyId: "company-1",
      name: "Hermes",
      adapterType: "hermes_gateway",
      adapterConfig: config,
    },
    runtime: {
      sessionId: null,
      sessionParams: null,
      sessionDisplayId: null,
      taskKey: null,
    },
    config,
    context: {
      issueId: "issue-1",
      wakeReason: "manual",
      paperclipWake: {
        issue: { identifier: "PAP-1", title: "Do the thing" },
      },
    },
    onLog: vi.fn(async () => undefined),
    onMeta: vi.fn(async () => undefined),
  };
}

function sseStream(text: string): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text));
      controller.close();
    },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("resolveSessionKey", () => {
  it("derives issue-scoped session keys by default", () => {
    expect(
      resolveSessionKey({
        strategy: "issue",
        companyId: "company-1",
        agentId: "agent-1",
        runId: "run-1",
        issueId: "issue-1",
      }),
    ).toBe("paperclip:company:company-1:agent:agent-1:issue:issue-1");
  });

  it("omits the session key for none strategy", () => {
    expect(
      resolveSessionKey({
        strategy: "none",
        companyId: "company-1",
        agentId: "agent-1",
        runId: "run-1",
        issueId: "issue-1",
      }),
    ).toBeNull();
  });
});

describe("parseSseFramesForTest", () => {
  it("parses event and data lines while preserving partial frames", () => {
    const parsed = parseSseFramesForTest("event: message.delta\ndata: {\"delta\":\"hi\"}\n\n:data\ndata: later");
    expect(parsed.frames).toEqual([{ event: "message.delta", data: "{\"delta\":\"hi\"}" }]);
    expect(parsed.rest).toBe(":data\ndata: later");
  });
});

describe("execute", () => {
  it("rejects remote plain HTTP unless the unsafe dev escape hatch is enabled", async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ run_id: "unexpected" }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await execute(makeCtx({
      apiBaseUrl: "http://192.168.1.25:8642",
      apiKey: "secret-key",
    }));

    expect(result.exitCode).toBe(1);
    expect(result.errorCode).toBe("hermes_gateway_plain_http_remote_denied");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("reports dispatch before starting the remote run create request", async () => {
    const ctx = makeCtx({
      apiBaseUrl: "http://127.0.0.1:8642",
      apiKey: "secret-key",
      timeoutSec: 5,
    });
    const onDispatch = vi.fn();
    ctx.onDispatch = onDispatch;
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/v1/runs")) {
        expect(onDispatch).toHaveBeenCalledTimes(1);
        return new Response(JSON.stringify({ run_id: "run-hermes-1", status: "started" }), { status: 200 });
      }
      if (url.endsWith("/events")) {
        return new Response(
          sseStream(["event: run.completed", "data: {\"status\":\"completed\",\"output\":\"done\"}", ""].join("\n")),
          { status: 200, headers: { "content-type": "text/event-stream" } },
        );
      }
      return new Response(JSON.stringify({ status: "completed", output: "done" }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await execute(ctx);

    expect(result.exitCode).toBe(0);
    expect(onDispatch).toHaveBeenCalledTimes(1);
  });

  it("constructs POST /v1/runs with auth, idempotency, and Hermes session headers", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/v1/runs")) {
        return new Response(JSON.stringify({ run_id: "run-hermes-1", status: "started" }), { status: 200 });
      }
      if (url.endsWith("/events")) {
        return new Response(
          sseStream(
            [
              "event: message.delta",
              "data: {\"delta\":\"done\"}",
              "",
              "event: run.completed",
              "data: {\"status\":\"completed\",\"output\":\"done\",\"session_id\":\"session-1\",\"usage\":{\"input_tokens\":3,\"output_tokens\":2},\"model\":\"hermes-agent\"}",
              "",
            ].join("\n"),
          ),
          { status: 200, headers: { "content-type": "text/event-stream" } },
        );
      }
      return new Response(JSON.stringify({ status: "completed", output: "done" }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await execute(makeCtx({
      apiBaseUrl: "http://127.0.0.1:8642",
      apiKey: "secret-key",
      timeoutSec: 5,
    }));

    expect(result.exitCode).toBe(0);
    expect(result.summary).toBe("done");
    expect(result.usage).toEqual({ inputTokens: 3, outputTokens: 2 });

    const calls = fetchMock.mock.calls as Array<[RequestInfo | URL, RequestInit?]>;
    const createCall = calls.find(([input]) => String(input).endsWith("/v1/runs"));
    expect(createCall).toBeTruthy();
    const init = createCall?.[1] as RequestInit;
    expect(init.headers).toMatchObject({
      Authorization: "Bearer secret-key",
      "Content-Type": "application/json",
      "Idempotency-Key": "pc-run-1",
      "X-Hermes-Session-Key": "paperclip:company:company-1:agent:agent-1:issue:issue-1",
    });
    const body = JSON.parse(String(init.body));
    expect(body.input).toContain("Do the thing");
    expect(body.session_id).toBe("paperclip:company:company-1:agent:agent-1:issue:issue-1");
  });

  it.each([false, true])("preserves chat handoff policy on gateway turns (resumed=%s)", async (resumed) => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => new Response(JSON.stringify(
      String(input).endsWith("/v1/runs")
        ? { run_id: "run-hermes-1", status: "started" }
        : { status: "completed", output: "done" },
    ), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const ctx = makeCtx({ apiBaseUrl: "http://127.0.0.1:8642", apiKey: "secret-key", timeoutSec: 5 });
    ctx.config.payloadTemplate = { input: "Custom gateway instruction." };
    const directive = "Chat directive: clarify goals and hand plans off to project tasks.";
    ctx.context = {
      conversationMode: true,
      issueId: "issue-1",
      paperclipTaskMarkdown: directive,
      paperclipTaskMarkdownCompact: directive,
      paperclipWake: {
        reason: "issue_commented",
        issue: { id: "issue-1", workMode: "planning", status: "in_progress" },
        interactionKind: "request_confirmation",
        interactionStatus: "accepted",
      },
    };
    if (resumed) ctx.runtime.sessionId = "prior-session";
    await execute(ctx);
    const calls = fetchMock.mock.calls as Array<[RequestInfo | URL, RequestInit?]>;
    const call = calls.find(([input]) => String(input).endsWith("/v1/runs"));
    const prompt = JSON.parse(String(call?.[1]?.body)).input as string;
    expect(prompt).toContain("Custom gateway instruction.");
    expect(prompt).toContain(directive);
    expect(prompt).not.toContain("Execution contract:");
    expect(prompt).not.toContain("clear final disposition");
    expect(prompt).not.toContain("Create child issues");
  });

  it("sends the task brief once on fresh runs and compacts it on stable-session resumes", async () => {
    const description = "Update launch-card.svg and change the CTA to Try Team free.";
    const fullTaskMarkdown = [
      "Paperclip task context:",
      '- Issue: "PAP-1"',
      "",
      "Issue description:",
      "```text",
      description,
      "```",
    ].join("\n");
    const compactTaskMarkdown = ["Paperclip task context:", '- Issue: "PAP-1"'].join("\n");
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/v1/runs")) {
        return new Response(JSON.stringify({ run_id: "run-hermes-1", status: "started" }), { status: 200 });
      }
      return new Response(JSON.stringify({ status: "completed", output: "done" }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const wakeContext = (reason: string) => ({
      issueId: "issue-1",
      wakeReason: reason,
      paperclipTaskMarkdown: fullTaskMarkdown,
      paperclipTaskMarkdownCompact: compactTaskMarkdown,
      paperclipWake: {
        reason,
        issue: {
          id: "issue-1",
          identifier: "PAP-1",
          title: "Do the thing",
          description,
          descriptionTruncated: false,
          status: "in_progress",
        },
        commentWindow: { requestedCount: 0, includedCount: 0, missingCount: 0 },
        comments: [],
        fallbackFetchNeeded: false,
      },
    });

    const freshCtx = makeCtx({ apiBaseUrl: "http://127.0.0.1:8642", apiKey: "secret-key", timeoutSec: 5 });
    freshCtx.context = wakeContext("issue_assigned");
    await execute(freshCtx);

    const resumeCtx = makeCtx({ apiBaseUrl: "http://127.0.0.1:8642", apiKey: "secret-key", timeoutSec: 5 });
    resumeCtx.context = wakeContext("issue_commented");
    resumeCtx.runtime = {
      sessionId: "session-1",
      sessionParams: null,
      sessionDisplayId: "session-1",
      taskKey: "PAP-1",
    };
    await execute(resumeCtx);

    const calls = fetchMock.mock.calls as Array<[RequestInfo | URL, RequestInit?]>;
    const runBodies = calls
      .filter(([input]) => String(input).endsWith("/v1/runs"))
      .map(([, init]) => JSON.parse(String(init?.body)) as { input: string });
    expect(runBodies).toHaveLength(2);
    // Fresh run: brief exactly once (task markdown only; wake-prompt copy suppressed).
    expect(runBodies[0]!.input.split(description)).toHaveLength(2);
    // Stable-session resume: compact task markdown, no re-sent brief.
    expect(runBodies[1]!.input).toContain("Paperclip task context:");
    expect(runBodies[1]!.input).not.toContain(description);
  });

  it("routes a bare Hermes dashboard URL on port 9119 through the API prefix", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "http://127.0.0.1:9119/api/v1/runs") {
        return new Response(JSON.stringify({ run_id: "run-hermes-1", status: "started" }), { status: 200 });
      }
      if (url === "http://127.0.0.1:9119/api/v1/runs/run-hermes-1/events") {
        return new Response(
          sseStream(
            [
              "event: run.completed",
              "data: {\"status\":\"completed\",\"output\":\"done\"}",
              "",
            ].join("\n"),
          ),
          { status: 200, headers: { "content-type": "text/event-stream" } },
        );
      }
      return new Response(JSON.stringify({ status: "completed", output: "done" }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const ctx = makeCtx({
      apiBaseUrl: "http://127.0.0.1:9119",
      apiKey: "secret-key",
      timeoutSec: 5,
    });
    const result = await execute(ctx);

    expect(result.exitCode).toBe(0);
    expect(ctx.onMeta).toHaveBeenCalledWith(
      expect.objectContaining({
        commandArgs: ["http://127.0.0.1:9119/api/v1/runs"],
      }),
    );
    expect((ctx.onLog as ReturnType<typeof vi.fn>).mock.calls.map(([, line]) => String(line)).join("\n"))
      .toContain("creating run at http://127.0.0.1:9119/api/v1/runs");
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual(
      expect.arrayContaining([
        "http://127.0.0.1:9119/api/v1/runs",
        "http://127.0.0.1:9119/api/v1/runs/run-hermes-1/events",
      ]),
    );
  });

  it("routes the default Hermes dashboard chat URL on port 9119 through the API prefix", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "http://127.0.0.1:9119/api/v1/runs") {
        return new Response(JSON.stringify({ run_id: "run-hermes-chat", status: "started" }), { status: 200 });
      }
      if (url === "http://127.0.0.1:9119/api/v1/runs/run-hermes-chat/events") {
        return new Response(
          sseStream(
            [
              "event: run.completed",
              "data: {\"status\":\"completed\",\"output\":\"done\"}",
              "",
            ].join("\n"),
          ),
          { status: 200, headers: { "content-type": "text/event-stream" } },
        );
      }
      return new Response(JSON.stringify({ status: "completed", output: "done" }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const ctx = makeCtx({
      apiBaseUrl: "http://127.0.0.1:9119/chat",
      apiKey: "secret-key",
      timeoutSec: 5,
    });
    const result = await execute(ctx);

    expect(result.exitCode).toBe(0);
    expect(ctx.onMeta).toHaveBeenCalledWith(
      expect.objectContaining({
        commandArgs: ["http://127.0.0.1:9119/api/v1/runs"],
      }),
    );
    expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual(
      expect.arrayContaining([
        "http://127.0.0.1:9119/api/v1/runs",
        "http://127.0.0.1:9119/api/v1/runs/run-hermes-chat/events",
      ]),
    );
  });

  it("redacts echoed auth material from stream logs and summaries", async () => {
    const ctx = makeCtx({
      apiBaseUrl: "http://127.0.0.1:8642",
      apiKey: "secret-key",
      timeoutSec: 5,
    });
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/v1/runs")) {
        return new Response(JSON.stringify({ run_id: "run-hermes-1", status: "started" }), { status: 200 });
      }
      if (url.endsWith("/events")) {
        return new Response(
          sseStream(
            [
              "event: message.delta",
              "data: {\"delta\":\"Authorization: Bearer secret-key\\nX-Hermes-Session-Key: paperclip:company:company-1:agent:agent-1:issue:issue-1\"}",
              "",
              "event: run.completed",
              "data: {\"status\":\"completed\",\"output\":\"Authorization: Bearer secret-key\\nraw key secret-key\\nX-Hermes-Session-Key: paperclip:company:company-1:agent:agent-1:issue:issue-1\"}",
              "",
            ].join("\n"),
          ),
          { status: 200, headers: { "content-type": "text/event-stream" } },
        );
      }
      return new Response(JSON.stringify({ status: "completed" }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await execute(ctx);
    const logText = (ctx.onLog as ReturnType<typeof vi.fn>).mock.calls.map(([, line]) => String(line)).join("\n");

    expect(result.exitCode).toBe(0);
    expect(result.summary).toContain("Bearer [redacted]");
    expect(result.summary).toContain("raw key [redacted len=10]");
    expect(result.summary).toContain("X-Hermes-Session-Key: [redacted]");
    expect(result.summary).not.toContain("secret-key");
    expect(result.summary).not.toContain("paperclip:company:company-1:agent:agent-1:issue:issue-1");
    expect(result.resultJson?.output).toBe(result.summary);
    expect(logText).toContain("Bearer [redacted]");
    expect(logText).toContain("X-Hermes-Session-Key: [redacted]");
    expect(logText).not.toContain("secret-key");
    expect(logText).not.toContain("paperclip:company:company-1:agent:agent-1:issue:issue-1");
  });

  it("redacts agent-scoped Paperclip session keys from logs and public result metadata", async () => {
    const ctx = makeCtx({
      apiBaseUrl: "http://127.0.0.1:8642",
      apiKey: "secret-key",
      sessionKeyStrategy: "agent",
      timeoutSec: 5,
    });
    const agentSessionKey = "paperclip:company:company-1:agent:agent-1";
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/v1/runs")) {
        return new Response(JSON.stringify({ run_id: "run-hermes-1", status: "started" }), { status: 200 });
      }
      if (url.endsWith("/events")) {
        return new Response(
          sseStream(
            [
              "event: message.delta",
              `data: {"delta":"session ${agentSessionKey}"}`,
              "",
              "event: run.completed",
              `data: {"status":"completed","output":"session ${agentSessionKey}","session_id":"${agentSessionKey}"}`,
              "",
            ].join("\n"),
          ),
          { status: 200, headers: { "content-type": "text/event-stream" } },
        );
      }
      return new Response(JSON.stringify({ status: "completed" }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await execute(ctx);
    const logText = (ctx.onLog as ReturnType<typeof vi.fn>).mock.calls.map(([, line]) => String(line)).join("\n");

    expect(result.exitCode).toBe(0);
    expect(result.summary).toBe("session [redacted-session-key]");
    // Guard C: the redaction placeholder is NOT a session id — it must not be
    // echoed or persisted as sessionId/sessionDisplayId.
    expect(result.sessionId).toBeNull();
    expect(result.sessionDisplayId).toBeNull();
    expect(result.resultJson?.session_id).toBeNull();
    expect(result.sessionParams).toEqual({
      hermesRunId: "run-hermes-1",
      strategy: "agent",
    });
    expect(logText).toContain("[redacted-session-key]");
    expect(logText).not.toContain(agentSessionKey);
  });

  it("falls back to polling when SSE is unavailable", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/v1/runs")) {
        return new Response(JSON.stringify({ run_id: "run-hermes-1", status: "started" }), { status: 200 });
      }
      if (url.endsWith("/events")) {
        return new Response("no stream", { status: 503 });
      }
      return new Response(JSON.stringify({
        status: "completed",
        output: "polled done",
        session_id: "session-polled",
      }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await execute(makeCtx({
      apiBaseUrl: "http://127.0.0.1:8642",
      apiKey: "secret-key",
      timeoutSec: 5,
      pollIntervalMs: 250,
    }));

    expect(result.exitCode).toBe(0);
    expect(result.summary).toBe("polled done");
    expect(fetchMock.mock.calls.some(([input]) => String(input).endsWith("/v1/runs/run-hermes-1"))).toBe(true);
  });

  it("maps HTTP auth failures", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ error: "bad key" }), { status: 401 })));
    const result = await execute(makeCtx({
      apiBaseUrl: "http://127.0.0.1:8642",
      apiKey: "secret-key",
    }));
    expect(result.exitCode).toBe(1);
    expect(result.errorCode).toBe("hermes_gateway_auth_failed");
    expect(result.errorMessage).toContain("Check adapterConfig.apiKey matches the Hermes API_SERVER_KEY");
  });

  it("includes network causes in connection failure messages", async () => {
    const cause = Object.assign(new Error("getaddrinfo ENOTFOUND host.docker.internal"), { code: "ENOTFOUND" });
    vi.stubGlobal("fetch", vi.fn(async () => {
      throw Object.assign(new Error("fetch failed"), { cause });
    }));

    const result = await execute(makeCtx({
      apiBaseUrl: "http://host.docker.internal:8642",
      apiKey: "secret-key",
      dangerouslyAllowInsecureRemoteHttp: true,
    }));

    expect(result.exitCode).toBe(1);
    expect(result.errorCode).toBe("hermes_gateway_connect_failed");
    expect(result.errorMessage).toContain("ENOTFOUND");
    expect(result.errorMessage).toContain("host.docker.internal");
  });

  it("redacts echoed auth material from HTTP error payloads", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(
          JSON.stringify({
            message: "Authorization rejected: Bearer secret-key raw secret-key",
            detail: "X-Hermes-Session-Key: paperclip:company:company-1:agent:agent-1:issue:issue-1",
            nested: {
              note: "session paperclip:company:company-1:agent:agent-1",
            },
          }),
          { status: 401 },
        )),
    );

    const result = await execute(makeCtx({
      apiBaseUrl: "http://127.0.0.1:8642",
      apiKey: "secret-key",
    }));

    expect(result.exitCode).toBe(1);
    expect(result.errorCode).toBe("hermes_gateway_auth_failed");
    expect(result.errorMeta?.body).toEqual({
      message: "Authorization rejected: Bearer [redacted] raw [redacted len=10]",
      detail: "X-Hermes-Session-Key: [redacted]",
      nested: {
        note: "session [redacted-session-key]",
      },
    });
    expect(result.errorMessage).not.toContain("secret-key");
    expect(result.errorMessage).not.toContain("paperclip:company:company-1:agent:agent-1:issue:issue-1");
  });

  it("calls stop on timeout", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/v1/runs")) {
        return new Response(JSON.stringify({ run_id: "run-slow", status: "started" }), { status: 200 });
      }
      if (url.endsWith("/events")) {
        return new Promise<Response>(() => {});
      }
      if (url.endsWith("/stop")) {
        return new Response(JSON.stringify({ status: "stopping" }), { status: 200 });
      }
      if (init?.method === "GET") {
        return new Response(JSON.stringify({ status: "cancelled", last_event: "run.cancelled" }), { status: 200 });
      }
      return new Response(JSON.stringify({ status: "running" }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await execute(makeCtx({
      apiBaseUrl: "http://127.0.0.1:8642",
      apiKey: "secret-key",
      timeoutSec: 0.001,
    }));

    expect(result.timedOut).toBe(true);
    expect(result.errorCode).toBe("hermes_gateway_timeout");
    expect(fetchMock.mock.calls.some(([input]) => String(input).endsWith("/stop"))).toBe(true);
  });
});

describe("testEnvironment", () => {
  it("fails remote plain HTTP before probing health", async () => {
    const fetchMock = vi.fn(async () => new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await testEnvironment({
      companyId: "company-1",
      adapterType: "hermes_gateway",
      config: {
        apiBaseUrl: "http://hermes.example:8642",
        apiKey: "secret-key",
      },
    });

    expect(result.status).toBe("fail");
    expect(result.checks).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "hermes_gateway_plain_http_remote_denied",
          level: "error",
        }),
      ]),
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("allows remote plain HTTP only with the unsafe dev escape hatch", async () => {
    const fetchMock = vi.fn(async () => new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await testEnvironment({
      companyId: "company-1",
      adapterType: "hermes_gateway",
      config: {
        apiBaseUrl: "http://hermes.example:8642",
        apiKey: "secret-key",
        dangerouslyAllowInsecureRemoteHttp: true,
      },
    });

    expect(result.status).toBe("warn");
    expect(result.checks).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "hermes_gateway_plain_http_remote_unsafe_allowed",
          level: "warn",
        }),
        expect.objectContaining({
          code: "hermes_gateway_health_ok",
        }),
      ]),
    );
    expect(fetchMock).toHaveBeenCalled();
  });

  it("fails test environment checks when Hermes health is unreachable", async () => {
    const cause = Object.assign(new Error("getaddrinfo ENOTFOUND host.docker.internal"), { code: "ENOTFOUND" });
    vi.stubGlobal("fetch", vi.fn(async () => {
      throw Object.assign(new Error("fetch failed"), { cause });
    }));

    const result = await testEnvironment({
      companyId: "company-1",
      adapterType: "hermes_gateway",
      config: {
        apiBaseUrl: "http://host.docker.internal:8642",
        apiKey: "secret-key",
        dangerouslyAllowInsecureRemoteHttp: true,
      },
    });

    expect(result.status).toBe("fail");
    expect(result.checks).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "hermes_gateway_health_unreachable",
          level: "error",
          detail: expect.stringContaining("ENOTFOUND"),
        }),
      ]),
    );
  });

  it("fails test environment checks when Hermes health returns a non-ok status", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("bad key", { status: 401 })));

    const result = await testEnvironment({
      companyId: "company-1",
      adapterType: "hermes_gateway",
      config: {
        apiBaseUrl: "http://127.0.0.1:8642",
        apiKey: "wrong-key",
      },
    });

    expect(result.status).toBe("fail");
    expect(result.checks).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "hermes_gateway_health_failed",
          level: "error",
          message: "Hermes Gateway health endpoint returned HTTP 401.",
        }),
      ]),
    );
  });

  it("tests a bare Hermes dashboard URL on port 9119 through the API prefix", async () => {
    const fetchMock = vi.fn(async () => new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await testEnvironment({
      companyId: "company-1",
      adapterType: "hermes_gateway",
      config: {
        apiBaseUrl: "http://127.0.0.1:9119",
        apiKey: "secret-key",
      },
    });

    expect(result.status).toBe("pass");
    expect(result.checks).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "hermes_gateway_dashboard_root_mapped",
          level: "info",
          message: "Default Hermes dashboard root mapped to API base http://127.0.0.1:9119/api.",
          hint: expect.stringContaining("/api/v1/runs"),
        }),
      ]),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:9119/api/health",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("tests a Hermes dashboard chat URL on port 9119 through the API prefix", async () => {
    const fetchMock = vi.fn(async () => new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await testEnvironment({
      companyId: "company-1",
      adapterType: "hermes_gateway",
      config: {
        apiBaseUrl: "http://127.0.0.1:9119/chat",
        apiKey: "secret-key",
      },
    });

    expect(result.status).toBe("pass");
    expect(result.checks).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "hermes_gateway_dashboard_root_mapped",
          level: "info",
          message: "Default Hermes dashboard root mapped to API base http://127.0.0.1:9119/api.",
        }),
      ]),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:9119/api/health",
      expect.objectContaining({ method: "GET" }),
    );
  });
});

describe("mapFinalResultForTest", () => {
  it("maps failed statuses into adapter errors", () => {
    const result = mapFinalResultForTest({
      terminal: {
        runId: "run-1",
        status: "failed",
        payload: { status: "failed", error: "boom" },
      },
      outputChunks: [],
      sessionKey: "session-key",
      strategy: "issue",
    });
    expect(result.exitCode).toBe(1);
    expect(result.errorCode).toBe("hermes_gateway_run_failed");
    expect(result.errorMessage).toBe("boom");
  });

  it("treats only a terminal payload session id as a confirmed tip, preserving the prior tip otherwise", () => {
    const previous = { hermesRunId: "run-old", hermesSessionId: "tip-old", strategy: "issue" };

    // No session id in the terminal payload: the prior tip is preserved, and
    // the Paperclip session key fallback must not be recorded as a tip.
    const noTip = mapFinalResultForTest({
      terminal: { runId: "run-1", status: "completed", payload: { status: "completed" } },
      outputChunks: [],
      sessionKey: "paperclip:company:c:agent:a:issue:i",
      strategy: "issue",
      existingSessionId: "canon-session",
      previousSessionParams: previous,
    });
    expect(noTip.sessionParams).toEqual({
      hermesRunId: "run-1",
      hermesSessionId: "tip-old",
      hermesCanonicalSessionId: "canon-session",
      strategy: "issue",
    });

    // A Hermes-confirmed session id in the terminal payload becomes the tip.
    const confirmed = mapFinalResultForTest({
      terminal: { runId: "run-1", status: "completed", payload: { status: "completed", session_id: "tip-new" } },
      outputChunks: [],
      sessionKey: "paperclip:company:c:agent:a:issue:i",
      strategy: "issue",
      existingSessionId: "canon-session",
      previousSessionParams: previous,
    });
    expect(confirmed.sessionParams).toEqual({
      hermesRunId: "run-1",
      hermesSessionId: "tip-new",
      hermesCanonicalSessionId: "canon-session",
      strategy: "issue",
    });
  });
});

describe("sessionCodec", () => {
  it("round-trips gateway session params including the canonical session id", () => {
    const params = {
      hermesRunId: "run-1",
      hermesSessionId: "20260101_120000_abcd",
      hermesCanonicalSessionId: "20260101_110000_parent",
      strategy: "issue",
    };
    const serialized = sessionCodec.serialize(params);
    expect(serialized).toEqual(params);
    expect(sessionCodec.deserialize(serialized)).toEqual(params);
    // deserialize also accepts the legacy alias shape.
    expect(sessionCodec.deserialize({ sessionId: "legacy-id", hermesRunId: "run-1" })).toEqual({
      hermesRunId: "run-1",
      hermesSessionId: "legacy-id",
    });
  });

  it("keeps unknown keys out while preserving legacy field compatibility", () => {
    const serialized = sessionCodec.serialize({
      hermesRunId: "run-1",
      strategy: "issue",
      __paperclipConfigCategories: ["adapter"],
      cwd: "/tmp/workspace",
    });
    expect(serialized).toEqual({ hermesRunId: "run-1", strategy: "issue" });
    expect(sessionCodec.deserialize(serialized)).toEqual(serialized);
  });

  it("returns null for empty params and keeps canonical-only params", () => {
    expect(sessionCodec.serialize({})).toBeNull();
    expect(sessionCodec.deserialize(null)).toBeNull();
    expect(sessionCodec.deserialize("nope")).toBeNull();
    expect(sessionCodec.serialize({ hermesCanonicalSessionId: "canon-session" })).toEqual({
      hermesCanonicalSessionId: "canon-session",
    });
  });
});

describe("execute with existingHermesSessionId", () => {
  const CANONICAL = "20260101_120000_e2tg";

  function hermesRunMock(): { fetchMock: ReturnType<typeof vi.fn>; runBodies: Array<Record<string, unknown>> } {
    const runBodies: Array<Record<string, unknown>> = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/v1/runs")) {
        runBodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
        return new Response(JSON.stringify({ run_id: "run-hermes-1", status: "started" }), { status: 200 });
      }
      if (url.endsWith("/events")) {
        return new Response(
          sseStream(
            [
              "event: run.completed",
              `data: ${JSON.stringify({ status: "completed", output: "done", session_id: "tip-confirmed" })}`,
              "",
            ].join("\n"),
          ),
          { status: 200, headers: { "content-type": "text/event-stream" } },
        );
      }
      return new Response(JSON.stringify({ status: "completed", output: "done" }), { status: 200 });
    });
    return { fetchMock, runBodies };
  }

  it("sends the selected Hermes session id as session_id while keeping Paperclip headers unchanged", async () => {
    const { fetchMock, runBodies } = hermesRunMock();
    vi.stubGlobal("fetch", fetchMock);

    const ctx = makeCtx({
      apiBaseUrl: "http://127.0.0.1:8642",
      apiKey: "secret-key",
      existingHermesSessionId: CANONICAL,
      timeoutSec: 5,
    });
    ctx.runtime.sessionParams = {
      hermesRunId: "run-old",
      hermesSessionId: "tip-old",
      strategy: "issue",
    };
    const result = await execute(ctx);

    expect(result.exitCode).toBe(0);
    const createCall = (fetchMock.mock.calls as Array<[RequestInfo | URL, RequestInit?]>).find(([input]) =>
      String(input).endsWith("/v1/runs"),
    );
    const init = createCall?.[1] as RequestInit;
    const headers = init.headers as Record<string, string>;
    expect(JSON.parse(String(init.body)).session_id).toBe(CANONICAL);
    // The selected id must not become a credential, a header value, or a
    // Paperclip session key substitute.
    expect(headers.Authorization).toBe("Bearer secret-key");
    expect(headers["X-Hermes-Session-Key"]).toBe("paperclip:company:company-1:agent:agent-1:issue:issue-1");
    expect(Object.values(headers).join("\n")).not.toContain(CANONICAL);
    expect(ctx.onMeta).toHaveBeenCalledWith(
      expect.objectContaining({
        context: expect.objectContaining({ hasExistingHermesSessionId: true }),
      }),
    );

    // The confirmed tip comes from the terminal payload, not from the id we sent.
    expect(result.sessionParams).toEqual({
      hermesRunId: "run-hermes-1",
      hermesSessionId: "tip-confirmed",
      hermesCanonicalSessionId: CANONICAL,
      strategy: "issue",
    });
    // The selected id is not echoed into logs.
    const logText = (ctx.onLog as ReturnType<typeof vi.fn>).mock.calls.map(([, line]) => String(line)).join("\n");
    expect(logText).not.toContain(CANONICAL);
  });

  it("keeps default Paperclip addressing when existingHermesSessionId is empty or whitespace", async () => {
    for (const value of [undefined, "", "   "]) {
      const { fetchMock, runBodies } = hermesRunMock();
      vi.stubGlobal("fetch", fetchMock);

      const config: Record<string, unknown> = {
        apiBaseUrl: "http://127.0.0.1:8642",
        apiKey: "secret-key",
        timeoutSec: 5,
      };
      if (value !== undefined) config.existingHermesSessionId = value;
      const result = await execute(makeCtx(config));

      expect(result.exitCode).toBe(0);
      expect(runBodies[0]!.session_id).toBe("paperclip:company:company-1:agent:agent-1:issue:issue-1");
      expect(result.sessionParams).toEqual({
        hermesRunId: "run-hermes-1",
        hermesSessionId: "tip-confirmed",
        strategy: "issue",
      });
      vi.unstubAllGlobals();
    }
  });

  it("preserves the last confirmed tip when the terminal payload has no session id", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/v1/runs")) {
        return new Response(JSON.stringify({ run_id: "run-hermes-1", status: "started" }), { status: 200 });
      }
      if (url.endsWith("/events")) {
        // Terminal event WITHOUT a session id: nothing new is confirmed.
        return new Response(
          sseStream(["event: run.completed", 'data: {"status":"completed","output":"done"}', ""].join("\n")),
          { status: 200, headers: { "content-type": "text/event-stream" } },
        );
      }
      return new Response(JSON.stringify({ status: "completed" }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const ctx = makeCtx({
      apiBaseUrl: "http://127.0.0.1:8642",
      apiKey: "secret-key",
      existingHermesSessionId: CANONICAL,
      timeoutSec: 5,
    });
    ctx.runtime.sessionParams = {
      hermesRunId: "run-old",
      hermesSessionId: "tip-old",
      strategy: "issue",
    };
    const result = await execute(ctx);
    expect(result.exitCode).toBe(0);
    expect(result.sessionParams).toEqual({
      hermesRunId: "run-hermes-1",
      hermesSessionId: "tip-old",
      hermesCanonicalSessionId: CANONICAL,
      strategy: "issue",
    });
  });

  it("does not store a redacted echoed session id as the tip", async () => {
    const agentSessionKey = "paperclip:company:company-1:agent:agent-1";
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/v1/runs")) {
        return new Response(JSON.stringify({ run_id: "run-hermes-1", status: "started" }), { status: 200 });
      }
      if (url.endsWith("/events")) {
        return new Response(
          sseStream(
            [
              "event: run.completed",
              `data: ${JSON.stringify({ status: "completed", output: "done", session_id: agentSessionKey })}`,
              "",
            ].join("\n"),
          ),
          { status: 200, headers: { "content-type": "text/event-stream" } },
        );
      }
      return new Response(JSON.stringify({ status: "completed" }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const ctx = makeCtx({
      apiBaseUrl: "http://127.0.0.1:8642",
      apiKey: "secret-key",
      sessionKeyStrategy: "agent",
      existingHermesSessionId: CANONICAL,
      timeoutSec: 5,
    });
    ctx.runtime.sessionParams = { hermesRunId: "run-old", hermesSessionId: "tip-old", strategy: "agent" };
    const result = await execute(ctx);

    expect(result.exitCode).toBe(0);
    expect(result.sessionParams).toEqual({
      hermesRunId: "run-hermes-1",
      hermesSessionId: "tip-old",
      hermesCanonicalSessionId: CANONICAL,
      strategy: "agent",
    });
  });

  it("rejects a Paperclip session key as existingHermesSessionId before any request", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const result = await execute(makeCtx({
      apiBaseUrl: "http://127.0.0.1:8642",
      apiKey: "secret-key",
      existingHermesSessionId: "paperclip:company:company-1:agent:agent-1",
      timeoutSec: 5,
    }));

    expect(result.exitCode).toBe(1);
    expect(result.errorCode).toBe("hermes_gateway_existing_session_invalid");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("keeps the canonical id and prior tip across a timeout", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/v1/runs")) {
        return new Response(JSON.stringify({ run_id: "run-slow", status: "started" }), { status: 200 });
      }
      if (url.endsWith("/events")) {
        return new Promise<Response>(() => {});
      }
      if (url.endsWith("/stop")) {
        return new Response(JSON.stringify({ status: "stopping" }), { status: 200 });
      }
      if (init?.method === "GET") {
        return new Response(JSON.stringify({ status: "cancelled" }), { status: 200 });
      }
      return new Response(JSON.stringify({ status: "running" }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const ctx = makeCtx({
      apiBaseUrl: "http://127.0.0.1:8642",
      apiKey: "secret-key",
      existingHermesSessionId: CANONICAL,
      timeoutSec: 0.001,
    });
    ctx.runtime.sessionParams = { hermesRunId: "run-old", hermesSessionId: "tip-old", strategy: "issue" };
    const result = await execute(ctx);

    expect(result.timedOut).toBe(true);
    expect(result.sessionParams).toEqual({
      hermesRunId: "run-slow",
      hermesSessionId: "tip-old",
      hermesCanonicalSessionId: CANONICAL,
      strategy: "issue",
    });
  });

  it("keeps standard timeout session params unchanged without an explicit session", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/v1/runs")) {
        return new Response(JSON.stringify({ run_id: "run-slow", status: "started" }), { status: 200 });
      }
      if (url.endsWith("/events")) {
        return new Promise<Response>(() => {});
      }
      if (url.endsWith("/stop")) {
        return new Response(JSON.stringify({ status: "stopping" }), { status: 200 });
      }
      if (init?.method === "GET") {
        return new Response(JSON.stringify({ status: "cancelled" }), { status: 200 });
      }
      return new Response(JSON.stringify({ status: "running" }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await execute(makeCtx({
      apiBaseUrl: "http://127.0.0.1:8642",
      apiKey: "secret-key",
      timeoutSec: 0.001,
    }));

    expect(result.timedOut).toBe(true);
    expect(result.sessionParams).toEqual({ hermesRunId: "run-slow", strategy: "issue" });
  });

  it("uses the compact task brief when continuing an explicitly selected session", async () => {
    const description = "Update launch-card.svg and change the CTA to Try Team free.";
    const fullTaskMarkdown = [
      "Paperclip task context:",
      '- Issue: "PAP-1"',
      "",
      "Issue description:",
      "```text",
      description,
      "```",
    ].join("\n");
    const compactTaskMarkdown = ["Paperclip task context:", '- Issue: "PAP-1"'].join("\n");
    const { fetchMock, runBodies } = hermesRunMock();
    vi.stubGlobal("fetch", fetchMock);

    const ctx = makeCtx({
      apiBaseUrl: "http://127.0.0.1:8642",
      apiKey: "secret-key",
      existingHermesSessionId: CANONICAL,
      timeoutSec: 5,
    });
    ctx.context = {
      issueId: "issue-1",
      // A comment wake (not assignment-shaped): assignment-shaped reasons
      // deliberately receive the full brief even on resumed sessions.
      wakeReason: "issue_commented",
      paperclipTaskMarkdown: fullTaskMarkdown,
      paperclipTaskMarkdownCompact: compactTaskMarkdown,
      paperclipWake: {
        reason: "issue_commented",
        issue: {
          id: "issue-1",
          identifier: "PAP-1",
          title: "Do the thing",
          description,
          descriptionTruncated: false,
          status: "in_progress",
        },
        commentWindow: { requestedCount: 0, includedCount: 0, missingCount: 0 },
        comments: [],
        fallbackFetchNeeded: false,
      },
    };
    // No runtime.sessionId: only the explicit selection marks this as a resume.
    await execute(ctx);

    expect(runBodies).toHaveLength(1);
    expect(runBodies[0]!.session_id).toBe(CANONICAL);
    expect(runBodies[0]!.input).toContain("Paperclip task context:");
    expect(runBodies[0]!.input).not.toContain(description);
  });
});

describe("terminal SSE without session_id (authoritative status fill)", () => {
  const CANONICAL = "20261012_888888_e2par";
  const TIP = "20261012_888889_e2kid";

  it("fills the confirmed tip from GET /v1/runs when the terminal SSE frame omits session_id", async () => {
    const ctx = makeCtx({
      apiBaseUrl: "http://127.0.0.1:8642",
      apiKey: "k",
      existingHermesSessionId: CANONICAL,
      timeoutSec: 5,
      pollIntervalMs: 250,
    });
    let statusGets = 0;
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/v1/runs")) {
        return new Response(JSON.stringify({ run_id: "run-x", status: "started" }), { status: 200 });
      }
      if (url.endsWith("/events")) {
        // Real-contract shape: terminal SSE frame carries NO session_id.
        return new Response(
          sseStream(['event: run.failed', 'data: {"event":"run.failed","run_id":"run-x","error":"Connection error."}', ""].join("\n")),
          { status: 200, headers: { "content-type": "text/event-stream" } },
        );
      }
      if (/\/v1\/runs\/run-x$/.test(url)) {
        statusGets += 1;
        return new Response(
          JSON.stringify({ status: "failed", error: "Connection error.", session_id: TIP }),
          { status: 200 },
        );
      }
      return new Response(JSON.stringify({ status: "stopping" }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await execute(ctx);

    expect(statusGets).toBe(1);
    expect(result.exitCode).toBe(1);
    expect(result.sessionParams).toEqual({
      hermesRunId: "run-x",
      hermesSessionId: TIP,
      hermesCanonicalSessionId: CANONICAL,
      strategy: "issue",
    });
    expect(result.sessionDisplayId).toBe(TIP);
  });

  it("does not promote a tip when the authoritative status also lacks session_id", async () => {
    const ctx = makeCtx({
      apiBaseUrl: "http://127.0.0.1:8642",
      apiKey: "k",
      existingHermesSessionId: CANONICAL,
      timeoutSec: 5,
      pollIntervalMs: 250,
    });
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/v1/runs")) {
        return new Response(JSON.stringify({ run_id: "run-y", status: "started" }), { status: 200 });
      }
      if (url.endsWith("/events")) {
        return new Response(
          sseStream(['event: run.failed', 'data: {"event":"run.failed","run_id":"run-y","error":"boom"}', ""].join("\n")),
          { status: 200, headers: { "content-type": "text/event-stream" } },
        );
      }
      if (/\/v1\/runs\/run-y$/.test(url)) {
        return new Response(JSON.stringify({ status: "failed", error: "boom" }), { status: 200 });
      }
      return new Response(JSON.stringify({ status: "stopping" }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await execute(ctx);

    expect(result.exitCode).toBe(1);
    // No confirmed tip anywhere: params carry only the canonical id, no tip.
    expect(result.sessionParams).toEqual({
      hermesRunId: "run-y",
      hermesCanonicalSessionId: CANONICAL,
      strategy: "issue",
    });
    expect(result.sessionDisplayId).toBeNull();
  });
});
