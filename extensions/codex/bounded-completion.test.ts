import { Readable } from "node:stream";
import { describe, expect, it, vi } from "vitest";
import { runCodexBoundedCompletion } from "./bounded-completion.js";
import {
  createGBrainExtractionRouteHandler,
  GBRAIN_EXTRACTION_ROUTE_PATH,
} from "./gbrain-extraction-route.js";
import type { CodexAppServerClient } from "./src/app-server/client.js";
import type { CodexServerNotification, JsonValue } from "./src/app-server/protocol.js";

function codexModel(inputModalities: string[] = ["text", "image"]) {
  return {
    id: "gpt-5.4-mini",
    model: "gpt-5.4-mini",
    upgrade: null,
    upgradeInfo: null,
    availabilityNux: null,
    displayName: "GPT-5.4-Mini",
    description: "GPT-5.4-Mini",
    hidden: false,
    supportedReasoningEfforts: [{ reasoningEffort: "low", description: "fast" }],
    defaultReasoningEffort: "low",
    inputModalities,
    supportsPersonality: false,
    additionalSpeedTiers: [],
    isDefault: true,
  };
}

function createFakeClient() {
  const notifications = new Set<(notification: CodexServerNotification) => void>();
  const requestHandlers = new Set<(request: { method: string }) => JsonValue | undefined>();
  const requests: Array<{
    method: string;
    params?: JsonValue;
    options?: { timeoutMs?: number; signal?: AbortSignal };
  }> = [];
  const request = vi.fn(
    async (
      method: string,
      params?: JsonValue,
      options?: { timeoutMs?: number; signal?: AbortSignal },
    ) => {
      requests.push({ method, params, options });
      if (method === "model/list") {
        return { data: [codexModel()], nextCursor: null };
      }
      if (method === "thread/start") {
        return {
          thread: {
            id: "thread-1",
            forkedFromId: null,
            preview: "",
            ephemeral: true,
            modelProvider: "openai",
            createdAt: 1,
            updatedAt: 1,
            status: { type: "idle" },
            path: null,
            cwd: "/tmp/openclaw-agent",
            cliVersion: "0.125.0",
            source: "unknown",
            agentNickname: null,
            agentRole: null,
            gitInfo: null,
            name: null,
            turns: [],
          },
          model: "gpt-5.4-mini",
          modelProvider: "openai",
          serviceTier: null,
          cwd: "/tmp/openclaw-agent",
          instructionSources: [],
          approvalPolicy: "on-request",
          approvalsReviewer: "user",
          sandbox: { type: "dangerFullAccess" },
          permissionProfile: null,
          reasoningEffort: null,
        };
      }
      if (method === "turn/start") {
        for (const handler of requestHandlers) {
          handler({ method: "item/permissions/requestApproval" });
        }
        for (const notify of notifications) {
          notify({
            method: "item/agentMessage/delta",
            params: {
              threadId: "thread-1",
              turnId: "turn-1",
              itemId: "msg-1",
              delta:
                '{"schemaVersion":"gbrain.media-extraction.v1","kind":"image","sourceRef":"photo.png","summary":"A receipt on a desk.","tags":["receipt"],"segments":[{"id":"frame-1","kind":"frame","caption":"Receipt photo"}]}',
            },
          });
          notify({
            method: "turn/completed",
            params: {
              threadId: "thread-1",
              turnId: "turn-1",
              turn: {
                id: "turn-1",
                status: "completed",
                items: [],
                error: null,
                startedAt: null,
                completedAt: null,
                durationMs: null,
              },
            },
          });
        }
        return {
          turn: {
            id: "turn-1",
            status: "inProgress",
            items: [],
            error: null,
            startedAt: null,
            completedAt: null,
            durationMs: null,
          },
        };
      }
      return {};
    },
  );

  const client = {
    request,
    close: vi.fn(),
    addNotificationHandler(handler: (notification: CodexServerNotification) => void) {
      notifications.add(handler);
      return () => notifications.delete(handler);
    },
    addRequestHandler(handler: (request: { method: string }) => JsonValue | undefined) {
      requestHandlers.add(handler);
      return () => requestHandlers.delete(handler);
    },
  } as unknown as CodexAppServerClient;

  return { client, requests };
}

function createResponseCapture() {
  const headers = new Map<string, string | number | readonly string[]>();
  let body = "";
  const res = {
    statusCode: 200,
    headersSent: false,
    setHeader(name: string, value: string | number | readonly string[]) {
      headers.set(name, value);
    },
    end(value?: string) {
      body = value ?? "";
      this.headersSent = true;
    },
  };
  return {
    res,
    headers,
    get body() {
      return body;
    },
  };
}

describe("codex bounded completion", () => {
  it("runs a no-tools ephemeral Codex app-server turn without API-key payloads", async () => {
    const { client, requests } = createFakeClient();
    const result = await runCodexBoundedCompletion(
      {
        model: "gpt-5.4-mini",
        prompt: "Return JSON.",
        images: [{ mime: "image/png", base64: "aW1hZ2U=" }],
      },
      { clientFactory: async () => client },
    );

    expect(result.text).toContain("gbrain.media-extraction.v1");
    expect(requests.map((entry) => entry.method)).toEqual([
      "model/list",
      "thread/start",
      "turn/start",
    ]);
    expect(requests[1]?.params).toMatchObject({
      model: "gpt-5.4-mini",
      modelProvider: "openai",
      approvalPolicy: "on-request",
      sandbox: "read-only",
      dynamicTools: [],
      ephemeral: true,
    });
    expect(JSON.stringify(requests)).not.toMatch(/apiKey|OPENAI_API_KEY|refreshToken/i);
  });

  it("clamps caller supplied timeouts for bounded Codex turns", async () => {
    const { client, requests } = createFakeClient();
    await runCodexBoundedCompletion(
      {
        model: "gpt-5.4-mini",
        prompt: "Return JSON.",
        timeoutMs: 86_400_000,
      },
      { clientFactory: async () => client },
    );

    expect(requests[0]?.method).toBe("model/list");
    expect(requests[0]?.options).toMatchObject({ timeoutMs: 5_000 });
    expect(requests[1]?.options).toMatchObject({ timeoutMs: 120_000 });
    expect(requests[2]?.options).toMatchObject({ timeoutMs: 120_000 });
  });
});

describe("gbrain extraction route", () => {
  it("registers the public extraction route path", () => {
    expect(GBRAIN_EXTRACTION_ROUTE_PATH).toBe("/plugins/gbrain/extract");
  });

  it("returns normalized media extraction JSON for image requests", async () => {
    const complete = vi.fn(async () => ({
      text: '{"schemaVersion":"gbrain.media-extraction.v1","kind":"image","sourceRef":"photo.png","summary":"A receipt on a desk.","tags":["receipt"],"segments":[{"id":"frame-1","kind":"frame","caption":"Receipt photo"}]}',
      model: "gpt-5.4-mini",
    }));
    const handler = createGBrainExtractionRouteHandler({}, { complete });
    const req = Readable.from([
      JSON.stringify({
        kind: "image",
        sourceRef: "photo.png",
        file: { mime: "image/png", base64: "aW1hZ2U=" },
      }),
    ]) as unknown as import("node:http").IncomingMessage;
    req.method = "POST";
    const capture = createResponseCapture();

    await handler(req, capture.res as unknown as import("node:http").ServerResponse);

    expect(capture.res.statusCode).toBe(200);
    const payload = JSON.parse(capture.body) as {
      ok: boolean;
      extraction: { schemaVersion: string; kind: string; sourceRef: string; tags: string[] };
    };
    expect(payload).toMatchObject({
      ok: true,
      extraction: {
        schemaVersion: "gbrain.media-extraction.v1",
        kind: "image",
        sourceRef: "photo.png",
        tags: ["receipt"],
      },
    });
    const firstCall = complete.mock.calls.at(0) as
      | [Record<string, unknown>, Record<string, unknown>?]
      | undefined;
    expect(firstCall?.[0]).toMatchObject({
      model: "gpt-5.4-mini",
      images: [{ mime: "image/png", base64: "aW1hZ2U=" }],
    });
    expect(JSON.stringify(firstCall?.[0])).not.toMatch(/apiKey|refreshToken/i);
  });

  it("trims blank media kind and rejects credential fields in request bodies", async () => {
    const complete = vi.fn(async () => ({
      text: '{"schemaVersion":"gbrain.media-extraction.v1","kind":"pdf","sourceRef":"note.txt","summary":"A note.","segments":[{"id":"page-1","kind":"page","summary":"A note."}]}',
      model: "gpt-5.4-mini",
    }));
    const handler = createGBrainExtractionRouteHandler({}, { complete });
    const req = Readable.from([
      JSON.stringify({
        kind: "   ",
        sourceRef: "note.txt",
        text: "A note.",
      }),
    ]) as unknown as import("node:http").IncomingMessage;
    req.method = "POST";
    const capture = createResponseCapture();

    await handler(req, capture.res as unknown as import("node:http").ServerResponse);

    expect(capture.res.statusCode).toBe(200);
    expect(JSON.parse(capture.body)).toMatchObject({
      ok: true,
      extraction: { kind: "pdf", sourceRef: "note.txt" },
    });

    const credentialReq = Readable.from([
      JSON.stringify({
        kind: "image",
        sourceRef: "photo.png",
        file: { mime: "image/png", base64: "aW1hZ2U=" },
        apiKey: "sk-nope",
      }),
    ]) as unknown as import("node:http").IncomingMessage;
    credentialReq.method = "POST";
    const credentialCapture = createResponseCapture();

    await handler(
      credentialReq,
      credentialCapture.res as unknown as import("node:http").ServerResponse,
    );

    expect(credentialCapture.res.statusCode).toBe(400);
    expect(JSON.parse(credentialCapture.body)).toMatchObject({
      ok: false,
      error: "forbidden_field",
    });
  });

  it("returns a stable error when Codex returns malformed JSON", async () => {
    const complete = vi.fn(async () => ({
      text: "```json\n{bad}\n```",
      model: "gpt-5.4-mini",
    }));
    const handler = createGBrainExtractionRouteHandler({}, { complete });
    const req = Readable.from([
      JSON.stringify({
        kind: "image",
        sourceRef: "photo.png",
        file: { mime: "image/png", base64: "aW1hZ2U=" },
      }),
    ]) as unknown as import("node:http").IncomingMessage;
    req.method = "POST";
    const capture = createResponseCapture();

    await handler(req, capture.res as unknown as import("node:http").ServerResponse);

    expect(capture.res.statusCode).toBe(502);
    expect(JSON.parse(capture.body)).toMatchObject({
      ok: false,
      error: "invalid_extraction_json",
      message: "Codex returned invalid extraction JSON.",
    });
  });
});
