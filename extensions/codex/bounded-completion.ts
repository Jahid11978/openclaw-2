import { type CodexAppServerClientFactory } from "./src/app-server/client-factory.js";
import type { CodexAppServerClient } from "./src/app-server/client.js";
import { resolveCodexAppServerRuntimeOptions } from "./src/app-server/config.js";
import { readModelListResult } from "./src/app-server/models.js";
import {
  assertCodexThreadStartResponse,
  assertCodexTurnStartResponse,
  readCodexErrorNotification,
  readCodexTurnCompletedNotification,
} from "./src/app-server/protocol-validators.js";
import {
  isJsonObject,
  type CodexServerNotification,
  type CodexThreadItem,
  type CodexThreadStartParams,
  type CodexTurn,
  type CodexTurnStartParams,
  type JsonObject,
  type JsonValue,
} from "./src/app-server/protocol.js";

export type CodexBoundedImageInput = {
  mime?: string;
  base64: string;
};

export type CodexBoundedCompletionOptions = {
  pluginConfig?: unknown;
  clientFactory?: CodexAppServerClientFactory;
};

export type RunCodexBoundedCompletionParams = {
  model: string;
  prompt: string;
  images?: CodexBoundedImageInput[];
  timeoutMs?: number;
  cwd?: string;
  developerInstructions?: string;
};

const DEFAULT_TIMEOUT_MS = 60_000;
const MAX_TIMEOUT_MS = 120_000;

export async function runCodexBoundedCompletion(
  params: RunCodexBoundedCompletionParams,
  options: CodexBoundedCompletionOptions = {},
): Promise<{ text: string; model: string }> {
  const model = params.model.trim();
  if (!model) {
    throw new Error("Codex bounded completion requires model id.");
  }
  const prompt = params.prompt.trim();
  if (!prompt) {
    throw new Error("Codex bounded completion requires prompt.");
  }

  const timeoutMs = Math.min(MAX_TIMEOUT_MS, Math.max(100, params.timeoutMs ?? DEFAULT_TIMEOUT_MS));
  const appServer = resolveCodexAppServerRuntimeOptions({ pluginConfig: options.pluginConfig });
  const ownsClient = !options.clientFactory;
  const client = options.clientFactory
    ? await options.clientFactory(appServer.start, undefined)
    : await import("./src/app-server/shared-client.js").then(
        ({ createIsolatedCodexAppServerClient }) =>
          createIsolatedCodexAppServerClient({
            startOptions: appServer.start,
            timeoutMs,
          }),
      );
  const abortController = new AbortController();
  const timeout = setTimeout(() => abortController.abort("timeout"), timeoutMs);
  timeout.unref?.();

  try {
    await assertCodexModelSupportsModalities({
      client,
      model,
      needsImage: (params.images?.length ?? 0) > 0,
      timeoutMs,
      signal: abortController.signal,
    });
    const thread = assertCodexThreadStartResponse(
      await client.request<unknown>(
        "thread/start",
        {
          model,
          modelProvider: "openai",
          cwd: params.cwd || process.cwd(),
          approvalPolicy: "on-request",
          sandbox: "read-only",
          serviceName: "OpenClaw",
          developerInstructions:
            params.developerInstructions ??
            "You are OpenClaw's bounded extraction worker. Return only the requested content. Do not call tools, edit files, or ask follow-up questions.",
          dynamicTools: [],
          experimentalRawEvents: true,
          persistExtendedHistory: false,
          ephemeral: true,
        } satisfies CodexThreadStartParams,
        { timeoutMs, signal: abortController.signal },
      ),
    );
    const collector = createCodexBoundedTurnCollector(thread.thread.id);
    const cleanup = client.addNotificationHandler(collector.handleNotification);
    const requestCleanup = client.addRequestHandler(denyCodexBoundedApprovalRequest);
    try {
      const turn = assertCodexTurnStartResponse(
        await client.request<unknown>(
          "turn/start",
          {
            threadId: thread.thread.id,
            input: [
              { type: "text", text: prompt, text_elements: [] },
              ...(params.images ?? []).map((image) => ({
                type: "image" as const,
                url: `data:${image.mime ?? "image/png"};base64,${image.base64}`,
              })),
            ],
            cwd: params.cwd || process.cwd(),
            approvalPolicy: "on-request",
            model,
            effort: "low",
          } satisfies CodexTurnStartParams,
          { timeoutMs, signal: abortController.signal },
        ),
      );
      return {
        text: await collector.collect(turn.turn, { timeoutMs, signal: abortController.signal }),
        model,
      };
    } finally {
      requestCleanup();
      cleanup();
    }
  } finally {
    clearTimeout(timeout);
    if (ownsClient) {
      client.close();
    }
  }
}

function denyCodexBoundedApprovalRequest(request: { method: string }): JsonValue | undefined {
  if (
    request.method === "item/commandExecution/requestApproval" ||
    request.method === "item/fileChange/requestApproval"
  ) {
    return {
      decision: "decline",
      reason: "OpenClaw bounded Codex extraction does not grant tool or file approvals.",
    };
  }
  if (request.method === "item/permissions/requestApproval") {
    return { permissions: {}, scope: "turn" };
  }
  if (request.method.includes("requestApproval")) {
    return {
      decision: "decline",
      reason: "OpenClaw bounded Codex extraction does not grant native approvals.",
    };
  }
  if (request.method === "mcpServer/elicitation/request") {
    return { action: "decline" };
  }
  return undefined;
}

async function assertCodexModelSupportsModalities(params: {
  client: CodexAppServerClient;
  model: string;
  needsImage: boolean;
  timeoutMs: number;
  signal: AbortSignal;
}): Promise<void> {
  const result = await params.client.request<unknown>(
    "model/list",
    { limit: 100, cursor: null, includeHidden: false },
    { timeoutMs: Math.min(params.timeoutMs, 5_000), signal: params.signal },
  );
  const listed = readModelListResult(result).models;
  const match = listed.find((entry) => entry.model === params.model || entry.id === params.model);
  if (!match) {
    throw new Error(`Codex app-server model not found: ${params.model}`);
  }
  if (params.needsImage && !match.inputModalities.includes("image")) {
    throw new Error(`Codex app-server model does not support images: ${params.model}`);
  }
}

function createCodexBoundedTurnCollector(threadId: string) {
  let turnId: string | undefined;
  let completedTurn: CodexTurn | undefined;
  let promptError: string | undefined;
  const pending: CodexServerNotification[] = [];
  const assistantTextByItem = new Map<string, string>();
  const assistantItemOrder: string[] = [];
  let resolveCompletion: (() => void) | undefined;
  const completion = new Promise<void>((resolve) => {
    resolveCompletion = resolve;
  });

  const rememberAssistantText = (itemId: string, text: string) => {
    if (!text) {
      return;
    }
    if (!assistantTextByItem.has(itemId)) {
      assistantItemOrder.push(itemId);
    }
    assistantTextByItem.set(itemId, text);
  };

  const handleNotification = (notification: CodexServerNotification): void => {
    const params = isJsonObject(notification.params) ? notification.params : undefined;
    if (!params || readString(params, "threadId") !== threadId) {
      return;
    }
    if (!turnId) {
      pending.push(notification);
      return;
    }
    const notificationTurnId = readNotificationTurnId(params);
    if (notificationTurnId !== turnId) {
      return;
    }
    if (notification.method === "item/agentMessage/delta") {
      const itemId = readString(params, "itemId") ?? readString(params, "id") ?? "assistant";
      const delta = readString(params, "delta") ?? "";
      rememberAssistantText(itemId, `${assistantTextByItem.get(itemId) ?? ""}${delta}`);
      return;
    }
    if (notification.method === "turn/completed") {
      completedTurn =
        readCodexTurnCompletedNotification(notification.params)?.turn ?? completedTurn;
      resolveCompletion?.();
      return;
    }
    if (notification.method === "error") {
      promptError =
        readCodexErrorNotification(notification.params)?.error.message ??
        "codex app-server bounded turn failed";
      resolveCompletion?.();
    }
  };

  return {
    handleNotification,
    async collect(
      startedTurn: CodexTurn,
      options: { timeoutMs: number; signal: AbortSignal },
    ): Promise<string> {
      turnId = startedTurn.id;
      if (isTerminalTurn(startedTurn)) {
        completedTurn = startedTurn;
      }
      for (const notification of pending.splice(0)) {
        handleNotification(notification);
      }
      if (!completedTurn && !promptError) {
        await waitForTurnCompletion({
          completion,
          timeoutMs: options.timeoutMs,
          signal: options.signal,
        });
      }
      if (promptError) {
        throw new Error(promptError);
      }
      if (completedTurn?.status === "failed") {
        throw new Error(completedTurn.error?.message ?? "codex app-server bounded turn failed");
      }
      const itemText = collectAssistantTextFromItems(completedTurn?.items);
      const deltaText = assistantItemOrder
        .map((itemId) => assistantTextByItem.get(itemId)?.trim())
        .filter((text): text is string => Boolean(text))
        .join("\n\n")
        .trim();
      const text = (itemText || deltaText).trim();
      if (!text) {
        throw new Error("Codex app-server bounded turn returned no text.");
      }
      return text;
    },
  };
}

async function waitForTurnCompletion(params: {
  completion: Promise<void>;
  timeoutMs: number;
  signal: AbortSignal;
}): Promise<void> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let cleanupAbort: (() => void) | undefined;
  try {
    await Promise.race([
      params.completion,
      new Promise<never>((_, reject) => {
        timeout = setTimeout(
          () => reject(new Error("codex app-server bounded turn timed out")),
          params.timeoutMs,
        );
        timeout.unref?.();
        const abortListener = () => reject(new Error("codex app-server bounded turn aborted"));
        params.signal.addEventListener("abort", abortListener, { once: true });
        cleanupAbort = () => params.signal.removeEventListener("abort", abortListener);
      }),
    ]);
  } finally {
    if (timeout) {
      clearTimeout(timeout);
    }
    cleanupAbort?.();
  }
}

function collectAssistantTextFromItems(items: CodexThreadItem[] | undefined): string {
  return (items ?? [])
    .filter((item) => item.type === "agentMessage")
    .map((item) => item.text.trim())
    .filter(Boolean)
    .join("\n\n")
    .trim();
}

function readNotificationTurnId(record: JsonObject): string | undefined {
  const direct = readString(record, "turnId");
  if (direct) {
    return direct;
  }
  return isJsonObject(record.turn) ? readString(record.turn, "id") : undefined;
}

function readString(record: JsonObject, key: string): string | undefined {
  const value = record[key];
  return typeof value === "string" ? value : undefined;
}

function isTerminalTurn(turn: CodexTurn): boolean {
  return turn.status === "completed" || turn.status === "interrupted" || turn.status === "failed";
}
