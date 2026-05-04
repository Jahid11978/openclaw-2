import type { IncomingMessage, ServerResponse } from "node:http";
import {
  runCodexBoundedCompletion,
  type CodexBoundedCompletionOptions,
  type CodexBoundedImageInput,
} from "./bounded-completion.js";
import { FALLBACK_CODEX_MODELS } from "./provider-catalog.js";

type MediaExtractionKind = "image" | "pdf" | "video" | "audio";
type MediaExtractionSegmentKind =
  | "asset"
  | "page"
  | "frame"
  | "transcript_segment"
  | "audio_segment";

type MediaExtractionSegment = {
  id: string;
  kind: MediaExtractionSegmentKind;
  label?: string;
  summary?: string;
  caption?: string;
  ocrText?: string;
  transcriptText?: string;
  tags?: string[];
  entities?: MediaExtractionEntity[];
};

type MediaExtractionEntity = {
  text: string;
  type?: string;
};

type MediaExtraction = {
  schemaVersion: "gbrain.media-extraction.v1";
  kind: MediaExtractionKind;
  sourceRef: string;
  title?: string;
  summary?: string;
  tags?: string[];
  entities?: MediaExtractionEntity[];
  segments?: MediaExtractionSegment[];
  metadata?: Record<string, unknown>;
};

type GBrainExtractRequest = {
  protocol?: string;
  kind?: string;
  sourceRef?: string;
  title?: string;
  text?: string;
  file?: {
    name?: string;
    mime?: string;
    base64?: string;
  };
  model?: string;
  timeoutMs?: number;
};

type GBrainExtractionDeps = {
  complete?: (
    params: Parameters<typeof runCodexBoundedCompletion>[0],
    options?: CodexBoundedCompletionOptions,
  ) => Promise<{ text: string; model: string }>;
};

const ROUTE_PATH = "/plugins/gbrain/extract";
const DEFAULT_CODEX_EXTRACTION_MODEL =
  FALLBACK_CODEX_MODELS.find((model) => model.id === "gpt-5.4-mini")?.id ??
  FALLBACK_CODEX_MODELS[0]?.id ??
  "gpt-5.4-mini";
const MAX_BODY_BYTES = 12 * 1024 * 1024;

export function createGBrainExtractionRouteHandler(
  options: CodexBoundedCompletionOptions = {},
  deps: GBrainExtractionDeps = {},
) {
  const complete = deps.complete ?? runCodexBoundedCompletion;
  return async (req: IncomingMessage, res: ServerResponse): Promise<boolean> => {
    if (req.method !== "POST") {
      writeJson(res, 405, { ok: false, error: "method_not_allowed" });
      return true;
    }

    try {
      const body = await readJsonBody(req, MAX_BODY_BYTES);
      const request = readExtractionRequest(body);
      const kind = normalizeMediaKind(request.kind);
      const sourceRef = normalizeRequiredString(request.sourceRef, "sourceRef");
      const title = normalizeOptionalString(request.title);
      const text = normalizeOptionalString(request.text);
      const image = readImageInput(request);
      if (!text && !image) {
        throw new RequestError(400, "missing_content", "Provide text or file.base64.");
      }
      if (kind !== "image" && !text) {
        throw new RequestError(
          400,
          "missing_text",
          "Video, audio, and document MVP extraction requires text or transcript content.",
        );
      }

      const result = await complete(
        {
          model: normalizeOptionalString(request.model) ?? DEFAULT_CODEX_EXTRACTION_MODEL,
          prompt: buildExtractionPrompt({ kind, sourceRef, title, text, hasImage: Boolean(image) }),
          ...(image ? { images: [image] } : {}),
          ...(typeof request.timeoutMs === "number" ? { timeoutMs: request.timeoutMs } : {}),
          developerInstructions:
            "You are OpenClaw's bounded GBrain extraction worker. Return valid JSON only. Do not call tools, edit files, log secrets, or ask follow-up questions.",
        },
        options,
      );

      const extraction = normalizeExtraction({
        parsed: parseJsonObject(result.text),
        kind,
        sourceRef,
        title,
      });
      writeJson(res, 200, {
        ok: true,
        protocol: "gbrain.media-extraction.v1",
        provider: "codex",
        model: result.model,
        extraction,
      });
      return true;
    } catch (err) {
      if (err instanceof RequestError) {
        writeJson(res, err.statusCode, {
          ok: false,
          error: err.code,
          message: err.message,
        });
        return true;
      }
      writeJson(res, 502, {
        ok: false,
        error: "extraction_failed",
        message: "Codex extraction failed.",
      });
      return true;
    }
  };
}

export { ROUTE_PATH as GBRAIN_EXTRACTION_ROUTE_PATH };

function buildExtractionPrompt(params: {
  kind: MediaExtractionKind;
  sourceRef: string;
  title?: string;
  text?: string;
  hasImage: boolean;
}): string {
  const segmentKind = defaultSegmentKind(params.kind);
  const parts = [
    "Return only JSON matching this TypeScript shape:",
    "{ schemaVersion:'gbrain.media-extraction.v1', kind:'image'|'pdf'|'video'|'audio', sourceRef:string, title?:string, summary?:string, tags?:string[], entities?:{text:string,type?:string}[], segments:{id:string, kind:'asset'|'page'|'frame'|'transcript_segment'|'audio_segment', label?:string, summary?:string, caption?:string, ocrText?:string, transcriptText?:string, tags?:string[], entities?:{text:string,type?:string}[]}[] }",
    "",
    `kind: ${params.kind}`,
    `sourceRef: ${params.sourceRef}`,
    `preferred segment kind: ${segmentKind}`,
  ];
  if (params.title) {
    parts.push(`title: ${params.title}`);
  }
  if (params.hasImage) {
    parts.push(
      "Analyze the attached image. Categorize it, summarize it, add searchable tags, extract visible text into ocrText, and list visible entities when identifiable.",
    );
  }
  if (params.text) {
    parts.push("Text/transcript content:", params.text);
  }
  return parts.join("\n");
}

function defaultSegmentKind(kind: MediaExtractionKind): MediaExtractionSegmentKind {
  if (kind === "image") {
    return "frame";
  }
  if (kind === "video") {
    return "transcript_segment";
  }
  if (kind === "audio") {
    return "audio_segment";
  }
  return "page";
}

function normalizeExtraction(params: {
  parsed: Record<string, unknown>;
  kind: MediaExtractionKind;
  sourceRef: string;
  title?: string;
}): MediaExtraction {
  const candidate = isRecord(params.parsed.extraction) ? params.parsed.extraction : params.parsed;
  const summary = normalizeOptionalString(candidate.summary);
  const extraction: MediaExtraction = {
    schemaVersion: "gbrain.media-extraction.v1",
    kind: params.kind,
    sourceRef: params.sourceRef,
    ...(params.title ? { title: params.title } : {}),
    ...(summary ? { summary } : {}),
    tags: normalizeStringArray(candidate.tags),
    entities: normalizeEntities(candidate.entities),
    segments: normalizeSegments(candidate.segments, params.kind, summary),
  };
  const returnedTitle = normalizeOptionalString(candidate.title);
  if (returnedTitle && !extraction.title) {
    extraction.title = returnedTitle;
  }
  return extraction;
}

function normalizeSegments(
  value: unknown,
  kind: MediaExtractionKind,
  fallbackSummary?: string,
): MediaExtractionSegment[] {
  const source = Array.isArray(value) ? value : [];
  const segments = source.filter(isRecord).map((segment, index): MediaExtractionSegment => {
    const segmentKind = normalizeSegmentKind(segment.kind) ?? defaultSegmentKind(kind);
    const normalized: MediaExtractionSegment = {
      id: normalizeOptionalString(segment.id) ?? `${segmentKind}-${index + 1}`,
      kind: segmentKind,
      tags: normalizeStringArray(segment.tags),
      entities: normalizeEntities(segment.entities),
    };
    const label = normalizeOptionalString(segment.label);
    const summary = normalizeOptionalString(segment.summary);
    const caption = normalizeOptionalString(segment.caption);
    const ocrText =
      normalizeOptionalString(segment.ocrText) ?? normalizeOptionalString(segment.ocr);
    const transcriptText =
      normalizeOptionalString(segment.transcriptText) ??
      normalizeOptionalString(segment.transcript);
    if (label) {
      normalized.label = label;
    }
    if (summary) {
      normalized.summary = summary;
    }
    if (caption) {
      normalized.caption = caption;
    }
    if (ocrText) {
      normalized.ocrText = ocrText;
    }
    if (transcriptText) {
      normalized.transcriptText = transcriptText;
    }
    return normalized;
  });
  if (segments.length > 0) {
    return segments;
  }
  return [
    {
      id: `${defaultSegmentKind(kind)}-1`,
      kind: defaultSegmentKind(kind),
      ...(fallbackSummary ? { summary: fallbackSummary } : {}),
    },
  ];
}

function normalizeSegmentKind(value: unknown): MediaExtractionSegmentKind | undefined {
  if (
    value === "asset" ||
    value === "page" ||
    value === "frame" ||
    value === "transcript_segment" ||
    value === "audio_segment"
  ) {
    return value;
  }
  return undefined;
}

function normalizeMediaKind(value: unknown): MediaExtractionKind {
  const kind = normalizeOptionalString(value);
  if (kind === "image" || kind === "pdf" || kind === "video" || kind === "audio") {
    return kind;
  }
  if (kind === "text" || kind === undefined) {
    return "pdf";
  }
  throw new RequestError(400, "invalid_kind", "kind must be image, pdf, video, audio, or text.");
}

function readImageInput(request: GBrainExtractRequest): CodexBoundedImageInput | undefined {
  const base64 = normalizeOptionalString(request.file?.base64);
  if (!base64) {
    return undefined;
  }
  const mime = normalizeOptionalString(request.file?.mime);
  return {
    base64,
    ...(mime ? { mime } : {}),
  };
}

async function readJsonBody(req: IncomingMessage, maxBytes: number): Promise<unknown> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    total += buffer.length;
    if (total > maxBytes) {
      throw new RequestError(413, "body_too_large", "Request body is too large.");
    }
    chunks.push(buffer);
  }
  const raw = Buffer.concat(chunks).toString("utf8").trim();
  if (!raw) {
    throw new RequestError(400, "empty_body", "Request body is empty.");
  }
  try {
    return JSON.parse(raw) as unknown;
  } catch {
    throw new RequestError(400, "invalid_json", "Request body must be valid JSON.");
  }
}

function readExtractionRequest(value: unknown): GBrainExtractRequest {
  if (!isRecord(value)) {
    throw new RequestError(400, "invalid_body", "Request body must be a JSON object.");
  }
  const forbiddenKeys = new Set([
    "apiKey",
    "openaiApiKey",
    "oauthToken",
    "accessToken",
    "refreshToken",
    "token",
    "secret",
  ]);
  const found = Object.keys(value).filter((key) => forbiddenKeys.has(key));
  if (found.length > 0) {
    throw new RequestError(
      400,
      "forbidden_field",
      `${found.join(", ")} is not allowed in request payload.`,
    );
  }
  return value as GBrainExtractRequest;
}

function parseJsonObject(text: string): Record<string, unknown> {
  const trimmed = text.trim();
  const fenced = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i)?.[1]?.trim();
  const jsonText = fenced ?? trimmed;
  try {
    const parsed = JSON.parse(jsonText) as unknown;
    if (isRecord(parsed)) {
      return parsed;
    }
  } catch {
    const start = jsonText.indexOf("{");
    const end = jsonText.lastIndexOf("}");
    if (start >= 0 && end > start) {
      try {
        const parsed = JSON.parse(jsonText.slice(start, end + 1)) as unknown;
        if (isRecord(parsed)) {
          return parsed;
        }
      } catch {
        // Fall through to the stable invalid_extraction_json error below.
      }
    }
  }
  throw new RequestError(502, "invalid_extraction_json", "Codex returned invalid extraction JSON.");
}

function normalizeRequiredString(value: unknown, field: string): string {
  const text = normalizeOptionalString(value);
  if (!text) {
    throw new RequestError(400, "missing_field", `${field} is required.`);
  }
  return text;
}

function normalizeOptionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function normalizeStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const strings = value
    .map((entry) => normalizeOptionalString(entry))
    .filter((entry): entry is string => Boolean(entry));
  return strings.length > 0 ? strings : undefined;
}

function normalizeEntities(value: unknown): MediaExtractionEntity[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const entities = value.filter(isRecord).flatMap((entry): MediaExtractionEntity[] => {
    const text = normalizeOptionalString(entry.text) ?? normalizeOptionalString(entry.name);
    if (!text) {
      return [];
    }
    const type = normalizeOptionalString(entry.type);
    return [{ text, ...(type ? { type } : {}) }];
  });
  return entities.length > 0 ? entities : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function writeJson(res: ServerResponse, statusCode: number, payload: unknown): void {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(payload));
}

class RequestError extends Error {
  constructor(
    readonly statusCode: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}
