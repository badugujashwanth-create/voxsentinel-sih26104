import type { ScenarioId } from "../../scenarios/scenarios";

const CALLS_PATH = "/api/v1/calls";

export interface CreateCallRequest {
  claimed_identity: string;
  scenario: ScenarioId;
  transaction_value?: number;
  currency?: string;
  speaker_profile_id?: string;
}

export interface CreateCallResponse {
  call_id: string;
  status: "CREATED";
  claimed_identity: string;
  scenario: ScenarioId;
}

export interface CallSessionResponse {
  call_id: string;
  status: "CREATED" | "LIVE" | "VERIFYING" | "BLOCKED" | "COMPLETED" | "FAILED";
  claimed_identity: string;
  scenario: ScenarioId;
  transaction_value: number | null;
  currency: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  speaker_profile_id?: string | null;
}

export class CallSessionError extends Error {
  public readonly status: number | null;

  /** Creates a typed error for failed call-session requests. */
  public constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "CallSessionError";
    this.status = status;
  }
}

export interface CallSessionClientOptions {
  baseUrl?: string;
  fetcher?: typeof fetch;
}

export interface CallSessionOperations {
  createCall(request: CreateCallRequest): Promise<CreateCallResponse>;
  startCall(callId: string): Promise<CallSessionResponse>;
  stopCall(callId: string): Promise<CallSessionResponse>;
}

/** Owns the frontend HTTP calls for the backend call lifecycle. */
export class CallSessionClient implements CallSessionOperations {
  private readonly baseUrl: string;
  private readonly fetcher: typeof fetch;

  /** Creates a call-session client from the configured backend origin. */
  public constructor(options: CallSessionClientOptions = {}) {
    this.baseUrl = options.baseUrl ?? getConfiguredBaseUrl();
    this.fetcher = (options.fetcher ?? fetch).bind(globalThis);
  }

  /** Creates a backend call session in CREATED state. */
  public async createCall(request: CreateCallRequest): Promise<CreateCallResponse> {
    return this.request<CreateCallResponse>(CALLS_PATH, { method: "POST", body: JSON.stringify(request) }, isCreateCallResponse);
  }

  /** Transitions a created backend call to LIVE. */
  public async startCall(callId: string): Promise<CallSessionResponse> {
    return this.request<CallSessionResponse>(buildCallPath(callId, "/start"), { method: "POST" }, isCallSessionResponse);
  }

  /** Transitions a live backend call to COMPLETED. */
  public async stopCall(callId: string): Promise<CallSessionResponse> {
    return this.request<CallSessionResponse>(buildCallPath(callId, "/stop"), { method: "POST" }, isCallSessionResponse);
  }

  /** Sends one JSON request and validates its response shape. */
  private async request<T>(path: string, init: RequestInit, isResponse: (value: unknown) => value is T): Promise<T> {
    let response: Response;
    try {
      response = await this.fetcher(buildApiUrl(this.baseUrl, path), {
        ...init,
        headers: { "Content-Type": "application/json", ...init.headers },
      });
    } catch (error) {
      throw new CallSessionError(error instanceof Error ? error.message : "Backend request failed");
    }

    const body = await readJson(response);
    if (!response.ok) throw new CallSessionError(readErrorMessage(body, response.status), response.status);
    if (!isResponse(body)) throw new CallSessionError("Backend returned an invalid call-session response", response.status);
    return body;
  }
}

/** Reads the configured backend origin or same-origin fallback. */
function getConfiguredBaseUrl(): string {
  return import.meta.env.VITE_API_BASE_URL ?? window.location.origin;
}

/** Builds an API URL while preserving an optional deployment path prefix. */
function buildApiUrl(baseUrl: string, path: string): string {
  const parsed = new URL(baseUrl || window.location.origin);
  parsed.pathname = `${trimTrailingSlash(parsed.pathname)}${path}`;
  parsed.search = "";
  return parsed.toString();
}

/** Builds a URL-safe call lifecycle path. */
function buildCallPath(callId: string, suffix: string): string {
  if (callId.trim().length === 0) throw new CallSessionError("call_id is required");
  return `${CALLS_PATH}/${encodeURIComponent(callId)}${suffix}`;
}

/** Parses a response body without trusting its content type. */
async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

/** Returns a useful backend error detail when one is available. */
function readErrorMessage(body: unknown, status: number): string {
  if (isRecord(body) && typeof body.detail === "string") return body.detail;
  return `Backend request failed with status ${status}`;
}

/** Validates the minimal create-call response contract. */
function isCreateCallResponse(value: unknown): value is CreateCallResponse {
  return isRecord(value) && typeof value.call_id === "string" && value.call_id.length > 0 && value.status === "CREATED" && typeof value.claimed_identity === "string" && isScenario(value.scenario);
}

/** Validates the backend call-session response contract. */
function isCallSessionResponse(value: unknown): value is CallSessionResponse {
  return isRecord(value) && typeof value.call_id === "string" && value.call_id.length > 0 && isCallStatus(value.status) && typeof value.claimed_identity === "string" && isScenario(value.scenario) && (typeof value.transaction_value === "number" || value.transaction_value === null) && (typeof value.currency === "string" || value.currency === null) && typeof value.created_at === "string" && (typeof value.started_at === "string" || value.started_at === null) && (typeof value.completed_at === "string" || value.completed_at === null);
}

/** Checks whether an unknown value is a JSON object. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/** Checks the frontend/backend scenario enum intersection. */
function isScenario(value: unknown): value is ScenarioId {
  return value === "GENUINE" || value === "HUMAN_IMPOSTOR" || value === "AI_CLONE" || value === "HIGH_VALUE_TRANSFER_ATTACK";
}

/** Checks the backend lifecycle status enum. */
function isCallStatus(value: unknown): value is CallSessionResponse["status"] {
  return value === "CREATED" || value === "LIVE" || value === "VERIFYING" || value === "BLOCKED" || value === "COMPLETED" || value === "FAILED";
}

/** Removes trailing slashes before appending an API path. */
function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}
