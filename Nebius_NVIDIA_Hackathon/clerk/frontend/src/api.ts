export type Mode = "live" | "offline";

export interface Skill {
  id: string;
  description: string;
  enabled: boolean;
}

export interface Config {
  mode: Mode;
  search_mode: Mode;
  models: Record<"super" | "nano" | "vision", string>;
  skills: Skill[];
  limits: { max_upload_bytes: number; audits_per_hour: number };
}

export interface DocSummary {
  id: string;
  filename: string;
  kind: "bill" | "contract" | "unknown";
  provider: string | null;
  created: number;
  last_audit_id: string | null;
}

export interface LineItem {
  index: number;
  description: string;
  amount: number;
  date: string | null;
  quantity: number;
  unit_price: number | null;
}

export interface Bill {
  provider: string;
  account: string | null;
  customer: string | null;
  statement_date: string | null;
  due_date: string | null;
  line_items: LineItem[];
  subtotal: number | null;
  tax: number | null;
  tax_rate: number | null;
  total: number | null;
}

export interface Source {
  title: string;
  url: string;
  snippet: string;
}

export interface Finding {
  id: string;
  type: string;
  title: string;
  line_indexes: number[];
  claimed_amount: number;
  rationale: string;
  source: "detector" | "llm";
  verified: boolean | null;
  verified_amount: number | null;
  verification_note: string | null;
  verdict: "upheld" | "dismissed" | null;
  verdict_reason: string | null;
  confidence: number | null;
  sources: Source[];
}

export interface Summary {
  provider: string;
  statement_date: string | null;
  recoverable_total: number;
  upheld: number;
  dismissed: number;
  findings: Finding[];
  facts_used: Record<string, unknown> | null;
  action_ids: string[];
}

export interface TraceStep {
  agent: string;
  model: string;
  mode: Mode;
  prompt_tokens: number;
  completion_tokens: number;
  latency_ms: number;
  cost_usd: number | null;
}

export interface Action {
  id: string;
  audit_id: string | null;
  type: "letter" | "call_script" | "reminder";
  status: "pending" | "approved" | "rejected";
  title: string;
  content: string;
  filename: string;
  mime: string;
  created: number;
}

export interface Fact {
  id: string;
  document_id: string | null;
  provider: string;
  key: string;
  value: string;
}

export interface Scoreboard {
  recoverable_total: number;
  upheld: number;
  dismissed: number;
  bills_audited: number;
}

export type StageName = "reader" | "vault" | "auditors" | "verifier" | "skeptic" | "researcher" | "advocate";

export type AuditEvent =
  | { type: "start"; audit_id: string }
  | { type: "stage"; stage: StageName; status: "start" | "done" | "skipped"; detail: string }
  | { type: "bill"; bill: Bill }
  | { type: "facts"; facts: Record<string, unknown> | null }
  | { type: "candidate" | "verified" | "verdict" | "sources"; finding: Finding }
  | { type: "action"; action: Action }
  | { type: "trace"; step: TraceStep }
  | { type: "notice"; message: string }
  | { type: "summary"; summary: Summary }
  | { type: "done"; audit_id: string }
  | { type: "error"; message: string; audit_id?: string };

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { credentials: "same-origin", ...init });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  config: () => request<Config>("/api/config"),
  documents: () => request<DocSummary[]>("/api/documents"),
  document: (id: string) =>
    request<DocSummary & { text: string; parsed: Bill | Record<string, unknown> | null }>(`/api/documents/${id}`),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ id: string; kind: string; provider: string; facts_remembered: number }>("/api/documents", {
      method: "POST",
      body: form,
    });
  },
  audit: (id: string) => request<{ status: string; result: Summary | null }>(`/api/audits/${id}`),
  scoreboard: () => request<Scoreboard>("/api/scoreboard"),
  facts: () => request<Fact[]>("/api/vault/facts"),
  editFact: (id: string, value: string) => request<{ ok: boolean }>(`/api/vault/facts/${id}`, json("PATCH", { value })),
  deleteFact: (id: string) => request<{ ok: boolean }>(`/api/vault/facts/${id}`, { method: "DELETE" }),
  ask: (question: string) => request<{ answer: string; facts: Fact[] }>("/api/vault/ask", json("POST", { question })),
  actions: () => request<Action[]>("/api/actions"),
  decide: (id: string, decision: "approve" | "reject") =>
    request<{ status: string }>(`/api/actions/${id}/${decision}`, { method: "POST" }),
  traces: () => request<(TraceStep & { id: string; audit_id: string | null; created: number })[]>("/api/traces"),
  setPermission: (skill: string, enabled: boolean) =>
    request<{ ok: boolean }>("/api/permissions", json("PUT", { skill, enabled })),
  loadDemo: () => request<unknown[]>("/api/demo/load", { method: "POST" }),
  samples: () => request<string[]>("/api/samples"),
  wipe: () => request<{ ok: boolean }>("/api/workspace", { method: "DELETE" }),
};

/** Streams the Tribunal's Server-Sent Events. Resolves when the stream ends. */
export async function streamAudit(docId: string, onEvent: (event: AuditEvent) => void, signal?: AbortSignal) {
  const res = await fetch(`/api/documents/${docId}/audit/stream`, { credentials: "same-origin", signal });
  if (!res.ok || !res.body) {
    let detail = `Audit failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(detail, res.status);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let split;
    while ((split = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      const data = chunk
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("");
      if (data) onEvent(JSON.parse(data) as AuditEvent);
    }
  }
}

export const money = (n: number | null | undefined) =>
  n == null ? "—" : n.toLocaleString("en-US", { style: "currency", currency: "USD" });
