import { useEffect, useReducer, useRef } from "react";
import {
  api,
  money,
  streamAudit,
  type AuditEvent,
  type Bill,
  type Config,
  type Finding,
  type StageName,
  type Summary,
  type TraceStep,
} from "../api";

const STAGES: { id: StageName; label: string; who: (c: Config | null) => string }[] = [
  { id: "reader", label: "Reader", who: () => "Nemotron Nano" },
  { id: "vault", label: "Vault", who: () => "your memory" },
  { id: "auditors", label: "Auditors", who: () => "code + Nemotron Nano" },
  { id: "verifier", label: "Verifier", who: () => "plain Python" },
  { id: "skeptic", label: "Skeptic", who: () => "Nemotron Super" },
  { id: "researcher", label: "Researcher", who: (c) => (c?.search_mode === "live" ? "Tavily web search" : "Tavily (off offline)") },
  { id: "advocate", label: "Advocate", who: () => "Nemotron Super" },
];

type StageState = { status: "idle" | "start" | "done" | "skipped"; detail: string };

interface State {
  running: boolean;
  stages: Record<string, StageState>;
  bill: Bill | null;
  order: string[];
  findings: Record<string, Finding>;
  notices: string[];
  traces: TraceStep[];
  summary: Summary | null;
  error: string | null;
}

type Action =
  | { type: "reset"; bill: Bill | null }
  | { type: "event"; event: AuditEvent }
  | { type: "fail"; message: string }
  | { type: "restore"; summary: Summary; bill: Bill | null };

const idleStages = () => Object.fromEntries(STAGES.map((s) => [s.id, { status: "idle", detail: "" } as StageState]));

const initial = (bill: Bill | null): State => ({
  running: false,
  stages: idleStages(),
  bill,
  order: [],
  findings: {},
  notices: [],
  traces: [],
  summary: null,
  error: null,
});

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "reset":
      return initial(action.bill);
    case "fail":
      return { ...state, running: false, error: action.message };
    case "restore": {
      const s = initial(action.bill);
      const stages = Object.fromEntries(STAGES.map((st) => [st.id, { status: "done", detail: "" } as StageState]));
      return {
        ...s,
        stages,
        summary: action.summary,
        order: action.summary.findings.map((f) => f.id),
        findings: Object.fromEntries(action.summary.findings.map((f) => [f.id, f])),
      };
    }
    case "event": {
      const e = action.event;
      switch (e.type) {
        case "start":
          return { ...initial(state.bill), running: true };
        case "stage":
          return { ...state, stages: { ...state.stages, [e.stage]: { status: e.status, detail: e.detail } } };
        case "bill":
          return { ...state, bill: e.bill };
        case "candidate":
        case "verified":
        case "verdict":
        case "sources":
          return {
            ...state,
            order: state.order.includes(e.finding.id) ? state.order : [...state.order, e.finding.id],
            findings: { ...state.findings, [e.finding.id]: e.finding },
          };
        case "trace":
          return { ...state, traces: [...state.traces, e.step] };
        case "notice":
          return { ...state, notices: [...state.notices, e.message] };
        case "summary":
          return { ...state, summary: e.summary };
        case "done":
          return { ...state, running: false };
        case "error":
          return { ...state, running: false, error: e.message };
        default:
          return state;
      }
    }
  }
}

function Pipeline({ stages, config }: { stages: Record<string, StageState>; config: Config | null }) {
  return (
    <ol className="grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-7" data-testid="pipeline">
      {STAGES.map((s, i) => {
        const st = stages[s.id];
        const tone =
          st.status === "done"
            ? "border-upheld/40 bg-upheld-bg"
            : st.status === "start"
              ? "border-pending/50 bg-pending-bg animate-pulse"
              : st.status === "skipped"
                ? "border-paper-line bg-dismissed-bg opacity-70"
                : "border-paper-line bg-white opacity-60";
        return (
          <li key={s.id} className={`rounded-lg border p-2 text-xs transition ${tone}`} data-testid={`stage-${s.id}`} data-status={st.status}>
            <div className="font-semibold text-ink">
              {i + 1}. {s.label}
            </div>
            <div className="text-ink-mute">{s.who(config)}</div>
            {st.detail && <div className="mt-1 text-ink-soft">{st.detail}</div>}
          </li>
        );
      })}
    </ol>
  );
}

const typeLabel: Record<string, string> = {
  duplicate: "Duplicate",
  math: "Arithmetic",
  rate: "Contract rate",
  fee: "Fee cap",
  unauthorized: "Not in agreement",
  other: "Other",
};

function FindingCard({ f }: { f: Finding }) {
  const amount = f.verified_amount ?? f.claimed_amount;
  const state = f.verdict ?? (f.verified === false ? "dismissed" : "pending");
  const tone =
    state === "upheld"
      ? "border-upheld/40 bg-upheld-bg"
      : state === "dismissed"
        ? "border-paper-line bg-dismissed-bg"
        : "border-pending/40 bg-pending-bg";
  return (
    <li className={`animate-rise rounded-xl border p-4 ${tone}`} data-testid="finding" data-verdict={state}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="mb-1 flex flex-wrap items-center gap-2 text-xs">
            <span className="chip bg-white/80 text-ink-soft border border-paper-line">{typeLabel[f.type] ?? f.type}</span>
            <span className="text-ink-mute">raised by {f.source === "llm" ? "Nemotron AI Auditor" : "rule-based Auditor"}</span>
          </div>
          <div className={`font-semibold ${state === "dismissed" ? "line-through text-dismissed" : ""}`}>{f.title}</div>
          <p className="mt-1 text-sm text-ink-soft">{f.rationale}</p>
        </div>
        <div className="shrink-0 text-right">
          <div className={`font-serif text-xl font-bold ${state === "upheld" ? "text-upheld" : state === "dismissed" ? "text-dismissed line-through" : "text-pending"}`}>
            {money(amount)}
          </div>
          <div className="text-xs font-semibold uppercase tracking-wide">
            {state === "upheld" ? (
              <span className="text-upheld">Upheld</span>
            ) : state === "dismissed" ? (
              <span className="text-dismissed">Dismissed</span>
            ) : (
              <span className="text-pending">On trial…</span>
            )}
          </div>
        </div>
      </div>
      {(f.verification_note || f.verdict_reason) && (
        <dl className="mt-3 space-y-1 border-t border-black/5 pt-2 text-sm">
          {f.verification_note && (
            <div className="flex gap-2">
              <dt className="w-20 shrink-0 text-xs font-semibold uppercase text-ink-mute">Verifier</dt>
              <dd className="text-ink-soft">{f.verification_note}</dd>
            </div>
          )}
          {f.verdict_reason && f.verified !== false && (
            <div className="flex gap-2">
              <dt className="w-20 shrink-0 text-xs font-semibold uppercase text-ink-mute">Skeptic</dt>
              <dd className="text-ink-soft" data-testid="skeptic-reason">
                {f.verdict_reason}
                {f.confidence != null && <span className="text-ink-mute"> ({Math.round(f.confidence * 100)}% confident)</span>}
              </dd>
            </div>
          )}
          {f.sources.length > 0 && (
            <div className="flex gap-2">
              <dt className="w-20 shrink-0 text-xs font-semibold uppercase text-ink-mute">Sources</dt>
              <dd className="space-y-0.5">
                {f.sources.map((s) => (
                  <a key={s.url} href={s.url} target="_blank" rel="noreferrer noopener" className="block truncate text-ink underline">
                    {s.title}
                  </a>
                ))}
              </dd>
            </div>
          )}
        </dl>
      )}
    </li>
  );
}

function BillTable({ bill, findings }: { bill: Bill; findings: Finding[] }) {
  const flagged = new Map<number, "upheld" | "dismissed" | "pending">();
  for (const f of findings) {
    const s = f.verdict ?? (f.verified === false ? "dismissed" : "pending");
    for (const i of f.line_indexes) if (flagged.get(i) !== "upheld") flagged.set(i, s);
  }
  return (
    <div className="card overflow-hidden" data-testid="bill">
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-paper-line px-4 py-3">
        <div>
          <div className="font-serif text-lg font-bold">{bill.provider}</div>
          <div className="text-xs text-ink-mute">
            Account {bill.account ?? "—"} · statement {bill.statement_date ?? "—"} {bill.customer ? `· ${bill.customer}` : ""}
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs text-ink-mute">Total due</div>
          <div className="font-mono font-semibold">{money(bill.total)}</div>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <tbody>
            {bill.line_items.map((li) => {
              const s = flagged.get(li.index);
              const tone = s === "upheld" ? "bg-red-50" : s === "pending" ? "bg-pending-bg" : "";
              return (
                <tr key={li.index} className={`border-b border-paper-line last:border-0 ${tone}`}>
                  <td className="w-8 px-4 py-2 font-mono text-xs text-ink-mute">{li.index}</td>
                  <td className="whitespace-nowrap px-2 py-2 text-xs text-ink-mute">{li.date}</td>
                  <td className="px-2 py-2">
                    {li.description}
                    {s === "upheld" && <span className="chip ml-2 bg-red-100 text-red-700">disputed</span>}
                  </td>
                  <td className="px-4 py-2 text-right font-mono">{money(li.amount)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="flex justify-end gap-6 border-t border-paper-line px-4 py-2 font-mono text-xs text-ink-mute">
        <span>Subtotal {money(bill.subtotal)}</span>
        {bill.tax != null && <span>Tax {money(bill.tax)}</span>}
      </div>
    </div>
  );
}

export function Tribunal(props: {
  docId: string;
  config: Config | null;
  autoRun?: boolean;
  onFinished: () => void;
  onOpenActions: () => void;
}) {
  const { docId, config, autoRun, onFinished, onOpenActions } = props;
  const [state, dispatch] = useReducer(reducer, null, () => initial(null));
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    abort.current?.abort();
    dispatch({ type: "reset", bill: null });
    (async () => {
      const [doc, list] = await Promise.all([api.document(docId), api.documents()]);
      if (cancelled) return;
      const bill = doc.kind === "bill" ? (doc.parsed as Bill) : null;
      dispatch({ type: "reset", bill });
      const last = list.find((d) => d.id === docId)?.last_audit_id;
      if (autoRun && bill) {
        void run();
      } else if (last) {
        const audit = await api.audit(last);
        if (!cancelled && audit.result) dispatch({ type: "restore", summary: audit.result, bill });
      }
    })().catch((e) => dispatch({ type: "fail", message: (e as Error).message }));
    return () => {
      cancelled = true;
    };
  }, [docId]);

  async function run() {
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    try {
      await streamAudit(docId, (event) => dispatch({ type: "event", event }), controller.signal);
    } catch (e) {
      if ((e as Error).name !== "AbortError") dispatch({ type: "fail", message: (e as Error).message });
    }
    onFinished();
  }

  const findings = state.order.map((id) => state.findings[id]);

  return (
    <section className="flex flex-col gap-4" data-testid="tribunal">
      {state.bill && <BillTable bill={state.bill} findings={findings} />}

      <div className="flex flex-wrap items-center gap-3">
        <button className="btn-go" onClick={() => void run()} disabled={state.running || !state.bill} data-testid="run-tribunal">
          {state.running ? "The Tribunal is in session…" : state.summary ? "Run the Tribunal again" : "Put this bill on trial"}
        </button>
        <span className="text-sm text-ink-mute">Auditors accuse, code verifies, the Skeptic cross-examines, then the Advocate writes.</span>
      </div>

      {(state.running || state.summary || state.order.length > 0) && <Pipeline stages={state.stages} config={config} />}

      {state.error && (
        <div className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-800" role="alert" data-testid="audit-error">
          {state.error}
        </div>
      )}
      {state.notices.map((n, i) => (
        <div key={i} className="rounded-lg border border-pending/30 bg-pending-bg p-2 text-xs text-pending">
          {n}
        </div>
      ))}

      {state.summary && !state.running && (
        <div
          className={`card flex flex-wrap items-center justify-between gap-3 p-5 ${state.summary.upheld ? "border-upheld/40" : ""}`}
          data-testid="verdict-banner"
        >
          {state.summary.upheld > 0 ? (
            <div>
              <div className="font-serif text-3xl font-bold text-upheld" data-testid="recoverable">
                {money(state.summary.recoverable_total)}
              </div>
              <div className="text-sm text-ink-soft">
                owed to you by {state.summary.provider}: {state.summary.upheld} error{state.summary.upheld === 1 ? "" : "s"} upheld,{" "}
                {state.summary.dismissed} false alarm{state.summary.dismissed === 1 ? "" : "s"} thrown out
              </div>
            </div>
          ) : (
            <div>
              <div className="font-serif text-2xl font-bold" data-testid="recoverable">
                This bill checks out.
              </div>
              <div className="text-sm text-ink-soft">
                {state.summary.dismissed} possible problem{state.summary.dismissed === 1 ? " was" : "s were"} raised and dismissed on
                cross-examination. No accusation without evidence.
              </div>
            </div>
          )}
          {state.summary.action_ids.length > 0 && (
            <button className="btn-primary" onClick={onOpenActions} data-testid="review-actions">
              Review {state.summary.action_ids.length} drafted action{state.summary.action_ids.length === 1 ? "" : "s"}
            </button>
          )}
        </div>
      )}

      {findings.length > 0 && (
        <ul className="flex flex-col gap-3" data-testid="findings">
          {[...findings].sort((a, b) => rank(a) - rank(b)).map((f) => (
            <FindingCard key={f.id} f={f} />
          ))}
        </ul>
      )}

      {state.traces.length > 0 && (
        <p className="text-xs text-ink-mute" data-testid="live-trace">
          {state.traces.length} model call{state.traces.length === 1 ? "" : "s"} ·{" "}
          {state.traces.reduce((n, t) => n + t.prompt_tokens + t.completion_tokens, 0).toLocaleString()} tokens ·{" "}
          {[...new Set(state.traces.map((t) => t.model))].join(", ")}
        </p>
      )}
    </section>
  );
}

function rank(f: Finding) {
  if (f.verdict === "upheld") return 0;
  if (f.verdict == null && f.verified !== false) return 1;
  return 2;
}
