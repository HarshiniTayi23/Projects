import { useEffect, useState } from "react";
import { api, money, type Action, type Config, type Fact, type TraceStep } from "../api";

const factLabel: Record<string, string> = {
  account: "Account",
  plan_name: "Plan",
  monthly_rate: "Monthly rate ($)",
  rate_locked_until: "Rate locked until",
  included_services: "Included services",
  late_fee_cap: "Late fee cap ($)",
  dispute_window_days: "Dispute window (days)",
  note: "Note",
};

export function VaultPanel({ onError, onChanged }: { onError: (m: string) => void; onChanged: () => void }) {
  const [facts, setFacts] = useState<Fact[]>([]);
  const [editing, setEditing] = useState<{ id: string; value: string } | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);

  const refresh = () => api.facts().then(setFacts).catch((e) => onError(e.message));
  useEffect(() => {
    void refresh();
  }, []);

  const byProvider = facts.reduce<Record<string, Fact[]>>((acc, f) => {
    (acc[f.provider] ??= []).push(f);
    return acc;
  }, {});

  async function save() {
    if (!editing) return;
    try {
      await api.editFact(editing.id, editing.value);
      setEditing(null);
      await refresh();
    } catch (e) {
      onError((e as Error).message);
    }
  }

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setAsking(true);
    try {
      setAnswer((await api.ask(question)).answer);
    } catch (err) {
      onError((err as Error).message);
    }
    setAsking(false);
  }

  return (
    <section className="flex flex-col gap-4" data-testid="vault">
      <div className="card p-4">
        <h2 className="font-serif text-xl font-bold">The Vault</h2>
        <p className="text-sm text-ink-soft">
          Everything Clerk remembers from your agreements. Edit a fact and the next audit uses it. Nothing here is shared or used
          for training.
        </p>
        <form onSubmit={(e) => void ask(e)} className="mt-3 flex gap-2">
          <input
            className="flex-1 rounded-lg border border-paper-line px-3 py-2 text-sm"
            placeholder="Ask Clerk, e.g. “What's my internet rate locked at?”"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            maxLength={500}
            data-testid="ask-input"
          />
          <button className="btn-primary" disabled={asking} data-testid="ask-button">
            {asking ? "Thinking…" : "Ask"}
          </button>
        </form>
        {answer && (
          <pre className="mt-3 whitespace-pre-wrap rounded-lg bg-paper p-3 font-sans text-sm" data-testid="ask-answer">
            {answer}
          </pre>
        )}
      </div>

      {Object.keys(byProvider).length === 0 && <p className="text-sm text-ink-mute">Upload an agreement and Clerk will remember its terms.</p>}

      {Object.entries(byProvider).map(([provider, rows]) => (
        <div key={provider} className="card overflow-hidden">
          <div className="border-b border-paper-line px-4 py-2 font-semibold">{provider}</div>
          <table className="w-full text-sm">
            <tbody>
              {rows.map((f) => (
                <tr key={f.id} className="border-b border-paper-line last:border-0" data-testid={`fact-${f.key}`}>
                  <td className="w-48 px-4 py-2 text-ink-mute">{factLabel[f.key] ?? f.key}</td>
                  <td className="px-2 py-2">
                    {editing?.id === f.id ? (
                      <input
                        autoFocus
                        className="w-full rounded border border-paper-line px-2 py-1"
                        value={editing.value}
                        onChange={(e) => setEditing({ id: f.id, value: e.target.value })}
                        onKeyDown={(e) => e.key === "Enter" && void save()}
                        data-testid="fact-edit-input"
                      />
                    ) : (
                      <span className="font-mono text-xs">{f.value}</span>
                    )}
                  </td>
                  <td className="whitespace-nowrap px-4 py-2 text-right">
                    {editing?.id === f.id ? (
                      <button className="text-xs font-semibold text-upheld" onClick={() => void save()} data-testid="fact-save">
                        Save
                      </button>
                    ) : (
                      <>
                        <button className="mr-3 text-xs font-semibold underline" onClick={() => setEditing({ id: f.id, value: f.value })}>
                          Edit
                        </button>
                        <button
                          className="text-xs text-red-700 underline"
                          onClick={() => api.deleteFact(f.id).then(refresh).catch((e) => onError(e.message))}
                        >
                          Forget
                        </button>
                      </>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}

      <div className="flex flex-wrap gap-2">
        <a className="btn-ghost" href="/api/vault/export" download>
          Export my Vault (JSON)
        </a>
        <button
          className="btn-ghost text-red-700"
          data-testid="wipe"
          onClick={() => {
            if (window.confirm("Erase every document, memory and action in this workspace? This can't be undone.")) {
              api.wipe().then(() => {
                void refresh();
                onChanged();
              });
            }
          }}
        >
          Forget everything
        </button>
      </div>
    </section>
  );
}

const actionIcon: Record<Action["type"], string> = { letter: "✉️", call_script: "📞", reminder: "📅" };

export function ActionsPanel({ onError, onChanged }: { onError: (m: string) => void; onChanged: () => void }) {
  const [actions, setActions] = useState<Action[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  const refresh = () => api.actions().then(setActions).catch((e) => onError(e.message));
  useEffect(() => {
    void refresh();
  }, []);

  async function decide(id: string, decision: "approve" | "reject") {
    try {
      await api.decide(id, decision);
      await refresh();
      onChanged();
    } catch (e) {
      onError((e as Error).message);
    }
  }

  return (
    <section className="flex flex-col gap-3" data-testid="actions">
      <div className="card p-4">
        <h2 className="font-serif text-xl font-bold">Approval queue</h2>
        <p className="text-sm text-ink-soft">Clerk drafts; you decide. Nothing leaves this page until you approve it and download it yourself.</p>
      </div>
      {actions.length === 0 && <p className="text-sm text-ink-mute">No actions yet. Put a bill on trial first.</p>}
      {[...actions].reverse().map((a) => (
        <div key={a.id} className="card p-4" data-testid="action" data-status={a.status} data-type={a.type}>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <button className="text-left" onClick={() => setOpen(open === a.id ? null : a.id)}>
              <span className="mr-2">{actionIcon[a.type]}</span>
              <span className="font-semibold">{a.title}</span>
              <span
                className={`chip ml-2 ${
                  a.status === "approved" ? "bg-upheld-bg text-upheld" : a.status === "rejected" ? "bg-dismissed-bg text-dismissed" : "bg-pending-bg text-pending"
                }`}
              >
                {a.status}
              </span>
            </button>
            <div className="flex gap-2">
              {a.status !== "approved" && (
                <button className="btn-go py-1" onClick={() => void decide(a.id, "approve")} data-testid="approve">
                  Approve
                </button>
              )}
              {a.status === "pending" && (
                <button className="btn-ghost py-1" onClick={() => void decide(a.id, "reject")} data-testid="reject">
                  Reject
                </button>
              )}
              {a.status === "approved" && (
                <a className="btn-primary py-1" href={`/api/actions/${a.id}/download`} download={a.filename} data-testid="download">
                  Download {a.filename.split(".").pop()}
                </a>
              )}
            </div>
          </div>
          {(open === a.id || a.type === "letter") && (
            <pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap rounded-lg bg-paper p-3 text-xs" data-testid="action-content">
              {a.type === "reminder" ? a.content.split(/\r?\n/).filter((l) => /^(SUMMARY|DTSTART|DESCRIPTION)/.test(l)).join("\n") : a.content}
            </pre>
          )}
        </div>
      ))}
    </section>
  );
}

export function TracePanel({ config }: { config: Config | null }) {
  const [rows, setRows] = useState<(TraceStep & { id: string; created: number })[]>([]);
  useEffect(() => {
    void api.traces().then(setRows);
  }, []);
  const tokens = rows.reduce((n, r) => n + r.prompt_tokens + r.completion_tokens, 0);
  const costs = rows.map((r) => r.cost_usd);
  const cost = costs.every((c) => c != null) && costs.length ? costs.reduce((a, b) => a! + b!, 0) : null;
  return (
    <section className="flex flex-col gap-3" data-testid="trace">
      <div className="card p-4">
        <h2 className="font-serif text-xl font-bold">Glass box</h2>
        <p className="text-sm text-ink-soft">
          Every model call Clerk made, with the model that ran it, on {config?.mode === "live" ? "Nebius Token Factory" : "the offline simulator"}.
        </p>
        <div className="mt-2 flex flex-wrap gap-4 text-sm">
          <span>
            <b>{rows.length}</b> calls
          </span>
          <span>
            <b>{tokens.toLocaleString()}</b> tokens
          </span>
          <span>
            cost <b>{cost == null ? "not configured" : `$${cost.toFixed(4)}`}</b>
          </span>
        </div>
        {config && (
          <ul className="mt-2 font-mono text-xs text-ink-mute">
            <li>Super: {config.models.super}</li>
            <li>Nano: {config.models.nano}</li>
            <li>Vision: {config.models.vision}</li>
          </ul>
        )}
      </div>
      <div className="card overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="border-b border-paper-line text-left text-xs uppercase text-ink-mute">
            <tr>
              <th className="px-4 py-2">Agent</th>
              <th className="px-2 py-2">Model</th>
              <th className="px-2 py-2 text-right">Tokens in/out</th>
              <th className="px-2 py-2 text-right">Latency</th>
              <th className="px-4 py-2 text-right">Cost</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="border-b border-paper-line last:border-0">
                <td className="px-4 py-1.5 font-semibold">{r.agent}</td>
                <td className="px-2 py-1.5 font-mono text-xs">{r.model}</td>
                <td className="px-2 py-1.5 text-right font-mono text-xs">
                  {r.prompt_tokens}/{r.completion_tokens}
                </td>
                <td className="px-2 py-1.5 text-right font-mono text-xs">{r.latency_ms} ms</td>
                <td className="px-4 py-1.5 text-right font-mono text-xs">{r.cost_usd == null ? "—" : money(r.cost_usd)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function PermissionsPanel({ config, onChanged }: { config: Config | null; onChanged: () => void }) {
  const [skills, setSkills] = useState(config?.skills ?? []);
  useEffect(() => setSkills(config?.skills ?? []), [config]);

  async function toggle(id: string, enabled: boolean) {
    setSkills((prev) => prev.map((s) => (s.id === id ? { ...s, enabled } : s)));
    try {
      await api.setPermission(id, enabled);
    } catch {
      setSkills((prev) => prev.map((s) => (s.id === id ? { ...s, enabled: !enabled } : s)));
    }
    onChanged();
  }

  if (!config) return null;
  return (
    <section className="flex flex-col gap-3" data-testid="permissions">
      <div className="card p-4">
        <h2 className="font-serif text-xl font-bold">What Clerk may do</h2>
        <p className="text-sm text-ink-soft">Every skill is yours to switch off. Changes apply to the next audit.</p>
      </div>
      <div className="card divide-y divide-paper-line">
        {skills.map((s) => (
          <label key={s.id} className="flex cursor-pointer items-center justify-between gap-4 px-4 py-3">
            <span className="text-sm">{s.description}</span>
            <input
              type="checkbox"
              className="h-5 w-5 accent-[#76b900]"
              checked={s.enabled}
              data-testid={`perm-${s.id}`}
              onChange={(e) => void toggle(s.id, e.target.checked)}
            />
          </label>
        ))}
      </div>
    </section>
  );
}
