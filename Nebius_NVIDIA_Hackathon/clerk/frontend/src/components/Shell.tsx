import { useRef, useState } from "react";
import { api, money, type Config, type DocSummary, type Scoreboard } from "../api";

export function ModeBadge({ config }: { config: Config | null }) {
  if (!config) return null;
  if (config.mode === "live") {
    return (
      <span className="chip bg-nv/15 text-nv-dark border border-nv/40" data-testid="mode-badge" title={Object.values(config.models).join(" · ")}>
        <span className="h-2 w-2 rounded-full bg-nv animate-pulse" /> LIVE · NVIDIA Nemotron on Nebius Token Factory
      </span>
    );
  }
  return (
    <span
      className="chip bg-pending-bg text-pending border border-pending/30"
      data-testid="mode-badge"
      title="Model calls are answered by a deterministic simulator so the app runs without API keys."
    >
      OFFLINE SIMULATOR · not real model output
    </span>
  );
}

export function Header({ config, score }: { config: Config | null; score: Scoreboard | null }) {
  return (
    <header className="flex flex-wrap items-center justify-between gap-3 border-b border-paper-line bg-paper/80 px-5 py-3 backdrop-blur sticky top-0 z-10">
      <div className="flex items-center gap-3">
        <img src="/favicon.svg" alt="" className="h-8 w-8" />
        <div>
          <div className="font-serif text-xl font-bold leading-none">Clerk</div>
          <div className="text-xs text-ink-mute">audits your bills and fights back</div>
        </div>
        <ModeBadge config={config} />
      </div>
      {score && score.bills_audited > 0 && (
        <div className="flex items-center gap-4 text-sm" data-testid="scoreboard">
          <div>
            <span className="font-serif text-2xl font-bold text-upheld" data-testid="score-total">
              {money(score.recoverable_total)}
            </span>{" "}
            <span className="text-ink-mute">found across {score.bills_audited} bill{score.bills_audited === 1 ? "" : "s"}</span>
          </div>
          <div className="hidden sm:block text-ink-mute">
            <b className="text-ink">{score.upheld}</b> upheld · <b className="text-ink">{score.dismissed}</b> dismissed by the Skeptic
          </div>
        </div>
      )}
    </header>
  );
}

const kindLabel: Record<DocSummary["kind"], string> = { bill: "Bills", contract: "Agreements in your Vault", unknown: "Other documents" };

export function Sidebar(props: {
  docs: DocSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
  onChanged: (selectId?: string) => void;
  onError: (message: string) => void;
  samples: string[];
}) {
  const { docs, selected, onSelect, onChanged, onError, samples } = props;
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState<string | null>(null);

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    let lastId: string | undefined;
    for (const file of Array.from(files)) {
      setBusy(`Reading ${file.name}…`);
      try {
        const doc = await api.upload(file);
        if (doc.kind === "bill") lastId = doc.id;
      } catch (e) {
        onError(`${file.name}: ${(e as Error).message}`);
      }
    }
    setBusy(null);
    if (input.current) input.current.value = "";
    onChanged(lastId);
  }

  async function loadDemo() {
    setBusy("Loading Maria's documents…");
    try {
      await api.loadDemo();
      onChanged("first-bill");
    } catch (e) {
      onError((e as Error).message);
    }
    setBusy(null);
  }

  const groups = (["bill", "contract", "unknown"] as const).map((kind) => ({ kind, docs: docs.filter((d) => d.kind === kind) }));

  return (
    <aside className="flex w-full flex-col gap-4 lg:w-80 lg:shrink-0">
      <div
        className="card border-dashed p-4 text-center"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          void upload(e.dataTransfer.files);
        }}
      >
        <p className="text-sm font-semibold">Drop a bill or agreement</p>
        <p className="mb-3 text-xs text-ink-mute">.txt · .pdf · photos (live mode)</p>
        <input
          ref={input}
          type="file"
          multiple
          accept=".txt,.md,.pdf,.png,.jpg,.jpeg,.webp"
          className="hidden"
          data-testid="file-input"
          onChange={(e) => void upload(e.target.files)}
        />
        <div className="flex flex-col gap-2">
          <button className="btn-primary" onClick={() => input.current?.click()} disabled={!!busy}>
            Upload documents
          </button>
          <button className="btn-ghost" onClick={() => void loadDemo()} disabled={!!busy} data-testid="load-demo">
            Load demo: Maria's paperwork
          </button>
        </div>
        {busy && (
          <p className="mt-2 text-xs text-ink-mute" role="status">
            {busy}
          </p>
        )}
      </div>

      {groups.map(
        (g) =>
          g.docs.length > 0 && (
            <div key={g.kind}>
              <h3 className="mb-1 px-1 text-xs font-semibold uppercase tracking-wide text-ink-mute">{kindLabel[g.kind]}</h3>
              <ul className="flex flex-col gap-1" data-testid={`doc-list-${g.kind}`}>
                {g.docs.map((d) => (
                  <li key={d.id}>
                    <button
                      onClick={() => onSelect(d.id)}
                      className={`w-full rounded-lg px-3 py-2 text-left text-sm transition ${
                        selected === d.id ? "bg-white shadow-sm border border-paper-line" : "hover:bg-white/60"
                      }`}
                    >
                      <div className="font-semibold">{d.provider ?? "Unknown"}</div>
                      <div className="truncate text-xs text-ink-mute">{d.filename}</div>
                      {d.kind === "bill" && d.last_audit_id && <span className="chip mt-1 bg-upheld-bg text-upheld">audited</span>}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ),
      )}

      {samples.length > 0 && (
        <details className="px-1 text-xs text-ink-mute">
          <summary className="cursor-pointer font-semibold">Sample files to try uploading</summary>
          <ul className="mt-1 space-y-0.5">
            {samples.map((s) => (
              <li key={s}>
                <a className="underline hover:text-ink" href={`/api/samples/${s}`} download>
                  {s}
                </a>
              </li>
            ))}
          </ul>
        </details>
      )}
    </aside>
  );
}

export function Welcome({ onDemo }: { onDemo: () => void }) {
  return (
    <div className="card mx-auto max-w-2xl p-8 text-center" data-testid="welcome">
      <h1 className="font-serif text-4xl font-bold leading-tight">
        Drop in a bill.
        <br />
        Clerk finds the money you're owed.
      </h1>
      <p className="mx-auto mt-4 max-w-lg text-ink-soft">
        Clerk remembers your agreements, checks every line of every bill against them, and puts each accusation on trial.
        A Skeptic cross-examines it, plain code re-does the maths, and only then does Clerk draft the letter to get your
        money back. Nothing is sent without your approval.
      </p>
      <button className="btn-go mt-6 px-6 py-3 text-base" onClick={onDemo}>
        Try it with Maria's paperwork
      </button>
      <p className="mt-3 text-xs text-ink-mute">Demo data is synthetic. Your documents stay in your private workspace, and you can erase everything at any time.</p>
    </div>
  );
}
