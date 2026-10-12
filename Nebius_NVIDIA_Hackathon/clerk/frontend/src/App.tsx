import { useCallback, useEffect, useState } from "react";
import { api, type Action, type Config, type DocSummary, type Scoreboard } from "./api";
import { ActionsPanel, PermissionsPanel, TracePanel, VaultPanel } from "./components/Panels";
import { Header, Sidebar, Welcome } from "./components/Shell";
import { Tribunal } from "./components/Tribunal";

type Tab = "tribunal" | "vault" | "actions" | "trace" | "permissions";

function DocumentView({ docId }: { docId: string }) {
  const [doc, setDoc] = useState<{ filename: string; kind: string; text: string } | null>(null);
  useEffect(() => {
    void api.document(docId).then(setDoc);
  }, [docId]);
  if (!doc) return null;
  return (
    <div className="card p-4" data-testid="document-view">
      <p className="mb-3 text-sm text-ink-soft">
        {doc.kind === "contract"
          ? "Clerk read this agreement and saved its terms to your Vault. Every future bill from this company is checked against them."
          : "Clerk couldn't recognise this as a bill or an agreement, so there's nothing to audit."}
      </p>
      <pre className="max-h-[60vh] overflow-auto whitespace-pre-wrap rounded-lg bg-paper p-3 font-mono text-xs">{doc.text}</pre>
    </div>
  );
}

export default function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [docs, setDocs] = useState<DocSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [autoRun, setAutoRun] = useState(false);
  const [tab, setTab] = useState<Tab>("tribunal");
  const [score, setScore] = useState<Scoreboard | null>(null);
  const [actions, setActions] = useState<Action[]>([]);
  const [samples, setSamples] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  const refreshMeta = useCallback(() => {
    void api.scoreboard().then(setScore);
    void api.actions().then(setActions);
    void api.config().then(setConfig);
  }, []);

  const refreshDocs = useCallback(
    async (selectId?: string) => {
      const list = await api.documents();
      setDocs(list);
      setLoaded(true);
      if (selectId === "first-bill") {
        const bill = list.find((d) => d.kind === "bill");
        setSelected(bill?.id ?? null);
        setTab("tribunal");
      } else if (selectId) {
        setSelected(selectId);
        setTab("tribunal");
      } else if (selected && !list.some((d) => d.id === selected)) {
        setSelected(null);
      }
      refreshMeta();
    },
    [refreshMeta, selected],
  );

  useEffect(() => {
    api
      .config()
      .then((c) => {
        setConfig(c);
        return Promise.all([refreshDocs(), api.samples().then(setSamples)]);
      })
      .catch((e) => setError(`Can't reach Clerk's server: ${(e as Error).message}`));
  }, []);

  useEffect(() => {
    if (!error) return;
    const t = setTimeout(() => setError(null), 8000);
    return () => clearTimeout(t);
  }, [error]);

  const selectedDoc = docs.find((d) => d.id === selected) ?? null;
  const pending = actions.filter((a) => a.status === "pending").length;

  const tabs: { id: Tab; label: string }[] = [
    { id: "tribunal", label: "Tribunal" },
    { id: "vault", label: "Vault" },
    { id: "actions", label: pending ? `Actions (${pending})` : "Actions" },
    { id: "trace", label: "Trace" },
    { id: "permissions", label: "Permissions" },
  ];

  async function startDemo() {
    try {
      await api.loadDemo();
      await refreshDocs("first-bill");
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <div className="min-h-screen">
      <Header config={config} score={score} />
      {error && (
        <div className="mx-5 mt-3 rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-800" role="alert" data-testid="error">
          {error}
        </div>
      )}
      <div className="mx-auto flex max-w-7xl flex-col gap-6 px-5 py-6 lg:flex-row">
        <Sidebar
          docs={docs}
          selected={selected}
          samples={samples}
          onSelect={(id) => {
            setSelected(id);
            setAutoRun(false);
            setTab("tribunal");
          }}
          onChanged={(id) => {
            // A freshly uploaded bill goes straight on trial; the demo waits for a click.
            setAutoRun(!!id && id !== "first-bill");
            void refreshDocs(id);
          }}
          onError={setError}
        />
        <main className="min-w-0 flex-1">
          <nav className="mb-4 flex flex-wrap gap-1" aria-label="Sections">
            {tabs.map((t) => (
              <button
                key={t.id}
                className={`tab ${tab === t.id ? "tab-active" : ""}`}
                onClick={() => setTab(t.id)}
                data-testid={`tab-${t.id}`}
              >
                {t.label}
              </button>
            ))}
          </nav>

          {tab === "tribunal" &&
            (selectedDoc ? (
              selectedDoc.kind === "bill" ? (
                <Tribunal
                  key={selectedDoc.id}
                  docId={selectedDoc.id}
                  config={config}
                  autoRun={autoRun}
                  onFinished={() => void refreshDocs()}
                  onOpenActions={() => setTab("actions")}
                />
              ) : (
                <DocumentView key={selectedDoc.id} docId={selectedDoc.id} />
              )
            ) : loaded && docs.length === 0 ? (
              <Welcome onDemo={() => void startDemo()} />
            ) : (
              <p className="text-sm text-ink-mute">Pick a bill on the left to put it on trial.</p>
            ))}
          {tab === "vault" && <VaultPanel onError={setError} onChanged={() => void refreshDocs()} />}
          {tab === "actions" && <ActionsPanel onError={setError} onChanged={refreshMeta} />}
          {tab === "trace" && <TracePanel config={config} />}
          {tab === "permissions" && <PermissionsPanel config={config} onChanged={refreshMeta} />}
        </main>
      </div>
      <footer className="px-5 pb-6 text-center text-xs text-ink-mute">
        Clerk is an open-source hackathon project built with NVIDIA Nemotron on Nebius Token Factory. It is not legal or financial
        advice; check each finding before you send anything.
      </footer>
    </div>
  );
}
