"""Clerk eval lab: run the Tribunal over synthetic bills with known, planted errors.

    python -m evals.run_eval                 # offline simulator (CI): checks the pipeline, NOT model quality
    CLERK_MODE=live python -m evals.run_eval # real Nemotron on Nebius: the numbers for the README
    CLERK_MODE=live python -m evals.run_eval --all-super   # ablation: Super for every step (no routing)

One run yields three ablations for free:
  * candidates  - every problem the auditors raised (no Verifier, no Skeptic)
  * verified    - after the Verifier's arithmetic check
  * tribunal    - after the Skeptic's cross-examination (what users see)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clerk import ingest
from clerk.agents.runtime import AgentContext
from clerk.agents.tribunal import run_tribunal
from clerk.config import Settings, load_settings
from clerk.db import Database
from clerk.llm import LLMProvider, LLMRequest, LLMResult, build_provider
from clerk.schemas import Bill, Finding
from clerk.search import OfflineSearch
from clerk.synth import Case, generate_suite

RESULTS_DIR = Path(__file__).parent / "results"
TOLERANCE = 0.011


class AllSuper:
    """Ablation wrapper: route every 'nano' call to the Super model."""

    def __init__(self, inner: LLMProvider):
        self.inner = inner
        self.mode = inner.mode

    def complete(self, request: LLMRequest) -> LLMResult:
        if request.role == "nano":
            request = LLMRequest(**{**request.__dict__, "role": "super"})
        return self.inner.complete(request)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed(texts)

    def model_for(self, role: str) -> str:
        return self.inner.model_for("super" if role == "nano" else role)


@dataclass
class Score:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    found_dollars: float = 0.0
    false_dollars: float = 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "true_positives": self.tp,
            "false_positives": self.fp,
            "missed": self.fn,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1": round(self.f1, 3),
            "dollars_correctly_found": round(self.found_dollars, 2),
            "dollars_wrongly_accused": round(self.false_dollars, 2),
        }


@dataclass
class Totals:
    scores: dict[str, Score] = field(
        default_factory=lambda: {k: Score() for k in ("candidates", "verified", "tribunal")}
    )
    planted_dollars: float = 0.0
    seconds: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = 0.0
    failures: list[str] = field(default_factory=list)


def matches(pred: Finding, planted: dict[str, Any]) -> bool:
    if pred.type != planted["type"]:
        return False
    if planted["type"] == "math":
        return pred.meta.get("check") == planted.get("check")
    return bool(set(pred.line_indexes) & set(planted["line_indexes"]))


def score(preds: list[Finding], planted: list[dict[str, Any]], s: Score) -> None:
    unmatched = list(planted)
    for pred in preds:
        hit = next((p for p in unmatched if matches(pred, p)), None)
        if hit is None:
            s.fp += 1
            s.false_dollars += pred.amount
        else:
            unmatched.remove(hit)
            s.tp += 1
            if abs(pred.amount - hit["amount"]) <= TOLERANCE:
                s.found_dollars += hit["amount"]
    s.fn += len(unmatched)


def run_case(case: Case, settings: Settings, llm: LLMProvider, totals: Totals) -> dict[str, Any]:
    db = Database(":memory:")
    ctx = AgentContext(
        settings=settings,
        llm=llm,
        search=OfflineSearch(),
        db=db,
        workspace="eval",
        permissions={"web_research": False, "draft_letters": False, "reminders": False},
    )
    started = time.perf_counter()
    ingest.ingest(ctx, "agreement.txt", case.contract_text.encode())
    bill_doc = ingest.ingest(ctx, "statement.txt", case.bill_text.encode())
    doc = db.get_document("eval", bill_doc["id"])
    if not doc or doc["kind"] != "bill":
        raise RuntimeError("Reader did not recognise the statement as a bill")
    ctx.audit_id = "eval"
    summary = run_tribunal(ctx, Bill.from_dict(doc["parsed"]), today=dt.date(2026, 10, 1))
    totals.seconds += time.perf_counter() - started

    findings = [Finding.from_dict(f) for f in summary["findings"]]
    score(findings, case.planted, totals.scores["candidates"])
    score([f for f in findings if f.verified], case.planted, totals.scores["verified"])
    upheld = [f for f in findings if f.verdict == "upheld"]
    score(upheld, case.planted, totals.scores["tribunal"])
    totals.planted_dollars += sum(p["amount"] for p in case.planted)

    for t in db.list_traces("eval"):
        totals.llm_calls += 1
        totals.prompt_tokens += t["prompt_tokens"]
        totals.completion_tokens += t["completion_tokens"]
        if t["cost_usd"] is None:
            totals.cost_usd = None
        elif totals.cost_usd is not None:
            totals.cost_usd += t["cost_usd"]
    return {
        "case_id": case.case_id,
        "planted": case.planted,
        "decoys": case.decoys,
        "upheld": [{"type": f.type, "lines": f.line_indexes, "amount": f.amount, "title": f.title} for f in upheld],
    }


def render_markdown(report: dict[str, Any]) -> str:
    mode = report["mode"]
    banner = (
        "> **Offline simulator run.** These numbers check that the pipeline is wired correctly. They are NOT a "
        "measure of Nemotron's quality. Run with `CLERK_MODE=live` for real numbers."
        if mode == "offline"
        else f"> **Live run** on Nebius Token Factory. Models: {', '.join(f'{k}: `{v}`' for k, v in report['models'].items())}."
    )
    rows = [
        "| Stage | Precision | Recall | F1 | Correct $ found | $ wrongly accused |",
        "|---|---|---|---|---|---|",
    ]
    labels = {
        "candidates": "Auditors only",
        "verified": "+ Verifier (code)",
        "tribunal": "+ Skeptic (what users see)",
    }
    for key, label in labels.items():
        s = report["scores"][key]
        rows.append(
            f"| {label} | {s['precision']:.0%} | {s['recall']:.0%} | {s['f1']:.2f} | "
            f"${s['dollars_correctly_found']:,.2f} | ${s['dollars_wrongly_accused']:,.2f} |"
        )
    cost = report["cost_usd"]
    cost_text = f"${cost:.4f}" if isinstance(cost, (int, float)) else "not configured (set CLERK_PRICE_* env vars)"
    return "\n".join(
        [
            f"# Clerk eval results ({mode})",
            "",
            banner,
            "",
            f"- Cases: **{report['cases']}** synthetic statements, **{report['planted_errors']}** planted errors "
            f"worth **${report['planted_dollars']:,.2f}**, plus {report['decoys']} decoys (legitimate charges that look wrong)",
            f"- Routing: {'Super for every step (ablation)' if report['all_super'] else 'Nano for reading/auditing, Super for judging/writing'}",
            f"- Model calls: {report['llm_calls']} · tokens: {report['prompt_tokens']:,} in / {report['completion_tokens']:,} out · cost: {cost_text}",
            f"- Average time per bill: {report['seconds_per_case']:.2f}s",
            f"- Failures: {len(report['failures'])}",
            "",
            *rows,
            "",
            f"_Generated {report['generated_at']} by `python -m evals.run_eval`._",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=50, help="number of synthetic cases")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--all-super", action="store_true", help="ablation: use the Super model for every step")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    settings = load_settings()
    llm: LLMProvider = build_provider(settings)
    if args.all_super:
        llm = AllSuper(llm)
    cases = generate_suite(args.n, args.seed)
    totals = Totals()
    per_case = []
    for i, case in enumerate(cases, 1):
        try:
            per_case.append(run_case(case, settings, llm, totals))
        except Exception as exc:  # keep going; failures are reported
            totals.failures.append(f"{case.case_id}: {exc}")
        print(f"\r{i}/{len(cases)} cases", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    report = {
        "mode": llm.mode,
        "models": {role: llm.model_for(role) for role in ("super", "nano")},
        "all_super": args.all_super,
        "cases": len(cases),
        "planted_errors": sum(len(c.planted) for c in cases),
        "planted_dollars": round(totals.planted_dollars, 2),
        "decoys": sum(len(c.decoys) for c in cases),
        "scores": {k: s.to_dict() for k, s in totals.scores.items()},
        "llm_calls": totals.llm_calls,
        "prompt_tokens": totals.prompt_tokens,
        "completion_tokens": totals.completion_tokens,
        "cost_usd": round(totals.cost_usd, 4) if totals.cost_usd is not None else None,
        "seconds_per_case": totals.seconds / max(1, len(cases) - len(totals.failures)),
        "failures": totals.failures,
        "generated_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "per_case": per_case,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{report['mode']}{'-allsuper' if args.all_super else ''}"
    (args.out / f"{tag}.json").write_text(json.dumps(report, indent=2))
    (args.out / f"{tag}.md").write_text(render_markdown(report))
    print(render_markdown(report))
    return report


if __name__ == "__main__":
    main()
