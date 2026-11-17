"""Step 4 demo: qualify real domains and print verdict, score breakdown, reasons and loop steps.
Usage: python scripts/qualify_demo.py rocketlane.com atlan.com retool.com sprinto.com"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rich.console import Console  # noqa: E402

from agent.llm import LLM  # noqa: E402
from pipeline.qualify import qualify  # noqa: E402
from pipeline.scraper import Fetcher  # noqa: E402

con = Console(width=160)
fetcher, llm = Fetcher(), LLM()
summary = []
for d in sys.argv[1:]:
    r = qualify(d, llm, fetcher)
    v = r.verdict
    color = {"QUALIFIED": "green", "QUALIFIED_FLAGGED": "yellow"}.get(v.verdict, "red")
    con.rule(f"[bold]{d}[/bold] → [{color}]{v.verdict}[/{color}]  score {v.score}  {v.breakdown}  ({r.seconds}s)")
    for s in r.loop_steps:
        con.print(f"  loop {s.step}: missing={s.missing} → [{s.picked_by}] {s.pick} for {s.target_question} :: {s.outcome}"
                  + (f"\n          why: {s.reason}" if s.reason else ""), markup=False)
    for reason in v.reasons:
        con.print(f"  • {reason}", markup=False)
    if v.flags:
        con.print(f"  flags: {v.flags}   unanswered: {v.unanswered}", markup=False)
    summary.append((d, v.verdict, v.score, r.seconds))
con.rule("summary")
for row in summary:
    con.print(f"{row[0]:<22} {row[1]:<18} {row[2]:>4}   {row[3]}s")
u = llm.usage
con.print(f"LLM calls {u.calls} · tokens {u.tokens_in:,}/{u.tokens_out:,} · ${u.cost_usd:.4f} · fetches {fetcher.stats}")
