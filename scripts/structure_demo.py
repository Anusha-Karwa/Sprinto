"""Step 3 demo: scrape real pages, let Gemini extract quoted facts, and show what code verified/rejected.
Usage: python scripts/structure_demo.py retool.com linear.app"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rich.console import Console  # noqa: E402

from agent.llm import LLM  # noqa: E402
from pipeline.scraper import Fetcher, Scraper  # noqa: E402
from pipeline.structure import structure  # noqa: E402

con = Console(width=150)
fetcher = Fetcher()
llm = LLM()
for d in [a for a in sys.argv[1:] if not a.startswith("-")] or ["retool.com", "linear.app"]:
    s = Scraper(d, fetcher)
    pages = [s.scrape(f"https://{d}/", "home"), s.scrape(f"https://{d}/pricing")]
    con.rule(f"[bold]{d}[/bold]  pages: " + ", ".join(f"{p.url} ({p.status})" for p in pages))
    facts, rejected, out = structure(llm, d, pages)
    if not out.ok:
        con.print(f"[red]STRUCTURE FAILED: {out.error_class} → {out.action}[/red]")
        continue
    for name, val in facts:
        items = val if isinstance(val, list) else ([val] if val else [])
        for it in items:
            label = f"{it.name} ({it.status})" if name == "certifications" else it.value
            con.print(f"[green]✔[/green] {name}: [bold]{label}[/bold]\n    “{it.quote}”  — {it.source_url}", markup=True)
    for r in rejected:
        con.print(f"[red]✘ rejected[/red] {r.field}={r.value!r}: {r.reason}\n    “{r.quote}”")
    if not rejected:
        con.print("[dim](no claims rejected)[/dim]")
u = llm.usage
con.print(f"\nLLM calls: {u.calls} · tokens in/out: {u.tokens_in:,}/{u.tokens_out:,} · cost ${u.cost_usd:.4f}")
