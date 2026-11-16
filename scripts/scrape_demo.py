"""Step 2 demo: run the 5-step scraper on real homepages and show the token reduction.
Usage: python scripts/scrape_demo.py linear.app posthog.com sprinto.com"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from agent.config import settings  # noqa: E402
from pipeline.scraper import Fetcher, Scraper  # noqa: E402

console = Console(width=140)
fetcher = Fetcher(refresh="--refresh" in sys.argv)
domains = [d for d in sys.argv[1:] if not d.startswith("--")] or ["linear.app", "posthog.com", "sprinto.com"]
t = Table(title="Scraper: raw HTML vs cleaned text (homepage)")
for c in ("domain", "status", "method", "raw HTML tokens", "clean tokens", "reduction", "signals found"):
    t.add_column(c)
recs = []
for d in domains:
    rec = Scraper(d, fetcher).scrape(f"https://{d}/", page_type="home")
    recs.append((d, rec))
    sig = ", ".join(sorted({f"{s.kind}:{s.value}" for s in rec.signals}))[:70] or "-"
    red = f"{100 * (1 - rec.clean_tokens / rec.raw_tokens):.1f}%" if rec.raw_tokens else "-"
    t.add_row(d, rec.status if rec.status == "ok" else f"FAILED: {rec.failure.value}", rec.method,
              f"{rec.raw_tokens:,}", f"{rec.clean_tokens:,}", red, sig)
console.print(t)
ok = [r for _, r in recs if r.status == "ok" and r.raw_tokens]
if ok:
    s = settings()
    raw = sum(r.raw_tokens for r in ok) / len(ok)
    cl = sum(r.clean_tokens for r in ok) / len(ok)
    per = lambda tok: tok * 200 / 1e6 * s.price_in_per_m  # noqa: E731
    console.print(f"Input cost to send ONE homepage for 200 companies at ${s.price_in_per_m}/1M: "
                  f"raw HTML ≈ ${per(raw):.2f}  vs  cleaned ≈ ${per(cl):.3f}")
d, rec = next(((d, r) for d, r in recs if r.status == "ok"), (None, None))
if rec:
    console.print(f"\n[bold]Cleaned text the AI will read for {d}[/bold] (first 1,200 chars, saved in data/{d}/clean/home.md):")
    console.print(rec.clean_text[:1200], markup=False)
print("fetches:", fetcher.stats)
