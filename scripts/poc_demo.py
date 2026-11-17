"""Step 5 demo: qualify, then find + rank the PoC (one real Hunter search per qualified domain, cached after)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.llm import LLM  # noqa: E402
from pipeline.contacts import find_contacts, hunter_credits  # noqa: E402
from pipeline.qualify import qualify  # noqa: E402
from pipeline.scraper import Fetcher  # noqa: E402

print(f"Hunter searches left (free check): {hunter_credits()}")
fetcher, llm = Fetcher(), LLM()
for d in sys.argv[1:]:
    q = qualify(d, llm, fetcher)
    print(f"\n=== {d}: {q.verdict.verdict} ({q.verdict.score}) ===")
    r = find_contacts(d, q.facts, q.verdict.verdict)
    print(f"PoC status: {r.status}   size: {r.size_bucket} ({r.size_source})   from cache: {r.from_cache}")
    if r.reason:
        print("  reason:", r.reason)
    if r.primary:
        print(f"  PRIMARY: {r.primary.first_name} {r.primary.last_name or ''} — {r.primary.position} — {r.primary.email}")
        print(f"     why: {r.primary.rank_reason}")
    if r.backup:
        print(f"  BACKUP:  {r.backup.first_name} {r.backup.last_name or ''} — {r.backup.position} — {r.backup.email}")
    for x in r.ranked:
        tag = f"#{x['rank']} {x['role']}" if "rank" in x else f"excluded: {x['excluded']}"
        print(f"    - {x['name'] or '(no name)'} | {x['position'] or '-'} | conf {x['confidence']} | {tag}")
print(f"\nHunter searches left now: {hunter_credits()}")
