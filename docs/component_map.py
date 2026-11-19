"""Draws the Sprinto Outbound Agent components map (same layout as the class template:
actor rows × Trigger / Input / Context / Processing / AI / Output columns) as SVG → HTML → PNG.

Run:  python docs/component_map.py      → docs/component_map.svg, docs/component_map.html, docs/component_map.png
"""
from __future__ import annotations

import html
import textwrap
from pathlib import Path

OUT = Path(__file__).resolve().parent

COLS = ["Trigger", "Input", "Context", "Processing", "AI", "Output"]
ROWS = [  # (actor, subtitle, lane colour, box fill, box stroke)
    ("Scheduler", "(cron-job.org → Vercel)", "#eef0fb", "#e6e6fb", "#6366f1"),
    ("Data Sources", "(prospect websites · Hunter.io)", "#eaf0fb", "#ffffff", "#3b82f6"),
    ("System", "(FastAPI on Vercel + Neon Postgres)", "#fdf6e3", "#fdefc4", "#d9a400"),
    ("AI Model", "(Gemini 3.5 Flash-Lite)", "#e8f5ec", "#d8f1e1", "#22a05a"),
    ("Anu", "(Growth Lead · reviewer)", "#fcecea", "#fde0d6", "#ea7a50"),
    ("Services", "(Gmail API · Google Calendar)", "#fbe9f1", "#f9d6e6", "#d63f86"),
    ("Prospect", "(MY_EMAIL stand-in)", "#eaf0fb", "#ffffff", "#3b82f6"),
    ("Agents", "(Seq 1 hand-built · Seq 2 LangGraph)", "#ecf4f4", "#d5eeee", "#0f8f8f"),
]

# id: (row, col, text)
BOXES = {
    "trig":   (0, 0, "Anu clicks New run (or domains.csv) → companies queued; every 5 min: process queue + read inbox"),
    "site":   (1, 1, "Prospect website: home + ≤4 pages (robots.txt, never bypasses bot blocks); Greenhouse/Lever jobs API"),
    "hunter": (1, 2, "Hunter.io Domain Search: named people + work emails (cached, ≤50/mo) · Apollo manual override"),
    "scrape": (2, 1, "5-step scraper: fetch → parse → extract → clean (~2k tokens/page) → store"),
    "config": (2, 2, "Config: ICP rubric + keyword variants, Sprinto approved claims, retry table & caps"),
    "rules":  (2, 3, "Rules in code (no AI): verify quotes · fit score /100 + short-circuits · PoC by company size · lint email"),
    "logs":   (2, 5, "Logs every decision in Postgres: pages, facts, verdicts, tool calls, CRM stage → dashboard"),
    "gemini": (3, 4, "Gemini: quoted facts · picks next page (≤4) · 2 pain points · drafts email · reads replies & times"),
    "learn":  (4, 4, "Feedback loop: your edits → lessons (pgvector) → retrieved per role → added to next drafts"),
    "final":  (4, 5, "FINAL CALL in Review queue: approve / edit / reject (reason req.); handles hand-offs"),
    "send":   (5, 5, "Gmail API sends ONLY to MY_EMAIL, in-thread · Calendar: free/busy, 30-min Meet + prep card"),
    "prosp":  (6, 5, "Receives email → replies: interested / not now / counter-time / silence"),
    "seq1":   (7, 3, "Seq 1 (code): classify reply → interested / not interested / no reply 5 days → close / unclear → Anu"),
    "seq2":   (7, 4, "Seq 2 LangGraph (checkpointed): check calendar → propose 3 times → await → book / counter ≤2 / escalate 48 h"),
    "booked": (7, 5, "Meeting booked: invite + Meet link, confirmation only after booking, CRM → meeting_booked 🎉"),
}

# (from, to, route): h = straight horizontal, v = straight vertical, vh = vertical then horizontal, hv = horizontal then vertical
EDGES = [
    ("trig", "site", "vh"), ("site", "scrape", "v"), ("scrape", "config", "h"), ("config", "rules", "h"),
    ("hunter", "rules", "hv"), ("rules", "gemini", "vh"), ("gemini", "final", "hv", -60),
    ("final", "learn", "h"), ("learn", "gemini", "v"), ("final", "logs", "v", 60), ("final", "send", "v"),
    ("send", "prosp", "v"), ("prosp", "seq1", "hv"), ("seq1", "seq2", "h"), ("seq2", "booked", "h"),
]

ACTOR_W, COL_W, HEAD_H, ROW_H, PAD_X, BOX_W, BOX_H = 250, 300, 70, 150, 24, 256, 116
TITLE_H, FOOT_H = 90, 120
W = PAD_X * 2 + ACTOR_W + COL_W * len(COLS)
H = TITLE_H + HEAD_H + ROW_H * len(ROWS) + FOOT_H
GRID_Y = TITLE_H


def box_rect(row: int, col: int):
    x = PAD_X + ACTOR_W + col * COL_W + (COL_W - BOX_W) / 2
    y = GRID_Y + HEAD_H + row * ROW_H + (ROW_H - BOX_H) / 2
    return x, y, BOX_W, BOX_H


def wrap(text: str, width: int = 31) -> list[str]:
    return textwrap.wrap(text, width=width)


def edge_path(a, b, route, dx=0):
    (ax, ay, aw, ah), (bx, by, bw, bh) = box_rect(*BOXES[a][:2]), box_rect(*BOXES[b][:2])
    acx, acy, bcx, bcy = ax + aw / 2, ay + ah / 2, bx + bw / 2 + dx, by + bh / 2
    if route == "v":
        acx += dx
    if route == "h":
        sx = ax + aw if bcx > acx else ax
        ex = bx if bcx > acx else bx + bw
        return f"M{sx},{acy} L{ex},{bcy}"
    if route == "v":
        sy = ay + ah if bcy > acy else ay
        ey = by if bcy > acy else by + bh
        return f"M{acx},{sy} L{bcx},{ey}"
    if route == "vh":   # leave top/bottom, turn, enter the target's side
        sy = ay + ah if bcy > acy else ay
        ex = bx if bcx > acx else bx + bw
        return f"M{acx},{sy} L{acx},{bcy} L{ex},{bcy}"
    if route == "hv":   # leave the side, turn, enter the target's top/bottom
        sx = ax + aw if bcx > acx else ax
        ey = by if bcy > acy else by + bh
        return f"M{sx},{acy} L{bcx},{acy} L{bcx},{ey}"
    raise ValueError(route)


def svg() -> str:
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Inter, Segoe UI, sans-serif">',
         '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
         '<path d="M0,0 L10,5 L0,10 z" fill="#1f2a44"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{W/2}" y="56" text-anchor="middle" font-size="34" font-weight="700" fill="#1f2a44">'
         'Sprinto · Outbound Agent — Components Map (research → email → review → reply → booked call)</text>']
    # header
    o.append(f'<rect x="{PAD_X}" y="{GRID_Y}" width="{W-2*PAD_X}" height="{HEAD_H}" rx="10" fill="#1f2a44"/>')
    o.append(f'<text x="{PAD_X+ACTOR_W/2}" y="{GRID_Y+HEAD_H/2+8}" text-anchor="middle" font-size="22" fill="#fff">Actor</text>')
    for i, c in enumerate(COLS):
        o.append(f'<text x="{PAD_X+ACTOR_W+i*COL_W+COL_W/2}" y="{GRID_Y+HEAD_H/2+8}" text-anchor="middle" font-size="22" fill="#fff">{c}</text>')
    # lanes
    for r, (actor, sub, lane, _, _) in enumerate(ROWS):
        y = GRID_Y + HEAD_H + r * ROW_H
        o.append(f'<rect x="{PAD_X+ACTOR_W}" y="{y}" width="{COL_W*len(COLS)}" height="{ROW_H}" fill="{lane}"/>')
        o.append(f'<rect x="{PAD_X}" y="{y}" width="{ACTOR_W}" height="{ROW_H}" fill="#1f2a44"/>')
        o.append(f'<line x1="{PAD_X}" y1="{y}" x2="{W-PAD_X}" y2="{y}" stroke="#2c3a5c" stroke-width="1"/>')
        o.append(f'<text x="{PAD_X+ACTOR_W/2}" y="{y+ROW_H/2-4}" text-anchor="middle" font-size="23" font-weight="700" fill="#fff">{html.escape(actor)}</text>')
        o.append(f'<text x="{PAD_X+ACTOR_W/2}" y="{y+ROW_H/2+22}" text-anchor="middle" font-size="14" fill="#c9d1e6">{html.escape(sub)}</text>')
    for i in range(1, len(COLS)):
        x = PAD_X + ACTOR_W + i * COL_W
        o.append(f'<line x1="{x}" y1="{GRID_Y+HEAD_H}" x2="{x}" y2="{GRID_Y+HEAD_H+ROW_H*len(ROWS)}" stroke="#d8dde8" stroke-width="1"/>')
    # edges (under boxes)
    for a, b, route, *dx in EDGES:
        o.append(f'<path d="{edge_path(a, b, route, *dx)}" fill="none" stroke="#1f2a44" stroke-width="2" stroke-dasharray="7 6" marker-end="url(#arr)"/>')
    # boxes
    for bid, (r, c, text) in BOXES.items():
        x, y, w, h = box_rect(r, c)
        _, _, _, fill, stroke = ROWS[r]
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2.2"/>')
        lines = wrap(text)
        lh = 19
        top = y + h / 2 - (len(lines) - 1) * lh / 2 + 5
        for i, line in enumerate(lines):
            o.append(f'<text x="{x+w/2}" y="{top+i*lh}" text-anchor="middle" font-size="14.5" fill="#1f2a44">{html.escape(line)}</text>')
    # footer: guardrails
    fy = GRID_Y + HEAD_H + ROW_H * len(ROWS) + 24
    o.append(f'<rect x="{PAD_X}" y="{fy}" width="{W-2*PAD_X}" height="{FOOT_H-40}" rx="12" fill="#f4f5fb" stroke="#d8dde8"/>')
    o.append(f'<text x="{PAD_X+24}" y="{fy+32}" font-size="17" font-weight="700" fill="#1f2a44">Guardrails (always on)</text>')
    o.append(f'<text x="{PAD_X+24}" y="{fy+60}" font-size="15" fill="#3a4563">'
             'Every email &amp; invite only to MY_EMAIL · AI suggests, code decides (quote check, rubric, lint) · never bypasses bot protection · '
             'no booking on a guessed time zone · caps: 4 extra pages, 2 counter-offers, 1 redraft · nothing reaches a prospect without Anu</text>')
    o.append("</svg>")
    return "\n".join(o)


def main():
    s = svg()
    (OUT / "component_map.svg").write_text(s)
    page = ("<!doctype html><meta charset='utf-8'><title>Components Map</title>"
            "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap' rel='stylesheet'>"
            f"<style>html,body{{margin:0;background:#fff}}</style>{s}")
    (OUT / "component_map.html").write_text(page)
    try:
        from playwright.sync_api import sync_playwright
        import os
        exe = "/opt/pw-browsers/chromium" if os.path.exists("/opt/pw-browsers/chromium") else None
        with sync_playwright() as p:
            b = p.chromium.launch(**({"executable_path": exe, "args": ["--no-sandbox"]} if exe else {}))
            pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
            pg.goto((OUT / "component_map.html").as_uri(), wait_until="networkidle")
            pg.screenshot(path=str(OUT / "component_map.png"), full_page=True)
            b.close()
        print("wrote docs/component_map.png")
    except Exception as e:  # noqa: BLE001
        print("SVG/HTML written; PNG needs playwright:", e)


if __name__ == "__main__":
    main()
