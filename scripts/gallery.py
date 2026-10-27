"""Gallery page of an evaluation (scripts/evaluate.py output): renders + metrics against featured levels.

    python scripts/gallery.py --eval build/kernels/gd-levelgen-eval/out/eval --out gallery/smoke [--per-source 6]

Renders the copied levels (levels/<source>/*.txt) with the game's sprites (gdgen/render.py, needs the local GD
install; a few rows at a small scale, so it stays light) to <out>/img/*.jpg and writes <out>/index.html, a
self-contained page next to its images (publish both: the page plus img/ as supporting files).
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gdgen.evaluate import HISTOGRAMS, SCALARS  # noqa: E402
from gdgen.levelstring import level_text, parse_level  # noqa: E402

LABELS = {
    "objects": "objects", "gameplay": "gameplay objects", "deco": "decoration objects", "deco_share": "decoration share",
    "length": "length, units", "seconds": "duration, s", "gameplay_segments": "segments with gameplay",
    "empty_share": "empty segments", "gameplay_per_segment": "gameplay / segment", "deco_per_segment": "deco / segment",
    "max_gap": "longest gap, units", "y_spread": "vertical spread, units", "hazard_share": "hazard share",
    "interactive_per_segment": "orbs + pads / segment", "portals": "portals", "triggers": "triggers",
    "groups": "groups", "color_channels": "color channels",
    "categories": "categories", "ids_gameplay": "gameplay object ids", "ids_deco": "decoration object ids",
}

STYLE = """
/* Layout: a header band with each source's verdict, the metrics table against the featured reference,
   then one grid of renders per source */
:root {
  --bg: #edf0f5; --surface: #ffffff; --fg: #17202e; --muted: #596377; --line: #d3d9e4;
  --accent: #b55f00; --good: #1d7f47; --warn: #9a6a12; --bad: #b8382b; --bar: #c9d2e0;
  --display: "Chakra Petch", "Segoe UI", system-ui, sans-serif;
  --body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f141d; --surface: #161d2a; --fg: #e2e7f0; --muted: #8d98ae; --line: #263044;
  --accent: #f0a040; --good: #4cc17d; --warn: #e2b04b; --bad: #ef6a5a; --bar: #2c374d; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0f141d; --surface: #161d2a; --fg: #e2e7f0; --muted: #8d98ae; --line: #263044;
  --accent: #f0a040; --good: #4cc17d; --warn: #e2b04b; --bad: #ef6a5a; --bar: #2c374d; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 15px/1.55 var(--body); }
.wrap { max-width: 1180px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px));
        padding-right: max(16px, env(safe-area-inset-right, 0px)); }
header.band { border-bottom: 1px solid var(--line); background: var(--surface); }
header.band .wrap { padding-block: 28px 24px; display: grid; gap: 14px; }
.eyebrow { margin: 0; font: 600 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }
h1 { margin: 0; font: 600 clamp(26px, 4vw, 38px)/1.1 var(--display); text-wrap: balance; }
h2 { margin: 0 0 12px; font: 600 21px/1.2 var(--display); text-wrap: balance; }
.lede { margin: 0; max-width: 65ch; color: var(--muted); }
.sources { display: flex; flex-wrap: wrap; gap: 10px; }
.source { display: grid; gap: 2px; min-width: 0; padding: 10px 14px; border: 1px solid var(--line); border-radius: 6px; }
.source b { font: 600 15px/1.3 var(--display); }
.source span { font: 12.5px/1.4 var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; }
main.wrap { display: grid; gap: 40px; padding-block: 32px 56px; }
.table-scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font: 13px/1.4 var(--mono); font-variant-numeric: tabular-nums; }
th, td { padding: 7px 12px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; font-family: var(--body); }
thead th { font: 600 12px/1.3 var(--body); color: var(--muted); letter-spacing: .03em; }
tbody tr:last-child td { border-bottom: 0; }
td .range { color: var(--muted); }
.inref { display: inline-flex; align-items: center; gap: 6px; margin-left: 8px; color: var(--muted); }
.inref i { display: inline-block; width: 44px; height: 6px; border-radius: 3px; background: var(--bar); overflow: hidden; }
.inref i::before { content: ""; display: block; height: 100%; width: var(--w); background: var(--accent); }
.note { margin: 10px 0 0; max-width: 65ch; color: var(--muted); font-size: 13.5px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr)); gap: 16px; }
figure.level { margin: 0; min-width: 0; border: 1px solid var(--line); border-radius: 6px; overflow: hidden;
               background: var(--surface); display: grid; grid-template-rows: auto 1fr; }
figure.level img { display: block; width: 100%; height: auto; background: var(--line); }
figure.level .noimg { aspect-ratio: 10 / 3; display: grid; place-items: center; color: var(--muted); font-size: 13px; }
figcaption { padding: 10px 12px 12px; display: grid; gap: 8px; }
.title { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; min-width: 0; }
.title b { font: 600 14.5px/1.3 var(--display); overflow-wrap: anywhere; }
.pill { flex: none; padding: 2px 8px; border-radius: 999px; font: 600 11.5px/1.5 var(--mono); border: 1px solid currentColor; }
.pill.good { color: var(--good); } .pill.bad { color: var(--bad); } .pill.none { color: var(--muted); }
dl { margin: 0; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px 10px; }
dl div { min-width: 0; }
dt { font-size: 11.5px; color: var(--muted); }
dd { margin: 0; font: 13px/1.3 var(--mono); font-variant-numeric: tabular-nums; }
@media (max-width: 420px) { dl { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
"""


def fmt(v) -> str:
    if v is None:
        return "-"
    v = float(v)
    if abs(v) >= 10_000:
        return f"{v / 1000:.0f}k"
    if abs(v) >= 1000:
        return f"{v / 1000:.1f}k"
    if v == int(v):
        return str(int(v))
    return f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"


def sim_pill(m: dict) -> str:
    sim = m.get("sim_layout") or m.get("layout_sim")
    if not isinstance(sim, dict) or "passed" not in sim:
        return '<span class="pill none">not simulated</span>'
    if sim["passed"]:
        return '<span class="pill good">passable</span>'
    return f'<span class="pill bad">dies at {sim["percent"]:.0f}%</span>'


def render_images(eval_dir: Path, out: Path, rows: list[dict], per_source: int, n_rows: int, scale: float) -> dict:
    from gdgen.render import Renderer

    renderer, images = Renderer(), {}
    (out / "img").mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for m in rows:
        if "file" not in m or counts.get(m["source"], 0) >= per_source:
            continue
        counts[m["source"]] = counts.get(m["source"], 0) + 1
        t0 = time.time()
        try:
            header, objects = parse_level(level_text((eval_dir / m["file"]).read_text()))
            img = renderer.render(header, objects, scale=scale, max_rows=n_rows)
            rel = f"img/{m['source']}-{Path(m['file']).stem}.jpg"
            img.convert("RGB").save(out / rel, quality=80, optimize=True)
            images[m["file"]] = (rel, img.size)
            print(f"  {rel}: {img.size[0]}x{img.size[1]}, {time.time() - t0:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001  (a level the renderer cannot draw keeps its card without an image)
            print(f"  {m['file']}: {type(e).__name__}: {e}", flush=True)
    return images


def card(m: dict, images: dict) -> str:
    name = html.escape(str(m.get("name") or m.get("id")))
    img = images.get(m.get("file"))
    pic = (f'<img src="{img[0]}" width="{img[1][0]}" height="{img[1][1]}" loading="lazy" alt="Render of {name}">'
           if img else '<div class="noimg">no render</div>')
    facts = [("objects", fmt(m["objects"])), ("gameplay", fmt(m["gameplay"])),
             ("deco share", f"{100 * m['deco_share']:.0f}%"), ("duration", f"{m['seconds']:.0f} s")]
    dl = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in facts)
    return (f'<figure class="level">{pic}<figcaption><div class="title"><b>{name}</b>{sim_pill(m)}</div>'
            f"<dl>{dl}</dl></figcaption></figure>")


def page(title: str, summary: dict, rows: list[dict], images: dict, per_source: int) -> str:
    labels = (["ref"] if "ref" in summary else []) + [k for k in summary if k != "ref"]
    chips = []
    for lb in labels:
        s = summary[lb]
        parts = [f"{s['n']} levels"]
        if "sim_layout" in s:
            parts.append(f"{100 * s['sim_layout']['passed']:.0f}% passable gameplay")
        if "js" in s:
            parts.append(f"JS ids {s['js']['ids_gameplay']:.2f}")
        name = "featured reference" if lb == "ref" else lb
        chips.append(f'<div class="source"><b>{html.escape(name)}</b><span>{" · ".join(parts)}</span></div>')

    head = "".join(f"<th>{html.escape('featured' if lb == 'ref' else lb)}</th>" for lb in labels)
    body = []
    for name in SCALARS:
        cells = []
        for lb in labels:
            q = summary[lb]["metrics"][name]
            cell = f'{fmt(q["median"])} <span class="range">[{fmt(q["p10"])}–{fmt(q["p90"])}]</span>'
            if "in_ref_range" in summary[lb]:
                share = summary[lb]["in_ref_range"][name]
                cell += (f'<span class="inref" title="share inside the featured 10–90 % range">'
                         f'<i style="--w:{100 * share:.0f}%"></i>{100 * share:.0f}%</span>')
            cells.append(f"<td>{cell}</td>")
        body.append(f"<tr><td>{LABELS[name]}</td>{''.join(cells)}</tr>")
    for h in HISTOGRAMS:
        cells = "".join(f"<td>{summary[lb]['js'][h]:.3f}</td>" if "js" in summary[lb] else "<td>–</td>"
                        for lb in labels)
        body.append(f"<tr><td>JS distance, {LABELS[h]}</td>{cells}</tr>")
    for key, row in (("sim", "passable, whole level"), ("sim_layout", "passable, gameplay only")):
        cells = "".join(f"<td>{100 * summary[lb][key]['passed']:.0f}% "
                        f"<span class=\"range\">(median {summary[lb][key]['percent_median']:.0f}%)</span></td>"
                        if key in summary[lb] else "<td>–</td>" for lb in labels)
        body.append(f"<tr><td>{row}</td>{cells}</tr>")

    sections = []
    for lb in labels:
        cards = [card(m, images) for m in rows if m["source"] == lb and "file" in m][:per_source]
        if cards:
            heading = "Featured reference" if lb == "ref" else f"Generated: {html.escape(lb)}"
            sections.append(f'<section id="src-{html.escape(lb)}"><h2>{heading}</h2>'
                            f'<div class="grid">{"".join(cards)}</div></section>')
    n_ref = summary.get("ref", {}).get("n", 0)
    return f"""<title>Levelgen Gallery</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@600&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{STYLE}</style>
<header class="band"><div class="wrap">
<p class="eyebrow">gd-levelgen · evaluation</p>
<h1>{html.escape(title)}</h1>
<p class="lede">Generated levels next to {n_ref} random featured levels: the same metrics for both, the share of
generated levels that fall inside the featured 10–90 % range, and renders of the first rows of each level.</p>
<div class="sources">{"".join(chips)}</div>
</div></header>
<main class="wrap">
<section><h2>Metrics against featured levels</h2>
<div class="table-scroll"><table><thead><tr><th>metric</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>
<p class="note">Median with the 10–90 % range in brackets. The bar is the share of a source's levels inside the
featured 10–90 % range. JS distance compares the average object id and category mix (0 same, 1 disjoint).
The C++ simulator searches a path through the whole level (triggers included) and through its gameplay
objects alone; the second is what selection during generation checks, and what the badges show.</p>
</section>
{"".join(sections)}
</main>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="scripts/evaluate.py --out directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--per-source", type=int, default=6)
    ap.add_argument("--rows", type=int, default=3, help="render rows (2400 units each) per level")
    ap.add_argument("--scale", type=float, default=0.25)
    args = ap.parse_args()
    eval_dir, out = Path(args.eval), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (eval_dir / "metrics.jsonl").read_text().splitlines() if line]
    rows = [m for m in rows if "error" not in m]
    rows.sort(key=lambda m: (m["source"], str(m.get("name") or m.get("id"))))
    summary = json.loads((eval_dir / "summary.json").read_text())
    images = render_images(eval_dir, out, rows, args.per_source, args.rows, args.scale)
    title = args.title or ", ".join(k for k in summary if k != "ref") or "Evaluation"
    (out / "index.html").write_text(page(title, summary, rows, images, args.per_source))
    print(f"{out / 'index.html'}: {len(images)} renders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
