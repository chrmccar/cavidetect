"""Build one self-contained HTML report from a cavidetect results directory.

Usage:
    python tools/build_report.py RESULTS_DIR OUTPUT.html

Scans RESULTS_DIR for per-stack subfolders (each with *_temporal.png, *_tubemap.png,
*_tubes.csv, *_manifest.json) and assembles a single browser-openable report grouped by plant.
"""
import os, sys, glob, json, base64
import pandas as pd


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def plant_of(stem):
    return stem.split("_")[0]


def main(res_dir, dest):
    folders = sorted(glob.glob(os.path.join(res_dir, "*")))
    sections, summary = {}, []
    for d in folders:
        man_p = glob.glob(os.path.join(d, "*_manifest.json"))
        if not man_p:
            continue
        stem = os.path.basename(man_p[0]).replace("_manifest.json", "")
        man = json.load(open(man_p[0]))
        tubes = pd.read_csv(os.path.join(d, f"{stem}_tubes.csv")) if os.path.exists(os.path.join(d, f"{stem}_tubes.csv")) else pd.DataFrame()
        cav = tubes[tubes["cavitated"]] if len(tubes) else tubes
        medf = int(cav["cav_frame"].median()) if len(cav) else 0
        summary.append(dict(stem=stem, plant=plant_of(stem), tubes=man.get("n_tubes", 0),
                            cavitated=man.get("n_cavitated", 0), medf=medf, sigma=round(man.get("noise_sigma_dn", 0), 2)))
        imgs = ""
        for suff in ("_temporal.png", "_tubemap.png"):
            p = os.path.join(d, f"{stem}{suff}")
            if os.path.exists(p):
                imgs += f'<img src="data:image/png;base64,{b64(p)}"/>'
        sections.setdefault(plant_of(stem), []).append(
            f'<div class=card><h3>{stem}</h3><p class=meta>{man.get("n_cavitated",0)}/{man.get("n_tubes",0)} tubes cavitated · median frame {medf} · σ={round(man.get("noise_sigma_dn",0),2)} DN</p>{imgs}</div>')

    sm = pd.DataFrame(summary)
    by = sm.groupby("plant").agg(stacks=("stem", "count"), tubes=("tubes", "sum"),
                                 cavitated=("cavitated", "sum")).reset_index()
    tbl = "<tr><th>plant</th><th>stacks</th><th>tubes</th><th>cavitated</th></tr>" + "".join(
        f"<tr><td>{r.plant}</td><td>{r.stacks}</td><td>{r.tubes}</td><td>{r.cavitated}</td></tr>" for r in by.itertuples())
    body = "".join(f"<h2>Plant {p}</h2><div class=grid>" + "".join(sections[p]) + "</div>" for p in sorted(sections))
    html = f"""<!doctype html><meta charset=utf-8><title>cavidetect report</title>
<style>body{{font-family:-apple-system,Arial;margin:26px;color:#222}} h1,h2{{color:#13506b}}
h2{{border-bottom:2px solid #13506b;padding-bottom:4px;margin-top:26px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px 10px;text-align:center}} th{{background:#13506b;color:#fff}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(620px,1fr));gap:16px}}
.card{{border:1px solid #ddd;border-radius:8px;padding:10px}} .card img{{width:100%;display:block;margin:4px 0}}
.meta{{color:#555;font-size:13px}}</style>
<h1>cavidetect — tube-based cavitation detection ({len(sm)} stacks)</h1>
<p>Each card: tubes embolising over time (red, spreading) + cumulative area curve, and the detected rhizoid tube map.</p>
<table>{tbl}</table>{body}
<p style='color:#666;font-size:13px'>Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}.</p>"""
    open(dest, "w").write(html)
    print("wrote", dest, round(os.path.getsize(dest) / 1e6, 1), "MB over", len(sm), "stacks")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
