#!/usr/bin/env python3
"""Plot a SCALE release-to-release comparison with its noise floor.

Input is the long-format CSV written by compare_regions.py --csv (run it with --min-us 0 so
small regions are kept in the totals). Needs only matplotlib.

  scripts/plot_version_diff.py --csv out/.../regions.csv --out-dir out/.../figures \\
      --new v174:cuda/scale-amd --old v173:cuda/scale-amd \\
      --ref-new v174:hip/hipcc --ref-old v173:hip/hipcc

Figures (PDF and PNG each):
  version_diff_ratio    per app, SCALE new/old for the kernel and total metrics. The grey band is the
                        noise floor for that row, max(floor-min, noise-mult x |reference ratio - 1|); the
                        diamond is the reference (HIP new/old, same code in both). Bars outside the band
                        are coloured: red = slower, green = faster.
  version_diff_regions  the regions with the largest absolute change that are outside their floor
  startup               runtime_initialization per app for every column in the CSV (or --init-cols)
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402


def load(path):
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        cols = header[3:]
        data = defaultdict(dict)  # app -> region -> {col: us}
        cls = {}
        for row in rd:
            app, region, c = row[0], row[1], row[2]
            cls[(app, region)] = c
            data[app][region] = {col: float(v) for col, v in zip(cols, row[3:]) if v != ""}
    return data, cls, cols


def metric(data, cls, app, col, which):
    vs = [v[col] for r, v in data[app].items()
          if col in v and (which == "total" or cls[(app, r)] == "kernel")]
    return sum(vs) if vs else None


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_dir / f"{name}.pdf", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--ref-new", required=True)
    ap.add_argument("--ref-old", required=True)
    ap.add_argument("--floor-min", type=float, default=0.10)
    ap.add_argument("--noise-mult", type=float, default=2.0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-us", type=float, default=20.0, help="ignore regions below this in the region figure")
    ap.add_argument("--init-cols", help="comma-separated columns for the start-up figure (default: all)")
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    data, cls, cols = load(a.csv)
    for c in (a.new, a.old, a.ref_new, a.ref_old):
        if c not in cols:
            sys.exit(f"column {c!r} not in CSV; available: {', '.join(cols)}")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    apps = sorted(data)

    # ---- figure 1: per-app ratios ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.38 * len(apps) + 2.2), sharey=True)
    for ax, which in zip(axes, ("kernel", "total")):
        rows = []
        for app in apps:
            vals = [metric(data, cls, app, c, which) for c in (a.new, a.old, a.ref_new, a.ref_old)]
            if any(v in (None, 0) for v in vals):
                continue
            r, noise = vals[0] / vals[1], vals[2] / vals[3]
            floor = max(a.floor_min, a.noise_mult * abs(noise - 1))
            rows.append((app, r, noise, floor))
        for i, (app, r, noise, floor) in enumerate(rows):
            ax.add_patch(Rectangle((1 - floor, i - 0.4), 2 * floor, 0.8, color="0.88", zorder=0))
            colour = "0.55"
            if r > 1 + floor:
                colour = "#c0392b"
            elif r < 1 - floor:
                colour = "#27ae60"
            ax.plot([1, r], [i, i], color=colour, lw=6, solid_capstyle="butt", zorder=2)
            ax.plot([noise], [i], marker="D", color="black", ms=5, zorder=3)
        ax.axvline(1, color="black", lw=0.8)
        ax.set_xscale("log")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows])
        ax.invert_yaxis()
        ax.set_title(f"{which} metric")
        ax.set_xlabel("ratio  (<1 faster, >1 slower)")
        lo, hi = ax.get_xlim()
        ax.set_xticks([t for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 2, 3, 5, 10)
                       if lo <= t <= hi])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"{a.title or 'SCALE release comparison'}\n{a.new} / {a.old}.  Grey band = noise floor, "
                 f"diamond = {a.ref_new} / {a.ref_old} (same code in both).", y=1.02, fontsize=10)
    save(fig, out, "version_diff_ratio")

    # ---- figure 2: biggest region movers -----------------------------------------------------
    movers = []
    for app in apps:
        for region, v in data[app].items():
            if any(c not in v or v[c] == 0 for c in (a.new, a.old, a.ref_new, a.ref_old)):
                continue
            if max(v[c] for c in (a.new, a.old, a.ref_new, a.ref_old)) < a.min_us:
                continue
            r, noise = v[a.new] / v[a.old], v[a.ref_new] / v[a.ref_old]
            floor = max(a.floor_min, a.noise_mult * abs(noise - 1))
            if abs(r - 1) > floor:
                movers.append((abs(v[a.new] - v[a.old]), app, region, v[a.new] - v[a.old], r))
    movers.sort(reverse=True)
    movers = movers[:a.top]
    fig, ax = plt.subplots(figsize=(9, 0.32 * max(len(movers), 3) + 1.5))
    labels = [f"{app}: {region}" for _, app, region, _, _ in movers]
    deltas = [d for _, _, _, d, _ in movers]
    ax.barh(range(len(movers)), deltas, color=["#c0392b" if d > 0 else "#27ae60" for d in deltas])
    ax.set_yticks(range(len(movers)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel(f"change in us per repeat, {a.new} minus {a.old} (red slower, green faster)")
    ax.set_title(f"Largest region changes outside the noise floor (top {len(movers)})")
    ax.grid(axis="x", alpha=0.3)
    save(fig, out, "version_diff_regions")

    # ---- figure 3: start-up ------------------------------------------------------------------
    scols = [c for c in (a.init_cols.split(",") if a.init_cols else cols)]
    present = [app for app in apps if "runtime_initialization" in data[app]]
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(present) + 2), 4.5))
    width = 0.8 / max(len(scols), 1)
    for k, c in enumerate(scols):
        xs, ys = [], []
        for i, app in enumerate(present):
            v = data[app]["runtime_initialization"].get(c)
            if v is not None:
                xs.append(i + (k - (len(scols) - 1) / 2) * width)
                ys.append(v / 1000.0)
        ax.bar(xs, ys, width=width, label=c)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=45, ha="right")
    ax.set_ylabel("runtime_initialization (ms)")
    ax.set_title("Device start-up by runtime and version")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    save(fig, out, "startup")


if __name__ == "__main__":
    main()
