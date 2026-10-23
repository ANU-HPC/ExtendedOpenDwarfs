#!/usr/bin/env python3
"""Distribution diagnostics for LSB results: should we report mean, median or mode?

For every (app, region, column) it looks at two levels:

  across runs   the per-process value (mean of the per-repeat samples, as compare_regions.py
                normalises it) over the run files .r0, .r1, ... of one configuration
  within a run  the per-repeat samples inside one process

and prints, grouped by region class, how dispersed (CV) and how skewed (mean / median) they
are, plus the most dispersed rows with their raw per-process values.

  scripts/region_stats.py --dir v174=out/.../results-1.7.4 --dir v173=out/.../results-1.7.3 \\
      --size tiny --device mi300a --min-us 20
"""
import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_regions import classify, parse_name  # noqa: E402


def read_samples(path):
    """-> {region: [per-repeat sums in us]} (rows of one repeat are summed, like compare_regions.py)"""
    header, ri, ti, pi = None, None, None, None
    per = defaultdict(lambda: defaultdict(float))
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            parts = [p2 for p in parts
                     for p2 in (["NA"] * (len(p) // 2) if re.fullmatch(r"(?:NA){2,}", p) else [p])]
            if header is None:
                if "region" in parts and "time" in parts:
                    header = parts
                    ri, ti = parts.index("region"), parts.index("time")
                    pi = parts.index("repeats_to_two_seconds") if "repeats_to_two_seconds" in parts else None
                continue
            if len(parts) <= max(ri, ti):
                continue
            try:
                t = float(parts[ti])
            except ValueError:
                continue
            rep = 0
            if pi is not None and len(parts) > pi:
                try:
                    rep = int(parts[pi])
                except ValueError:
                    rep = 0
            per[parts[ri]][rep] += t
    return {r: list(d.values()) for r, d in per.items()}


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def cv(xs):
    m = statistics.mean(xs)
    return statistics.stdev(xs) / m if len(xs) > 1 and m else float("nan")


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-us", type=float, default=20.0, help="ignore rows whose largest value is below this")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--csv", help="also write every row (n, min, median, mean, max, std dev, CV, IQR) here")
    a = ap.parse_args()

    procs = defaultdict(list)  # (bench, region, col) -> [per-repeat sample lists, one per process]
    for spec in a.dir:
        label, path = spec.split("=", 1)
        for p in sorted(Path(path).expanduser().glob("lsb.*")):
            meta = parse_name(p.name)
            if not meta or (a.size and meta["size"] != a.size) or (a.device and meta["device"] != a.device):
                continue
            for region, samples in read_samples(p).items():
                procs[(meta["bench"], region, f"{label}:{meta['impl']}")].append(samples)

    across = defaultdict(list)   # class -> [(cv, mean/median)]
    within = defaultdict(list)   # class -> [mean/median of per-repeat samples]
    rows, allrows = [], []
    for (bench, region, col), plist in procs.items():
        pm = [statistics.mean(s) for s in plist if s]
        if len(pm) < 3 or max(pm) < a.min_us:
            continue
        cls = classify(region)
        med, avg = statistics.median(pm), statistics.mean(pm)
        across[cls].append((cv(pm), avg / med if med else float("nan")))
        sd = statistics.stdev(pm)
        iqr = pct(pm, .75) - pct(pm, .25)
        rows.append((max(pm) / min(pm) if min(pm) else float("inf"), bench, region, col, pm, med, avg, sd, iqr))
        allrows.append([bench, region, col, cls, len(pm), min(pm), med, avg, max(pm), sd, sd / avg if avg else "", iqr])
        for s in plist:
            if len(s) >= 5 and statistics.median(s) > 0:
                within[cls].append(statistics.mean(s) / statistics.median(s))

    def sh(xs, q):
        return f"{pct(xs, q):.3f}"

    print(f"size={a.size} device={a.device or 'any'}; rows with >=3 run files and a value >= {a.min_us:g} us\n")
    print("## Across run files (one value per process)\n")
    body = []
    for cls in sorted(across):
        cvs = [c for c, _ in across[cls] if c == c]
        sk = [s for _, s in across[cls] if s == s]
        share = sum(abs(s - 1) > 0.05 for s in sk) / len(sk) if sk else float("nan")
        body.append([cls, str(len(sk)), sh(cvs, .5), sh(cvs, .9), sh(cvs, 1.0),
                     sh(sk, .5), sh(sk, .9), sh(sk, 1.0), f"{share:.0%}"])
    print(table(body, ["class", "rows", "CV p50", "CV p90", "CV max", "mean/median p50", "p90", "max",
                       "|mean/median-1|>5%"]))
    print("\n## Within a run (per-repeat samples of one process; mean/median of those samples)\n")
    body = []
    for cls in sorted(within):
        xs = within[cls]
        share = sum(abs(s - 1) > 0.05 for s in xs) / len(xs)
        body.append([cls, str(len(xs)), sh(xs, .5), sh(xs, .9), sh(xs, 1.0), f"{share:.0%}"])
    print(table(body, ["class", "processes", "mean/median p50", "p90", "max", "|mean/median-1|>5%"]))
    print(f"\n## {a.top} most dispersed rows across run files (max/min)\n")
    rows.sort(key=lambda t: -t[0])
    print(table([[f"{sp:.2f}x", b, r, c, " ".join(f"{v:,.0f}" for v in pm), f"{med:,.0f}", f"{avg:,.0f}",
                  f"{sd:,.1f}", f"{sd / avg:.1%}" if avg else "-", f"{iqr:,.1f}"]
                 for sp, b, r, c, pm, med, avg, sd, iqr in rows[:a.top]],
                ["max/min", "app", "region", "column", "per-process values (us)", "median", "mean",
                 "std dev", "CV", "IQR"]))
    print("\nstd dev = sqrt(variance) across the run files, in us; CV = std dev / mean; IQR = Q3 - Q1.")
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["app", "region", "column", "class", "n_runs", "min_us", "median_us", "mean_us",
                        "max_us", "stdev_us", "cv", "iqr_us"])
            w.writerows(allrows)
        print(f"wrote {a.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
