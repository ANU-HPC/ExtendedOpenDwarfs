#!/usr/bin/env python3
"""Per-sample distribution of kernel-class regions: is the mean driven by a tail?

compare_regions.py / the heatmap report sum(time)/repeats, i.e. a MEAN. This prints, for the
kernel-class regions of the chosen apps, the quantiles of the individual region samples
(all rows of all run files of one configuration pooled), so a mean can be compared to the
median and to the tail. Values are in microseconds; no profiler is involved.

  scripts/region_quantiles.py --dir v174=out/.../results-1.7.4 --apps tdm,kmeans,dwt,srad,hmm \\
      --size tiny --device mi300a --impl cuda/scale-amd --impl hip/hipcc
"""
import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_regions import classify, parse_name  # noqa: E402


def rows(path):
    """-> (region, time_us, repeat_id) for every data row"""
    hdr = None
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = [q for t in line.split() for q in (["NA"] * (len(t) // 2) if re.fullmatch(r"(?:NA){2,}", t) else [t])]
            if hdr is None:
                if "region" in p and "time" in p:
                    hdr = (p.index("region"), p.index("time"),
                           p.index("repeats_to_two_seconds") if "repeats_to_two_seconds" in p else None)
                continue
            if len(p) <= max(h for h in hdr if h is not None):
                continue
            try:
                yield p[hdr[0]], float(p[hdr[1]]), (p[hdr[2]] if hdr[2] is not None else "0")
            except ValueError:
                pass


def q(xs, f):
    k = (len(xs) - 1) * f
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--apps", required=True, help="comma-separated")
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--device", default=None)
    ap.add_argument("--impl", action="append", help="e.g. cuda/scale-amd (default: all)")
    ap.add_argument("--all-classes", action="store_true", help="not only kernel-class regions")
    a = ap.parse_args()
    apps = set(a.apps.split(","))

    data = defaultdict(list)  # (app, region, label:impl) -> samples
    reps = defaultdict(int)   # same key -> sum over files of distinct repeats_to_two_seconds
    for spec in a.dir:
        label, path = spec.split("=", 1)
        for p in sorted(Path(path).expanduser().glob("lsb.*")):
            m = parse_name(p.name)
            if not m or m["bench"] not in apps or (a.size and m["size"] != a.size) or (a.device and m["device"] != a.device):
                continue
            if a.impl and m["impl"] not in a.impl:
                continue
            seen = defaultdict(set)
            for region, t, rep in rows(p):
                if a.all_classes or classify(region) == "kernel":
                    key = (m["bench"], region, f"{label}:{m['impl']}")
                    data[key].append(t)
                    seen[key].add(rep)
            for key, s in seen.items():
                reps[key] += len(s)

    print(f"{'app':8s} {'region':28s} {'column':22s} {'n':>8s} {'reps':>7s} {'n/rep':>7s} {'mean':>9s} {'p50':>9s} "
          f"{'p99':>9s} {'max':>11s} {'mean/p50':>8s} {'per-rep us':>11s}")
    for key, xs in sorted(data.items()):
        app, region, col = key
        xs.sort()
        mean, med = statistics.mean(xs), q(xs, .5)
        r = max(reps[key], 1)
        print(f"{app:8s} {region:28s} {col:22s} {len(xs):8d} {r:7d} {len(xs) / r:7.2f} {mean:9.1f} {med:9.1f} "
              f"{q(xs, .99):9.1f} {xs[-1]:11.1f} {mean / med if med else float('nan'):8.2f} {sum(xs) / r:11.1f}")
    print("\n(us per sample; reps = distinct repeats_to_two_seconds summed over the run files; n/rep = region executions per "
          "repeat;\n per-rep us = sum(time) / reps, i.e. exactly what compare_regions.py and the heatmap normalise to)")


if __name__ == "__main__":
    main()
