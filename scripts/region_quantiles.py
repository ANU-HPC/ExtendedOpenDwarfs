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
    hdr = None
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = [q for t in line.split() for q in (["NA"] * (len(t) // 2) if re.fullmatch(r"(?:NA){2,}", t) else [t])]
            if hdr is None:
                if "region" in p and "time" in p:
                    hdr = (p.index("region"), p.index("time"))
                continue
            if len(p) <= max(hdr):
                continue
            try:
                yield p[hdr[0]], float(p[hdr[1]])
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
    for spec in a.dir:
        label, path = spec.split("=", 1)
        for p in sorted(Path(path).expanduser().glob("lsb.*")):
            m = parse_name(p.name)
            if not m or m["bench"] not in apps or (a.size and m["size"] != a.size) or (a.device and m["device"] != a.device):
                continue
            if a.impl and m["impl"] not in a.impl:
                continue
            for region, t in rows(p):
                if a.all_classes or classify(region) == "kernel":
                    data[(m["bench"], region, f"{label}:{m['impl']}")].append(t)

    print(f"{'app':8s} {'region':28s} {'column':22s} {'n':>8s} {'mean':>9s} {'p50':>9s} {'p90':>9s} {'p99':>9s} "
          f"{'max':>11s} {'mean/p50':>8s} {'p50 sum':>10s}")
    for (app, region, col), xs in sorted(data.items()):
        xs.sort()
        mean, med = statistics.mean(xs), q(xs, .5)
        print(f"{app:8s} {region:28s} {col:22s} {len(xs):8d} {mean:9.1f} {med:9.1f} {q(xs, .9):9.1f} {q(xs, .99):9.1f} "
              f"{xs[-1]:11.1f} {mean / med if med else float('nan'):8.2f} {med * len(xs) / 1e3:10.0f}")
    print("\n(us per sample; 'p50 sum' = median x count, in ms, i.e. what the region would total with no tail)")


if __name__ == "__main__":
    main()
