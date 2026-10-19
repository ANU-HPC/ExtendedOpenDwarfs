#!/usr/bin/env python3
"""Per-region comparison of LSB results across result directories.

Meant for examining slowdowns: SCALE version A vs B, SCALE vs HIP, OpenCL as a
reference. Not a replacement for plot_heatmap.R -- same normalisation, but it
prints per-region numbers so a ratio can be traced to the region causing it.

Normalisation (mirrors scripts/lsb_common.R, parse_lsb_table_dt):
  per file and region:  sum(time_us) / number of distinct repeats_to_two_seconds
  across run files (.r0, .r1, ...):  median
Region classes use the same regexes as classify_region() in lsb_common.R.

  scripts/compare_regions.py \\
      --dir v170=~/eod-tiny/results-1.7.0 \\
      --dir v174=~/eod-tiny/results-1.7.4 \\
      --dir ocl=~/eod-tiny/results-opencl \\
      --size tiny --device mi300a \\
      --ratio v174:cuda/scale-amd,v170:cuda/scale-amd \\
      --ratio v170:cuda/scale-amd,v170:hip/hipcc \\
      --ratio v174:cuda/scale-amd,v174:hip/hipcc \\
      --ratio v174:hip/hipcc,v170:hip/hipcc          # noise floor (same code)

Columns are "<label>:<backend>/<compiler>". A ratio is NUM/DEN, so >1 means NUM
is slower. Ratios beyond --tol are marked SLOW / fast.

Summary metrics per app:
  kernel  sum of regions classified "kernel" (includes setting_*_kernel_arguments,
          as in the heatmaps)
  total   sum of all regions except runtime_initialization / runtime_finalization
          (once-only costs outside the repeat loop)
"""
import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

FNAME = re.compile(r"^lsb\.(.+)\.r(\d+)(?:-(\d+))?$")
ONCE_ONLY = {"runtime_initialization", "runtime_finalization"}


def parse_name(base):
    """lsb.<bench>_<backend>_<compiler>[_<vendor>]_<size>_<device>.r<run>"""
    m = FNAME.match(base)
    if not m:
        return None
    toks = m.group(1).split("_")
    if len(toks) < 4:
        return None
    bench, backend, i = toks[0], toks[1], 2
    if toks[i].lower() == "scale":
        if i + 1 >= len(toks):
            return None
        compiler = f"scale-{toks[i + 1]}"
        i += 2
    else:
        compiler = toks[i]
        i += 1
    if i + 1 >= len(toks):
        return None
    size = toks[i].lower()
    device = "_".join(toks[i + 1:])
    return {"bench": bench, "impl": f"{backend}/{compiler}", "size": size, "device": device}


def classify(region):
    r = region.lower()
    if r == "runtime_initialization":
        return "init"
    if r == "host_input_load":
        return "input"
    if re.search(r"kernel_creation|program_creation|build", r):
        return "kernel_creation"
    if re.search(r"kernel|compute|solve|execute", r):
        return "kernel"
    if re.search(r"h2d|d2h|copy|transfer|memcpy|read|write", r):
        return "transfer"
    if re.search(r"setup|argument|arg|alloc|buffer|init", r):
        return "setup"
    return "other"


def read_regions(path):
    """-> {region: per-repeat mean in us}"""
    header = None
    agg = {}
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
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
            total, reps = agg.setdefault(parts[ri], [0.0, set()])
            agg[parts[ri]][0] = total + t
            reps.add(rep)
    return {r: tot / max(len(reps), 1) for r, (tot, reps) in agg.items()}


def collect(dirs, size, device):
    """-> data[bench][region][col] = median us;  files[col] = count"""
    samples = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    files = defaultdict(int)
    for label, d in dirs:
        for p in sorted(Path(d).expanduser().glob("lsb.*")):
            meta = parse_name(p.name)
            if not meta:
                continue
            if size and meta["size"] != size:
                continue
            if device and meta["device"] != device:
                continue
            col = f"{label}:{meta['impl']}"
            files[col] += 1
            for region, v in read_regions(p).items():
                samples[meta["bench"]][region][col].append(v)
    data = {
        b: {r: {c: statistics.median(v) for c, v in cols.items()} for r, cols in regs.items()}
        for b, regs in samples.items()
    }
    return data, files


def fmt_us(v):
    return "-" if v is None else f"{v:,.1f}"


def fmt_ratio(num, den, tol):
    if num is None or den is None or den <= 0:
        return "-"
    r = num / den
    tag = " SLOW" if r > 1 + tol else (" fast" if r < 1 - tol else "")
    return f"{r:.2f}x{tag}"


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--device", default=None)
    ap.add_argument("--app", action="append", help="restrict to these apps (repeatable)")
    ap.add_argument("--ratio", action="append", default=[], metavar="NUM,DEN")
    ap.add_argument("--tol", type=float, default=0.10, help="mark ratios outside 1+-tol (default 0.10)")
    ap.add_argument("--min-us", type=float, default=0.0,
                    help="hide regions whose largest value is below this many us (noisy ratios)")
    ap.add_argument("--csv", help="also write a long-format CSV here")
    a = ap.parse_args()

    dirs = []
    for spec in a.dir:
        if "=" not in spec:
            sys.exit(f"--dir needs LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        if not Path(path).expanduser().is_dir():
            sys.exit(f"not a directory: {path}")
        dirs.append((label, path))
    ratios = []
    for spec in a.ratio:
        if spec.count(",") != 1:
            sys.exit(f"--ratio needs NUM,DEN, got {spec!r}")
        ratios.append(tuple(spec.split(",")))

    data, files = collect(dirs, a.size, a.device)
    if not data:
        sys.exit("no matching lsb.* files (check --size/--device and the directories)")
    cols = sorted({c for b in data.values() for r in b.values() for c in r})
    for n, d in ratios:
        for c in (n, d):
            if c not in cols:
                sys.exit(f"ratio column {c!r} not found; available: {', '.join(cols)}")

    print(f"size={a.size} device={a.device or 'any'}  (median across runs; us per repeat)\n")
    print("Columns (files read):", ", ".join(f"{c} [{files[c]}]" for c in cols), "\n")
    rhead = [f"{n} / {d}" for n, d in ratios]

    summary, longrows = [], []
    for bench in sorted(data):
        if a.app and bench not in a.app:
            continue
        regs = data[bench]
        body = []
        for region in sorted(regs, key=lambda r: (classify(r), r)):
            vals = regs[region]
            if max(vals.values()) < a.min_us:
                continue
            row = [region, classify(region)] + [fmt_us(vals.get(c)) for c in cols]
            row += [fmt_ratio(vals.get(n), vals.get(d), a.tol) for n, d in ratios]
            body.append(row)
            longrows.append([bench, region, classify(region)] + [vals.get(c, "") for c in cols])
        print(f"### {bench}\n")
        print(table(body, ["region", "class"] + cols + rhead))
        print()

        def agg(c, pred):
            vs = [regs[r][c] for r in regs if c in regs[r] and pred(r)]
            return sum(vs) if vs else None

        for metric, pred in (("kernel", lambda r: classify(r) == "kernel"),
                             ("total", lambda r: r not in ONCE_ONLY)):
            vals = {c: agg(c, pred) for c in cols}
            summary.append([bench, metric] + [fmt_us(vals[c]) for c in cols]
                           + [fmt_ratio(vals.get(n), vals.get(d), a.tol) for n, d in ratios])

    print("## Summary (sum of per-repeat region means, us)\n")
    print(table(summary, ["app", "metric"] + cols + rhead))

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["app", "region", "class"] + cols)
            w.writerows(longrows)
        print(f"\nwrote {a.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
