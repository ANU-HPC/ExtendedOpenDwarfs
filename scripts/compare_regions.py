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
  total   sum of ALL regions, including runtime_initialization / runtime_finalization.
          Device start-up is real cost for every runtime, but it lands in different
          regions (SCALE defers it to the first allocation or stream creation; HIP pays
          it in the init region), so excluding any region would charge one side only.
  total_ex_init  (with --show-total-ex-init) as total but without the once-only
          init/finalisation regions; a diagnostic only, not comparable across runtimes.
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
    warn_cols = []
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            # liblsb prints an unset parameter as "NA" with no separator, so adjacent unset
            # parameters appear fused ("NANA"). Split them so columns line up with the header.
            parts = [p2 for p in parts
                     for p2 in (["NA"] * (len(p) // 2) if re.fullmatch(r"(?:NA){2,}", p) else [p])]
            if header is not None and len(parts) != len(header):
                warn_cols.append(path)
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
    if warn_cols:
        print(f"warning: {Path(path).name}: {len(warn_cols)} row(s) with a different field count "
              f"than the header; times in those rows may be wrong", file=sys.stderr)
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
    ap.add_argument("--show-total-ex-init", action="store_true",
                    help="also print total_ex_init (total without runtime_initialization/finalization); diagnostic only")
    ap.add_argument("--regress", metavar="NEW,OLD",
                    help="candidate pair to judge, e.g. v174:cuda/scale-amd,v173:cuda/scale-amd; prints a "
                         "REGRESSION/SPEEDUP table of rows outside the noise floor (needs --noise)")
    ap.add_argument("--noise", metavar="NEW,OLD",
                    help="same-code reference pair giving the noise floor, e.g. v174:hip/hipcc,v173:hip/hipcc "
                         "(identical source and toolchain in both, so any difference is noise or machine drift)")
    ap.add_argument("--noise-mult", type=float, default=2.0,
                    help="floor = max(--floor-min, noise-mult x |reference ratio - 1|) per row (default 2)")
    ap.add_argument("--floor-min", type=float, default=0.10,
                    help="smallest allowed floor, as a fraction (default 0.10; five-iteration medians are not "
                         "stable below about +-7%%)")
    ap.add_argument("--verdict-min-us", type=float, default=20.0,
                    help="ignore rows whose largest value is below this many us (default 20)")
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

    pairs = {}
    for flag, spec in (("--regress", a.regress), ("--noise", a.noise)):
        if spec:
            if spec.count(",") != 1:
                sys.exit(f"{flag} needs NEW,OLD, got {spec!r}")
            pairs[flag] = tuple(spec.split(","))
            for c in pairs[flag]:
                if c not in cols:
                    sys.exit(f"{flag} column {c!r} not found; available: {', '.join(cols)}")
    if a.regress and not a.noise:
        sys.exit("--regress needs --noise (a same-code reference pair, e.g. HIP in both versions)")

    summary, longrows, verdict_src = [], [], []
    for bench in sorted(data):
        if a.app and bench not in a.app:
            continue
        regs = data[bench]
        body = []
        for region in sorted(regs, key=lambda r: (classify(r), r)):
            vals = regs[region]
            verdict_src.append((bench, region, vals))
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

        metrics = [("kernel", lambda r: classify(r) == "kernel"),
                   ("total", lambda r: True)]
        if a.show_total_ex_init:
            metrics.append(("total_ex_init", lambda r: r not in ONCE_ONLY))
        for metric, pred in metrics:
            vals = {c: agg(c, pred) for c in cols}
            summary.append([bench, metric] + [fmt_us(vals[c]) for c in cols]
                           + [fmt_ratio(vals.get(n), vals.get(d), a.tol) for n, d in ratios])
            if metric in ("kernel", "total"):
                verdict_src.append((bench, f"[{metric}]", {c: v for c, v in vals.items() if v is not None}))

    print("## Summary (sum of per-repeat region means, us)\n")
    print(table(summary, ["app", "metric"] + cols + rhead))

    if a.regress:
        cn, co = pairs["--regress"]
        rn, ro = pairs["--noise"]
        rows, examined = [], 0
        for bench, name, vals in verdict_src:
            if any(vals.get(c) in (None, 0) for c in (cn, co, rn, ro)):
                continue
            if max(vals[c] for c in (cn, co, rn, ro)) < a.verdict_min_us:
                continue
            examined += 1
            r = vals[cn] / vals[co]
            noise = vals[rn] / vals[ro]
            floor = max(a.floor_min, a.noise_mult * abs(noise - 1.0))
            if r > 1 + floor:
                verdict = "REGRESSION"
            elif r < 1 - floor:
                verdict = "SPEEDUP"
            else:
                continue
            rows.append((verdict, abs(vals[cn] - vals[co]), bench, name, vals[co], vals[cn], r, noise, floor))
        print(f"\n## Verdict: {cn} vs {co}\n")
        print(f"Floor per row = max({a.floor_min:.0%}, {a.noise_mult:g} x |reference ratio - 1|) where the reference is "
              f"{rn} / {ro} (same code in both, so any difference there is noise or machine drift). "
              f"{examined} rows examined (>= {a.verdict_min_us:g} us), {len(rows)} outside the floor. "
              f"Rows are per-repeat means from medians over the run files; a region recorded once per run "
              f"(one-off) is a single sample, so treat those with care.\n")
        rows.sort(key=lambda t: (t[0] != "REGRESSION", -t[1]))
        print(table([[v, b, n, fmt_us(old), fmt_us(new), f"{r:.2f}x", f"{nz:.2f}x", f"+-{fl:.0%}", f"{new - old:+,.1f}"]
                     for v, d, b, n, old, new, r, nz, fl in rows],
                    ["verdict", "app", "region", "old us", "new us", "ratio", "noise ratio", "floor", "delta us"]))

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["app", "region", "class"] + cols)
            w.writerows(longrows)
        print(f"\nwrote {a.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
