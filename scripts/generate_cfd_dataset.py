#!/usr/bin/env python3
import argparse
from collections import deque
from pathlib import Path

NNB = 4
NDIM = 3

def read_domn(path: Path):
    toks = path.read_text().split()
    if not toks:
        raise SystemExit(f"empty input: {path}")

    it = iter(toks)
    nel = int(next(it))
    cells = []

    for _ in range(nel):
        area = float(next(it))
        neigh = []
        normals = []
        for _j in range(NNB):
            nb = int(next(it))
            normal = [float(next(it)) for _k in range(NDIM)]
            neigh.append(nb)
            normals.append(normal)
        cells.append((area, neigh, normals))

    return cells

def connected_subset(cells, target):
    n = len(cells)
    target = min(target, n)

    seen = {0}
    q = deque([0])
    order = []

    while q and len(order) < target:
        i = q.popleft()
        order.append(i)

        for nb in cells[i][1]:
            # Input format is 1-based for real neighbours, negative for boundary.
            j = nb - 1
            if 0 <= j < n and j not in seen:
                seen.add(j)
                q.append(j)

    # If the graph component was too small, fill deterministically.
    if len(order) < target:
        for i in range(n):
            if i not in seen:
                order.append(i)
                seen.add(i)
                if len(order) == target:
                    break

    return order

def write_subset(cells, selected, out: Path):
    remap = {old: new + 1 for new, old in enumerate(selected)}  # output is 1-based

    with out.open("w") as f:
        f.write(f"{len(selected)}\n")
        for old in selected:
            area, neigh, normals = cells[old]
            fields = [f"{area:.9g}"]

            for nb, normal in zip(neigh, normals):
                j = nb - 1
                out_nb = remap.get(j, -1) if nb > 0 else -1
                fields.append(str(out_nb))
                fields.extend(f"{x:.9g}" for x in normal)

            f.write(" ".join(fields) + "\n")

def write_tiled(cells, target, out: Path):
    base_n = len(cells)
    full_tiles, rem = divmod(target, base_n)

    selected_counts = [base_n] * full_tiles
    if rem:
        selected_counts.append(rem)

    with out.open("w") as f:
        f.write(f"{target}\n")
        offset = 0

        for count in selected_counts:
            for i in range(count):
                area, neigh, normals = cells[i]
                fields = [f"{area:.9g}"]

                for nb, normal in zip(neigh, normals):
                    j = nb - 1
                    # Preserve neighbours only inside this tile/subset.
                    out_nb = offset + j + 1 if nb > 0 and 0 <= j < count else -1
                    fields.append(str(out_nb))
                    fields.extend(f"{x:.9g}" for x in normal)

                f.write(" ".join(fields) + "\n")
            offset += count

def generate(input_path: Path, output_path: Path, target: int):
    cells = read_domn(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if target <= len(cells):
        selected = connected_subset(cells, target)
        write_subset(cells, selected, output_path)
    else:
        write_tiled(cells, target, output_path)

    print(f"wrote {output_path} with nel={target} from source_nel={len(cells)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("target_nel", type=int)
    args = ap.parse_args()

    if args.target_nel <= 0:
        raise SystemExit("target_nel must be positive")

    generate(args.input, args.output, args.target_nel)

if __name__ == "__main__":
    main()
