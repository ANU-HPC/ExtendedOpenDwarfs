#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path


def read_loc(path: Path):
    raw = path.read_bytes()
    if len(raw) < 4:
        raise SystemExit(f"Invalid SWAT .loc file: {path}")
    n = struct.unpack_from("i", raw, 0)[0]
    lengths = list(struct.unpack_from(f"{n}i", raw, 4))
    return lengths


def write_subset(src_prefix: Path, dst_prefix: Path, count: int) -> None:
    src_data = src_prefix.with_suffix(".data")
    src_loc = src_prefix.with_suffix(".loc")
    dst_data = dst_prefix.with_suffix(".data")
    dst_loc = dst_prefix.with_suffix(".loc")

    lengths = read_loc(src_loc)
    data = src_data.read_bytes()

    if count <= 0:
        raise SystemExit("count must be positive")

    dst_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_lengths = []
    out_data = bytearray()
    offset = 0
    seqs = []

    for length in lengths:
        seqs.append(data[offset:offset + length])
        offset += length

    for i in range(count):
        seq = seqs[i % len(seqs)]
        out_lengths.append(len(seq))
        out_data.extend(seq)

    dst_loc.write_bytes(struct.pack("i", count) + struct.pack(f"{count}i", *out_lengths))
    dst_data.write_bytes(out_data)

    print(f"Wrote {count} SWAT sequences to {dst_prefix}.loc/.data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_prefix", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--tiny", type=int, default=32)
    ap.add_argument("--small", type=int, default=8192)
    ap.add_argument("--medium", type=int, default=256)
    ap.add_argument("--large", type=int, default=32768)
    args = ap.parse_args()

    for name, count in [
        ("tiny", args.tiny),
        ("small", args.small),
        ("medium", args.medium),
        ("large", args.large),
    ]:
        write_subset(args.source_prefix, args.output_dir / f"sampledb-{name}", count)


if __name__ == "__main__":
    main()
