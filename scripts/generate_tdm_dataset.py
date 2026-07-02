#!/usr/bin/env python3
import argparse
from pathlib import Path


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def copy_prefix(src: Path, dst: Path, n: int) -> None:
    if n <= 0:
        raise SystemExit(f"prefix length must be positive for {dst}")

    if not src.exists():
        raise SystemExit(f"missing input file: {src}")

    total = count_lines(src)
    if n > total:
        raise SystemExit(f"{src} only has {total} lines, requested {n}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    copied = 0
    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if copied >= n:
                break
            f_out.write(line)
            copied += 1

    print(f"Wrote {copied} lines to {dst}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate TDM workload-size datasets by taking leading entries from existing event and episode files."
    )
    p.add_argument("input_events", type=Path)
    p.add_argument("input_episodes", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("test/finite-state-machine/tdm"))
    p.add_argument("--prefix", default="sim-64-size200")
    p.add_argument("--episodes-prefix", default="episodes")
    p.add_argument("--tiny-events", type=int, default=512)
    p.add_argument("--tiny-episodes", type=int, default=4)
    p.add_argument("--small-events", type=int, default=4096)
    p.add_argument("--small-episodes", type=int, default=8)
    p.add_argument("--medium-events", type=int, default=65536)
    p.add_argument("--medium-episodes", type=int, default=16)
    p.add_argument("--large-events", type=int, default=200000)
    p.add_argument("--large-episodes", type=int, default=30)
    args = p.parse_args()

    sizes = {
        "tiny": (args.tiny_events, args.tiny_episodes),
        "small": (args.small_events, args.small_episodes),
        "medium": (args.medium_events, args.medium_episodes),
        "large": (args.large_events, args.large_episodes),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for size, (events, episodes) in sizes.items():
        copy_prefix(
            args.input_events,
            args.output_dir / f"{args.prefix}-{size}.csv",
            events,
        )
        copy_prefix(
            args.input_episodes,
            args.output_dir / f"{args.episodes_prefix}-{size}.txt",
            episodes,
        )


if __name__ == "__main__":
    main()
