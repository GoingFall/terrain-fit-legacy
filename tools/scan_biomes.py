#!/usr/bin/env python3
"""Which biomes does a save actually use, and where?

1.20.1 stores biomes per section as a 4x4x4 palette, so a column's biome is found by looking at the
section that contains the surface. This prints the distribution overall and per region, which is
what decides the fixed biome for a test dimension.

Usage:
  python scan_biomes.py "<world>" --region r.0.0
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from world_heightmap import iter_chunks, read_nbt  # noqa: E402


def bits_for(n: int) -> int:
    return max(1, (n - 1).bit_length())


def to_u64(vals) -> np.ndarray:
    """NBT longs are signed; cast through int64 -> uint64 without overflowing."""
    return np.asarray(vals, dtype=np.int64).astype(np.uint64)


def unpack(longs: np.ndarray, bits: int, count: int) -> np.ndarray:
    per_long = 64 // bits
    mask = (1 << bits) - 1
    out = np.empty((len(longs), per_long), dtype=np.int64)
    shift = np.arange(per_long) * bits
    for i, v in enumerate(longs):
        u = np.uint64(v)
        out[i] = (u >> shift.astype(np.uint64)) & np.uint64(mask)
    return out.ravel()[:count]


def section_biomes(section: dict) -> np.ndarray | None:
    b = section.get("biomes")
    if not b or "palette" not in b:
        return None
    pal = [str(p) for p in b["palette"]]
    if "data" not in b:
        return np.zeros(64, dtype=np.int64)
    idx = unpack(to_u64(b["data"]), bits_for(len(pal)), 64)
    idx = np.clip(idx, 0, len(pal) - 1)
    return np.array([pal[i] for i in idx], dtype=object)


def column_biomes(root: dict) -> np.ndarray | None:
    """16x16 biome names for the chunk, taken at the surface of each column."""
    sections = root.get("sections") or []
    heights = np.full((16, 16), -999, dtype=np.int32)
    for sec in sections:
        bs = sec.get("block_states")
        if not bs or "palette" not in bs:
            continue
        pal = bs["palette"]
        y0 = int(sec.get("Y", 0))
        if "data" not in bs:
            ids = np.zeros((16, 16, 16), dtype=np.int64)
        else:
            n = max(4, bits_for(len(pal)))
            raw = unpack(to_u64(bs["data"]), n, 4096)
            ids = raw.reshape(16, 16, 16)
        names = np.array([str(p.get("Name", "")) for p in pal], dtype=object)
        solid = (ids != 0)
        # index of the top solid block per column, in this section
        for yy in range(15, -1, -1):
            layer = solid[:, :, yy]
            names_layer = names[ids[:, :, yy]]
            not_air = layer & (names_layer != "minecraft:air")
            take = not_air & (heights < y0 * 16)
            if take.any():
                heights[take] = y0 * 16 + yy
    # second pass: read the biome palette of the section containing each column's surface
    out = np.array([["" for _ in range(16)] for _ in range(16)], dtype=object)
    for sec in sections:
        y0 = int(sec.get("Y", 0))
        top = y0 * 16 + 15
        bot = y0 * 16
        cols = (heights >= bot) & (heights <= top)
        if not cols.any():
            continue
        biomes = section_biomes(sec)
        if biomes is None:
            continue
        biomes = biomes.reshape(4, 4, 4)
        for y in range(16):
            m = cols & (heights == bot + y)
            if not m.any():
                continue
            plate = np.repeat(np.repeat(biomes[:, :, y // 4], 4, axis=0), 4, axis=1)
            out[m] = plate[m]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("--region")
    args = ap.parse_args()

    counter = collections.Counter()
    total = 0
    region_dir = Path(args.world) / "region"
    files = sorted(region_dir.glob("*.mca"))
    if args.region:
        files = [f for f in files if f.stem == args.region]
    print(f"scanning {len(files)} region file(s)")
    for mca in files:
        for _idx, raw in iter_chunks(mca):
            try:
                root = read_nbt(raw)
            except Exception:
                continue
            names = column_biomes(root)
            if names is None:
                continue
            flat = [n for n in names.ravel() if n]
            counter.update(flat)
            total += len(flat)

    if not counter:
        print("no biome data found")
        return 1
    print(f"{total} columns\n")
    for name, n in counter.most_common(15):
        print(f"  {name:46} {n:7d}  {n / total * 100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
