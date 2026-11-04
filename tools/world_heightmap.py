#!/usr/bin/env python3
"""Turn a world's region files into a height map PNG + quantitative stats.

Why: screenshots are a bad feedback channel for terrain work (shader, FOV and camera angle
all lie). The height field is the ground truth, and it is fully reconstructable from the
region files -- no client needed.

Usage:
  python world_heightmap.py <world-dir> [-o out.png] [--region r.0.0] [--profile x0 z0 x1 z1]
  python world_heightmap.py <world-dir> --stats
"""
from __future__ import annotations

import argparse
import gzip
import math
import struct
import zlib
from pathlib import Path

from PIL import Image


# ---------------------------------------------------------------- minimal NBT reader
class Reader:
    def __init__(self, data: bytes) -> None:
        self.d = data
        self.i = 0

    def take(self, n: int) -> bytes:
        v = self.d[self.i : self.i + n]
        self.i += n
        return v

    def u1(self) -> int:
        return self.take(1)[0]

    def i1(self) -> int:
        return struct.unpack(">b", self.take(1))[0]

    def u2(self) -> int:
        return struct.unpack(">H", self.take(2))[0]

    def i2(self) -> int:
        return struct.unpack(">h", self.take(2))[0]

    def i4(self) -> int:
        return struct.unpack(">i", self.take(4))[0]

    def i8(self) -> int:
        return struct.unpack(">q", self.take(8))[0]

    def f4(self) -> float:
        return struct.unpack(">f", self.take(4))[0]

    def f8(self) -> float:
        return struct.unpack(">d", self.take(8))[0]

    def string(self) -> str:
        return self.take(self.u2()).decode("utf-8", "replace")

    def payload(self, t: int):
        if t == 1:
            return self.i1()
        if t == 2:
            return self.i2()
        if t == 3:
            return self.i4()
        if t == 4:
            return self.i8()
        if t == 5:
            return self.f4()
        if t == 6:
            return self.f8()
        if t == 7:
            return self.take(self.i4())
        if t == 8:
            return self.string()
        if t == 9:
            et = self.u1()
            n = self.i4()
            return [self.payload(et) for _ in range(n)]
        if t == 10:
            out = {}
            while True:
                tt = self.u1()
                if tt == 0:
                    return out
                name = self.string()
                out[name] = self.payload(tt)
        if t == 11:
            n = self.i4()
            return [self.i4() for _ in range(n)]
        if t == 12:
            n = self.i4()
            return [self.i8() for _ in range(n)]
        raise ValueError(f"bad tag {t}")


def read_nbt(raw: bytes) -> dict:
    r = Reader(raw)
    t = r.u1()
    if t != 10:
        raise ValueError("root is not a compound")
    r.string()
    return r.payload(10)


# ---------------------------------------------------------------- chunk decoding
def iter_chunks(mca: Path):
    data = mca.read_bytes()
    if len(data) < 8192:            # truncated / header-only file (server killed mid-write)
        return
    for i in range(1024):
        off = int.from_bytes(data[i * 4 : i * 4 + 3], "big")
        cnt = data[i * 4 + 3]
        if not off or not cnt:
            continue
        start = off * 4096
        if start + 5 > len(data):
            continue
        length = int.from_bytes(data[start : start + 4], "big")
        ctype = data[start + 4]
        blob = data[start + 5 : start + 4 + length]
        try:
            if ctype == 1:
                raw = gzip.decompress(blob)
            elif ctype == 2:
                raw = zlib.decompress(blob)
            elif ctype == 3:
                raw = blob
            else:
                continue
            yield i, raw
        except Exception:
            continue


def unpack_indices(data, n_entries: int, palette_size: int):
    bits = max(4, (palette_size - 1).bit_length())
    per_long = 64 // bits
    mask = (1 << bits) - 1
    out = []
    for word in data:
        w = word & 0xFFFFFFFFFFFFFFFF
        for k in range(per_long):
            if len(out) >= n_entries:
                return out
            out.append((w >> (k * bits)) & mask)
    return out


def chunk_heights(root: dict):
    """Return (tops, names): 16x16 top block y and block name per (z, x)."""
    tops = [[-65] * 16 for _ in range(16)]
    names = [["minecraft:air"] * 16 for _ in range(16)]
    for sec in root.get("sections") or []:
        if "block_states" not in sec:
            continue
        y = sec.get("Y", 0)
        bs = sec["block_states"]
        palette = [p.get("Name", "minecraft:air") for p in bs.get("palette") or []]
        if not palette or all(p == "minecraft:air" for p in palette):
            continue
        data = bs.get("data")
        idx = unpack_indices(data, 4096, len(palette)) if data else [0] * 4096
        for i, pi in enumerate(idx):
            if pi >= len(palette) or palette[pi] == "minecraft:air":
                continue
            yy = y * 16 + (i >> 8)
            z = (i >> 4) & 15
            x = i & 15
            if yy > tops[z][x]:
                tops[z][x] = yy
                names[z][x] = palette[pi]
    return tops, names


def collect(world: Path, only_region: str | None):
    cells: dict[tuple[int, int], int] = {}
    for mca in sorted(world.glob("region/*.mca")):
        if only_region and mca.stem != only_region:
            continue
        try:
            rx, rz = mca.stem.split(".")[1:3]
            rx, rz = int(rx), int(rz)
        except Exception:
            continue
        for slot, raw in iter_chunks(mca):
            try:
                root = read_nbt(raw)
            except Exception:
                continue
            if root.get("Status") not in (None, "minecraft:full") and root.get("Status") != "full":
                continue
            tops, _names = chunk_heights(root)
            cx = slot & 31
            cz = slot >> 5
            bx0 = (rx * 32 + cx) * 16
            bz0 = (rz * 32 + cz) * 16
            for z in range(16):
                for x in range(16):
                    cells[(bx0 + x, bz0 + z)] = tops[z][x]
    return cells


def collect_with_water(world: Path, only_region: str | None = None):
    """Like collect(), but also returns the set of columns whose top block is water."""
    cells: dict[tuple[int, int], int] = {}
    wet: set[tuple[int, int]] = set()
    for mca in sorted(world.glob("region/*.mca")):
        if only_region and mca.stem != only_region:
            continue
        try:
            rx, rz = mca.stem.split(".")[1:3]
            rx, rz = int(rx), int(rz)
        except Exception:
            continue
        for slot, raw in iter_chunks(mca):
            try:
                root = read_nbt(raw)
            except Exception:
                continue
            tops, names = chunk_heights(root)
            bx0 = (rx * 32 + (slot & 31)) * 16
            bz0 = (rz * 32 + (slot >> 5)) * 16
            for z in range(16):
                for x in range(16):
                    k = (bx0 + x, bz0 + z)
                    cells[k] = tops[z][x]
                    if "water" in names[z][x]:
                        wet.add(k)
    return cells, wet


def render(cells, path: Path):
    xs = [k[0] for k in cells]
    zs = [k[1] for k in cells]
    if not xs:
        raise SystemExit("no chunks decoded")
    x0, x1 = min(xs), max(xs) + 1
    z0, z1 = min(zs), max(zs) + 1
    w, h = x1 - x0, z1 - z0
    img = Image.new("RGB", (w, h), (0, 0, 0))
    px = img.load()

    def color(y: int):
        # -64..320 -> deep blue / green / tan / snow
        t = max(0.0, min(1.0, (y + 64) / 384.0))
        if t < 0.25:
            f = t / 0.25
            return (int(20 + 20 * f), int(60 + 60 * f), int(120 + 40 * f))
        if t < 0.5:
            f = (t - 0.25) / 0.25
            return (int(40 + 40 * f), int(120 + 50 * f), int(60 - 20 * f))
        if t < 0.75:
            f = (t - 0.5) / 0.25
            return (int(80 + 90 * f), int(170 - 60 * f), int(40 - 20 * f))
        f = (t - 0.75) / 0.25
        return (int(170 + 70 * f), int(110 + 120 * f), int(20 + 200 * f))

    for (bx, bz), y in cells.items():
        r, g, b = color(y)
        # cheap hillshade from the west neighbour
        nb = cells.get((bx - 1, bz - 1), y)
        shade = max(-30, min(30, (y - nb) * 3))
        px[bx - x0, bz - z0] = (
            max(0, min(255, r + shade)),
            max(0, min(255, g + shade)),
            max(0, min(255, b + shade)),
        )
    img.save(path)
    print(f"heightmap -> {path}  ({w}x{h} blocks, {len(cells)} columns)")
    return cells


def stats(cells):
    ys = list(cells.values())
    ys_sorted = sorted(ys)
    n = len(ys_sorted)
    print(f"columns={n}  min={ys_sorted[0]}  max={ys_sorted[-1]}  "
          f"median={ys_sorted[n // 2]}  mean={sum(ys) / n:.1f}")
    print(f"above y=200: {sum(1 for y in ys if y > 200) / n:.1%}   "
          f"below y=64: {sum(1 for y in ys if y < 64) / n:.1%}")
    # slope: |dy| between horizontal neighbours
    slopes = []
    for (bx, bz), y in cells.items():
        for dx, dz in ((1, 0), (0, 1)):
            nb = cells.get((bx + dx, bz + dz))
            if nb is not None:
                slopes.append(abs(y - nb))
    if slopes:
        slopes.sort()
        m = len(slopes)
        print(f"slope per block: median={slopes[m // 2]}  p90={slopes[int(m * 0.9)]}  "
              f"max={slopes[-1]}  (>2 = cliffy)")


def profile(cells, x0: int, z0: int, x1: int, z1: int):
    steps = max(abs(x1 - x0), abs(z1 - z0), 1)
    print(f"\nprofile ({x0},{z0}) -> ({x1},{z1}):")
    line = []
    for i in range(steps + 1):
        x = round(x0 + (x1 - x0) * i / steps)
        z = round(z0 + (z1 - z0) * i / steps)
        y = cells.get((x, z))
        line.append((i, x, z, y))
    for i, x, z, y in line[:: max(1, len(line) // 40)]:
        bar = "#" * max(0, int(((y or 0) + 64) / 8))
        print(f"  {i:>5}  x={x:>6} z={z:>6}  y={y if y is not None else '?':>5}  {bar}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("-o", "--out", default="heightmap.png")
    ap.add_argument("--region")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--profile", nargs=4, type=int, metavar=("X0", "Z0", "X1", "Z1"))
    args = ap.parse_args()

    cells = collect(Path(args.world), args.region)
    render(cells, Path(args.out))
    stats(cells)
    if args.profile:
        profile(cells, *args.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
