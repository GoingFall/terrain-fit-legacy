#!/usr/bin/env python3
"""Python mirror of the Java terrain model in the mod (FitTerrain.java).

The two implementations must produce identical heights, otherwise the preview and the game
disagree — which is exactly what the datapack route suffered from. Check with:

  python tools/fit_model.py --dump out.txt      # then compare with the Java main()
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

SEED = 1337
BASE_LEVEL = 110.0
RELIEF_MACRO = 180.0
CONTINENT_FREQ = 1.9
CONTINENT_GAMMA = 1.7
MASSIF_FREQ = 4.0
MASSIF_AMP = 40.0
MOUNTAIN_FREQ = 2.3
MOUNTAIN_THRESHOLD = 0.20
MOUNTAIN_SOFTNESS = 0.30
RIDGE_FREQ = 8.5
RIDGE_AMP = 55.0
RIDGE_SHARP = 1.5
HILL_FREQ = 11.0
HILL_AMP = 16.0
SPIKE_FREQ = 21.0
SPIKE_AMP = 26.0
SPIKE_THRESHOLD = 0.45
EROSION_FREQ = 3.2
EROSION_FLOOR = 0.30
RIVER_FREQ = 2.6
RIVER_DEPTH = 16.0
RIVER_WIDTH = 0.09
SEA_LEVEL = 74

M32 = 0xFFFFFFFF


def _hash(x: int, z: int, seed: int) -> int:
    h = (x * 374761393 + z * 668265263 + seed * 1442695041) & M32
    h = ((h ^ (h >> 13)) * 1274126177) & M32
    return (h ^ (h >> 16)) & M32


def _rand01(x: int, z: int, seed: int) -> float:
    return (_hash(x, z, seed) & 0x7FFFFFFF) / 0x7FFFFFFF


def _fade(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def noise2(x: float, z: float, freq: float, seed: int, octaves: int) -> float:
    total = 0.0
    amp = 1.0
    norm = 0.0
    f = freq
    for o in range(octaves):
        px, pz = x * f, z * f
        x0, z0 = math.floor(px), math.floor(pz)
        tx, tz = _fade(px - x0), _fade(pz - z0)
        s = seed + o * 7919
        v00 = _rand01(x0, z0, s)
        v10 = _rand01(x0 + 1, z0, s)
        v01 = _rand01(x0, z0 + 1, s)
        v11 = _rand01(x0 + 1, z0 + 1, s)
        a = v00 + (v10 - v00) * tx
        b = v01 + (v11 - v01) * tx
        total += amp * ((a + (b - a) * tz) * 2.0 - 1.0)
        norm += amp
        amp *= 0.5
        f *= 2.0
    return total / norm


def _smooth(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _folded(n: float) -> float:
    return 1.0 - abs(2.0 * abs(n) - 1.0)


def height(x: float, z: float) -> float:
    c = noise2(x, z, CONTINENT_FREQ / 512.0, SEED, 2)
    t = max(0.0, min(1.0, c * 0.5 + 0.5)) ** CONTINENT_GAMMA
    h = BASE_LEVEL + RELIEF_MACRO * t

    m = noise2(x, z, MOUNTAIN_FREQ / 512.0, SEED + 11, 2)
    mountain = _smooth((m - MOUNTAIN_THRESHOLD) / MOUNTAIN_SOFTNESS)

    e = noise2(x, z, EROSION_FREQ / 512.0, SEED + 22, 2) * 0.5 + 0.5
    erosion = EROSION_FLOOR + (1.0 - EROSION_FLOOR) * _smooth(e)

    ridge, amp, f = 0.0, 1.0, RIDGE_FREQ / 512.0
    for _ in range(2):
        ridge += _folded(noise2(x, z, f, SEED + 33, 1)) * amp
        amp *= 0.5
        f *= 2.0
    ridge = (ridge / 1.5) ** RIDGE_SHARP

    massif = noise2(x, z, MASSIF_FREQ / 512.0, SEED + 44, 2)
    h += massif * mountain * MASSIF_AMP * erosion
    h += mountain * ridge * RIDGE_AMP * erosion

    hill = _smooth(noise2(x, z, HILL_FREQ / 512.0, SEED + 55, 2) * 0.5 + 0.5)
    h += (hill * 2.0 - 1.0) * HILL_AMP * erosion

    spike = _smooth((noise2(x, z, SPIKE_FREQ / 512.0, SEED + 66, 2) * 0.5 + 0.5
                     - SPIKE_THRESHOLD) / (1.0 - SPIKE_THRESHOLD))
    h += hill * spike * SPIKE_AMP

    rv = noise2(x, z, RIVER_FREQ / 512.0, SEED + 77, 2)
    river = 1.0 - _smooth(abs(rv) / RIVER_WIDTH)
    h -= river * RIVER_DEPTH
    return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--dump")
    args = ap.parse_args()
    lines = []
    for z in range(args.size):
        for x in range(args.size):
            lines.append(f"{x} {z} {height(x, z):.6f}")
    text = "\n".join(lines)
    if args.dump:
        Path(args.dump).write_text(text, encoding="utf-8")
        print(f"-> {args.dump} ({len(lines)} rows)")
    else:
        print(text[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
