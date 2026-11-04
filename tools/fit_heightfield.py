#!/usr/bin/env python3
"""Fit the generator to a reference *height field* instead of to summary statistics.

Metric matching let many different landscapes score the same; this minimises the actual difference
between the generated heights and the reference heights, so the optimiser has to put the mountains
where the reference has them.

Vectorised with numpy, so one evaluation of a 512x512 window is ~0.1 s: thousands of evaluations are
cheap, and no game or server is involved.

Usage:
  python fit_heightfield.py --world "<save>" --region r.0.0 --rounds 8
  python fit_heightfield.py --world "<save>" --region r.0.0 --config _forgeserver/config/terrainfit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from world_layers import collect_layers  # noqa: E402

MASK32 = np.uint32(0xFFFFFFFF)

# parameter bounds: name -> (low, high, is_int)
BOUNDS = {
    "baseLevel": (40.0, 200.0, False),
    "reliefMacro": (40.0, 400.0, False),
    "continentFreq": (0.5, 6.0, False),
    "continentGamma": (0.6, 3.0, False),
    "massifFreq": (1.0, 12.0, False),
    "massifAmp": (0.0, 120.0, False),
    "mountainFreq": (0.5, 8.0, False),
    "mountainThreshold": (-0.6, 0.6, False),
    "mountainSoftness": (0.05, 0.8, False),
    "ridgeFreq": (1.0, 20.0, False),
    "ridgeAmp": (0.0, 120.0, False),
    "ridgeSharp": (0.6, 3.0, False),
    "hillFreq": (1.0, 25.0, False),
    "hillAmp": (0.0, 60.0, False),
    "spikeFreq": (3.0, 40.0, False),
    "spikeAmp": (0.0, 60.0, False),
    "spikeThreshold": (0.0, 0.9, False),
    "erosionFreq": (0.5, 10.0, False),
    "erosionFloor": (0.0, 1.0, False),
    "riverFreq": (0.5, 10.0, False),
    "riverDepth": (0.0, 60.0, False),
    "riverWidth": (0.02, 0.4, False),
    "valleyAngle": (0.0, 180.0, False),
    "valleyDepth": (0.0, 80.0, False),
    "valleyWidth": (0.02, 0.4, False),
    "valleyMeander": (0.0, 0.3, False),
    "valleyMeanderFreq": (0.5, 6.0, False),
}

DEFAULTS = {
    "seed": 1337, "baseLevel": 110.0, "reliefMacro": 180.0, "continentFreq": 1.9,
    "continentGamma": 1.7, "massifFreq": 4.0, "massifAmp": 40.0, "mountainFreq": 2.3,
    "mountainThreshold": 0.20, "mountainSoftness": 0.30, "ridgeFreq": 8.5, "ridgeAmp": 55.0,
    "ridgeSharp": 1.5, "hillFreq": 11.0, "hillAmp": 16.0, "spikeFreq": 21.0, "spikeAmp": 26.0,
    "spikeThreshold": 0.45, "erosionFreq": 3.2, "erosionFloor": 0.30, "riverFreq": 2.6,
    "riverDepth": 16.0, "riverWidth": 0.09, "valleyAngle": 146.0, "valleyDepth": 0.0,
    "valleyWidth": 0.10, "valleyMeander": 0.06, "valleyMeanderFreq": 2.0,
    "snowMaskFreq": 2.0, "snowMaskThreshold": 0.30, "seaLevel": 74, "snowLine": 193.0,
    "snowQuantile": 0.40,
}


# ------------------------------------------------------------------ vectorised model
def _hash(ix, iz, seed):
    h = (ix.astype(np.int64) * 374761393 + iz.astype(np.int64) * 668265263
         + np.int64(seed) * 1442695041)
    h = h.astype(np.uint32)
    h ^= h >> np.uint32(13)
    h *= np.uint32(1274126177)
    h ^= h >> np.uint32(16)
    return h


def _rand01(ix, iz, seed):
    return (_hash(ix, iz, seed) & np.uint32(0x7FFFFFFF)).astype(np.float64) / 0x7FFFFFFF


def _fade(t):
    return t * t * (3.0 - 2.0 * t)


def noise2(X, Z, freq, seed, octaves=2):
    total = np.zeros_like(X, dtype=np.float64)
    amp, norm, f = 1.0, 0.0, freq
    for o in range(octaves):
        px, pz = X * f, Z * f
        x0, z0 = np.floor(px), np.floor(pz)
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


def _smooth(t):
    return np.clip(t, 0.0, 1.0) ** 2 * (3.0 - 2.0 * np.clip(t, 0.0, 1.0))


def field(P, X, Z):
    seed = int(P["seed"])
    c = noise2(X, Z, P["continentFreq"] / 512.0, seed, 2)
    t = np.power(np.clip(c * 0.5 + 0.5, 0.0, 1.0), P["continentGamma"])
    h = P["baseLevel"] + P["reliefMacro"] * t

    m = noise2(X, Z, P["mountainFreq"] / 512.0, seed + 11, 2)
    mountain = _smooth((m - P["mountainThreshold"]) / max(P["mountainSoftness"], 1e-3))

    e = noise2(X, Z, P["erosionFreq"] / 512.0, seed + 22, 2) * 0.5 + 0.5
    erosion = P["erosionFloor"] + (1.0 - P["erosionFloor"]) * _smooth(e)

    ridge = np.zeros_like(h)
    amp, f = 1.0, P["ridgeFreq"] / 512.0
    for _ in range(2):
        n = noise2(X, Z, f, seed + 33, 1)
        ridge += (1.0 - np.abs(2.0 * np.abs(n) - 1.0)) * amp
        amp *= 0.5
        f *= 2.0
    ridge = np.power(np.maximum(ridge / 1.5, 1e-6), P["ridgeSharp"])

    massif = noise2(X, Z, P["massifFreq"] / 512.0, seed + 44, 2)
    h = h + massif * mountain * P["massifAmp"] * erosion
    h = h + mountain * ridge * P["ridgeAmp"] * erosion

    hill = _smooth(noise2(X, Z, P["hillFreq"] / 512.0, seed + 55, 2) * 0.5 + 0.5)
    h = h + (hill * 2.0 - 1.0) * P["hillAmp"] * erosion

    spike = _smooth((noise2(X, Z, P["spikeFreq"] / 512.0, seed + 66, 2) * 0.5 + 0.5
                     - P["spikeThreshold"]) / max(1.0 - P["spikeThreshold"], 1e-3))
    h = h + hill * spike * P["spikeAmp"]

    rv = noise2(X, Z, P["riverFreq"] / 512.0, seed + 77, 2)
    river = 1.0 - _smooth(np.abs(rv) / max(P["riverWidth"], 1e-3))
    h = h - river * P["riverDepth"]

    if P["valleyDepth"] > 0:
        a = np.radians(P["valleyAngle"])
        u = X * np.cos(a) + Z * np.sin(a)
        v = -X * np.sin(a) + Z * np.cos(a)
        meander = P["valleyMeander"] * noise2(u, np.zeros_like(u), P["valleyMeanderFreq"] / 512.0,
                                              seed + 88, 2)
        band = 1.0 - _smooth(np.abs(v - meander) / max(P["valleyWidth"], 1e-3))
        h = h - band * P["valleyDepth"]
    return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--region", default="r.0.0")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--samples", type=int, default=512, help="window size in blocks")
    ap.add_argument("--config", help="read/write parameters here")
    ap.add_argument("--out", default="preview/fit_heightfield.json")
    args = ap.parse_args()

    ground, water, canopy, flags, sid, x0, z0 = collect_layers(
        Path(args.world), args.region, workers=1, fill_missing=False)
    valid = ground > -63
    size = min(args.samples, ground.shape[0], ground.shape[1])
    ref = ground[:size, :size].astype(np.float64)
    mask = valid[:size, :size]
    print(f"reference window {size}x{size}, {mask.sum()} valid columns")

    gx, gz = np.meshgrid(np.arange(size, dtype=np.float64) + x0,
                         np.arange(size, dtype=np.float64) + z0)

    P = dict(DEFAULTS)
    if args.config and Path(args.config).exists():
        P.update(json.loads(Path(args.config).read_text(encoding="utf-8")))

    def loss(params):
        f = field(params, gx, gz)
        d = (f - ref)[mask]
        return float(np.sqrt(np.mean(d * d)))

    best = loss(P)
    print(f"start rmse {best:.2f} blocks")

    steps = {k: (hi - lo) * 0.08 for k, (lo, hi, _) in BOUNDS.items()}
    for r in range(args.rounds):
        improved = 0
        for k, (lo, hi, is_int) in BOUNDS.items():
            cur = P[k]
            for sign in (1.0, -1.0):
                cand = cur + sign * steps[k]
                cand = max(lo, min(hi, cand))
                if is_int:
                    cand = round(cand)
                if cand == cur:
                    continue
                P[k] = cand
                l = loss(P)
                if l < best - 1e-4:
                    best, improved = l, improved + 1
                    break
                P[k] = cur
            steps[k] *= 0.75
        print(f"  round {r}: rmse {best:.2f}  (improved {improved} params)")
        if improved == 0 and r > 2:
            break

    print(f"\nfinal rmse {best:.2f} blocks")
    for k in ("baseLevel", "reliefMacro", "continentFreq", "ridgeFreq", "ridgeAmp", "massifAmp",
              "valleyDepth", "valleyAngle"):
        print(f"  {k:18} {P[k]:.3f}")

    Path(args.out).write_text(json.dumps(P, indent=2), encoding="utf-8")
    if args.config:
        Path(args.config).write_text(json.dumps(P, indent=2), encoding="utf-8")
        print(f"-> {args.config}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
