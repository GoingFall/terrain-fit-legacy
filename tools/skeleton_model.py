#!/usr/bin/env python3
"""Skeleton terrain model: a handful of *placed* landforms instead of stationary noise.

The noise sum could match every summary statistic and still not look like the reference, because it
has no notion of where a mountain is. This model does:

    height(x, z) = base
                 + sum_i  A_i * exp(-((u_i / r_i)^2 + (v_i / (r_i * e_i))^2))
                 - sum_j  D_j * exp(-(dist_to_path_j / w_j)^2)

Each massif is an oriented, elongated gaussian (6 parameters: cx, cz, radius, elongation, angle,
amplitude); each valley is a polyline with depth and width. Massifs are seeded at the reference's own
peaks, then everything is refined by coordinate descent against the reference height field.

Usage:
  python skeleton_model.py --world "<save>" --region r.0.0 --massifs 16 --rounds 6
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from world_layers import collect_layers  # noqa: E402


def peaks(h: np.ndarray, mask: np.ndarray, count: int, min_sep: float) -> list[tuple[float, float, float]]:
    """Greedy peak picking with non-maximum suppression, weighted by prominence."""
    work = np.where(mask, h, -1e9).copy()
    picked = []
    for _ in range(count):
        idx = int(np.argmax(work))
        y, x = divmod(idx, work.shape[1])
        if work[y, x] < -1e8:
            break
        picked.append((float(x), float(y), float(work[y, x])))
        yy, xx = np.ogrid[: work.shape[0], : work.shape[1]]
        work[(xx - x) ** 2 + (yy - y) ** 2 <= min_sep ** 2] = -1e9
    return picked


def field(P: dict, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    h = np.full(X.shape, P["base"], dtype=np.float64)
    for m in P["massifs"]:
        a = math.radians(m["angle"])
        u = (X - m["cx"]) * math.cos(a) + (Z - m["cz"]) * math.sin(a)
        v = -(X - m["cx"]) * math.sin(a) + (Z - m["cz"]) * math.cos(a)
        r = max(m["radius"], 1.0)
        e = max(m["elong"], 0.2)
        h += m["amp"] * np.exp(-((u / r) ** 2 + (v / (r * e)) ** 2))
    for v in P["valleys"]:
        pts = np.asarray(v["path"], dtype=np.float64)
        d = np.full(X.shape, 1e9)
        for (x1, z1), (x2, z2) in zip(pts[:-1], pts[1:]):
            dx, dz = x2 - x1, z2 - z1
            t = np.clip(((X - x1) * dx + (Z - z1) * dz) / max(dx * dx + dz * dz, 1e-6), 0.0, 1.0)
            d = np.minimum(d, np.hypot(X - (x1 + t * dx), Z - (z1 + t * dz)))
        h -= v["depth"] * np.exp(-(d / max(v["width"], 1.0)) ** 2)
    return h


def initial_model(h: np.ndarray, mask: np.ndarray, count: int) -> dict:
    lo = float(np.percentile(h[mask], 2))
    pk = peaks(h, mask, count, min_sep=h.shape[0] / (count ** 0.5) * 1.3)
    massifs = []
    for x, z, y in pk:
        massifs.append({
            "cx": x, "cz": z,
            "radius": h.shape[0] / (count ** 0.5) * 1.1,
            "elong": 1.6, "angle": 35.0,
            "amp": max(y - lo, 5.0),
        })
    return {"base": lo, "massifs": massifs, "valleys": []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--region", default="r.0.0")
    ap.add_argument("--massifs", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--out", default="preview/skeleton.json")
    args = ap.parse_args()

    ground, water, canopy, flags, sid, x0, z0 = collect_layers(
        Path(args.world), args.region, workers=1, fill_missing=False)
    valid = ground > -63
    n = 512
    ref = ground[:n, :n].astype(np.float64)
    mask = valid[:n, :n]
    X, Z = np.meshgrid(np.arange(n, dtype=np.float64), np.arange(n, dtype=np.float64))

    P = initial_model(ref, mask, args.massifs)
    # scale the seed peaks into the same frame as the grid
    for m in P["massifs"]:
        m["angle"] = 0.0

    def loss(model):
        f = field(model, X, Z)
        d = (f - ref)[mask]
        return float(np.sqrt(np.mean(d * d)))

    best = loss(P)
    print(f"start rmse {best:.2f}  ({len(P['massifs'])} massifs)")

    steps = {"cx": 32.0, "cz": 32.0, "radius": 40.0, "elong": 0.4, "angle": 25.0, "amp": 25.0}
    for r in range(args.rounds):
        improved = 0
        for i, m in enumerate(P["massifs"]):
            for key, step in steps.items():
                for sign in (1.0, -1.0):
                    cur = m[key]
                    cand = cur + sign * step
                    if key == "radius":
                        cand = max(20.0, min(400.0, cand))
                    elif key == "elong":
                        cand = max(0.3, min(6.0, cand))
                    elif key == "angle":
                        cand = cand % 180.0
                    elif key == "amp":
                        cand = max(0.0, min(300.0, cand))
                    else:
                        cand = max(-64.0, min(n + 64.0, cand))
                    if cand == cur:
                        continue
                    m[key] = cand
                    l = loss(P)
                    if l < best - 1e-3:
                        best, improved = l, improved + 1
                        break
                    m[key] = cur
        for key in steps:
            steps[key] *= 0.7
        print(f"  round {r}: rmse {best:.2f}  (improved {improved})")

    f = field(P, X, Z)
    corr = float(np.corrcoef(ref[mask], f[mask])[0, 1])
    print(f"\nfinal rmse {best:.2f} blocks  correlation {corr:.3f}")
    print(f"  field range {f.min():.0f}..{f.max():.0f}   reference {ref[mask].min():.0f}..{ref[mask].max():.0f}")
    Path(args.out).write_text(json.dumps(P, indent=1), encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
