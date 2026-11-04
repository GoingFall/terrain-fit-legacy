#!/usr/bin/env python3
"""Evaluate an exported density function offline: solve for the surface height per column.

Why this works: the surface is where the density crosses zero. Every wrapper the vanilla router
puts around the height term (squeeze, min with the cave noise) is monotonic and zero-preserving,
so the crossing can be found from the height term alone -- and that means the export can be checked
against a target *without launching the game*.

Supported nodes (the subset we emit): noise, add, mul, abs, clamp, spline,
y_clamped_gradient, interpolated / blend_density / flat_cache / cache_* (pass-through), squeeze,
min / max (zero-crossing still needs the other branch to be evaluated at the same y).

Usage:
  python df_eval.py --datapack dist/fit-datapack --namespace fit --name fit --size 512 \
                    --target preview/m_r00.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from terrain_metrics import compare, compute, summary_line  # noqa: E402
from terrain_preview import noise2d  # noqa: E402

PASS_THROUGH = {"minecraft:interpolated", "minecraft:blend_density", "minecraft:flat_cache",
                "minecraft:cache_2d", "minecraft:cache_once", "minecraft:cache_all_in_cell"}


class Ctx:
    def __init__(self, noises: dict, xs: np.ndarray, zs: np.ndarray, size: int):
        self.noises = noises
        self.X = xs[None, :]
        self.Z = zs[:, None]
        self.size = size
        self._cache: dict[str, np.ndarray] = {}

    def band(self, noise_id: str, xz_scale: float) -> np.ndarray:
        """Approximate the vanilla noise entry with the generator's own noise (same wavelength)."""
        key = f"{noise_id}@{xz_scale}"
        if key in self._cache:
            return self._cache[key]
        entry = self.noises.get(noise_id)
        if entry is None:
            raise KeyError(f"noise {noise_id} not found in the datapack "
                           f"(have: {sorted(self.noises)[:8]})")
        feature = (1.0 / (2.0 ** entry["firstOctave"])) / max(xz_scale, 1e-6)
        freq = self.size / max(feature, 1e-6)
        rng = np.random.default_rng(zlib.crc32(noise_id.encode()) % (2**32))
        out = noise2d(self.size, self.size, freq, rng, octaves=len(entry["amplitudes"]))
        self._cache[key] = out
        return out


def ev(node, y: np.ndarray, ctx: Ctx):
    """node: density function JSON (or number); y: per-column array (H, W)."""
    if isinstance(node, (int, float)):
        return node
    if isinstance(node, str):
        raise ValueError(f"unresolved reference {node}")
    t = node.get("type")
    if t == "minecraft:noise":
        return ctx.band(node["noise"], float(node.get("xz_scale", 1.0)))
    if t == "minecraft:y_clamped_gradient":
        fy, ty = node["from_y"], node["to_y"]
        fv, tv = node["from_value"], node["to_value"]
        t = (np.clip(y, fy, ty) - fy) / max(ty - fy, 1e-9)
        return fv + t * (tv - fv)
    if t in PASS_THROUGH:
        return ev(node["argument"], y, ctx)
    if t == "minecraft:abs":
        return np.abs(ev(node["argument"], y, ctx))
    if t == "minecraft:clamp":
        return np.clip(ev(node["input"], y, ctx), node["min"], node["max"])
    if t == "minecraft:squeeze":
        d = ev(node["argument"], y, ctx)
        return np.sign(d) * np.abs(d) ** 0.5           # monotonic, zero-preserving
    if t in ("minecraft:add", "minecraft:mul"):
        a, b = node["argument1"], node["argument2"]
        fa, fb = ev(a, y, ctx), ev(b, y, ctx)
        return fa + fb if t.endswith("add") else fa * fb
    if t in ("minecraft:min", "minecraft:max"):
        fa, fb = ev(node["argument1"], y, ctx), ev(node["argument2"], y, ctx)
        return np.minimum(fa, fb) if t.endswith("min") else np.maximum(fa, fb)
    if t == "minecraft:spline":
        sp = node["spline"]
        coord = ev(sp["coordinate"], y, ctx)
        locs = np.array([p["location"] for p in sp["points"]], dtype=np.float64)
        vals = np.array([p["value"] if isinstance(p["value"], (int, float))
                         else np.mean(ev(p["value"], y, ctx)) for p in sp["points"]], dtype=np.float64)
        return np.interp(coord, locs, vals).astype(np.float64)
    raise ValueError(f"unsupported node type {t}")


def solve_heights(df: dict, ctx: Ctx, lo: float = -64.0, hi: float = 320.0, iters: int = 30):
    """Bisection on density == 0, per column. Density is positive below the surface."""
    a = np.full((ctx.size, ctx.size), lo, dtype=np.float64)
    b = np.full((ctx.size, ctx.size), hi, dtype=np.float64)
    for _ in range(iters):
        mid = 0.5 * (a + b)
        d = ev(df, mid, ctx)
        below = d > 0                                   # density positive = inside the ground
        a = np.where(below, mid, a)
        b = np.where(below, b, mid)
    return 0.5 * (a + b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datapack", required=True)
    ap.add_argument("--namespace", default="fit")
    ap.add_argument("--name", default="fit")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--target", help="metrics JSON to compare against")
    args = ap.parse_args()

    root = Path(args.datapack)
    ns, name = args.namespace, args.name
    settings = json.loads((root / "data" / ns / "worldgen" / "noise_settings" / f"{name}.json")
                          .read_text(encoding="utf-8"))
    noises = {}
    ndir = root / "data" / ns / "worldgen" / "noise"
    for f in sorted(ndir.glob("*.json")):
        noises[f"{ns}:{f.stem}"] = json.loads(f.read_text(encoding="utf-8"))

    np.random.seed(args.seed)
    xs = (np.arange(args.size) + 0.5) / args.size
    zs = (np.arange(args.size) + 0.5) / args.size
    ctx = Ctx(noises, xs * args.size, zs * args.size, args.size)

    h = solve_heights(settings["noise_router"]["final_density"], ctx)

    sea = float(settings.get("sea_level", 63))
    water = np.where(h < sea, sea, -1e4).astype(np.float32)
    flags = np.zeros(h.shape, dtype=np.uint8)
    m = compute(h.astype(np.float32), water, flags)
    print("exported model:", summary_line(m))
    print(f"  span {m['relief']['span']:.0f}  min {h.min():.0f}  max {h.max():.0f}  sea {sea:.0f}")
    print("  bands:", {k: round(v, 1) for k, v in m["spectrum"]["bands_pct"].items()
                        if k in (">=512", "256-512", "128-256", "64-128", "32-64")})
    hp = m["hypsometry"]
    print(f"  hypsometry: integral {hp['integral']:.3f}  p25/50/75 {hp['p25']:.2f}/{hp['p50']:.2f}/{hp['p75']:.2f}")

    if args.target:
        tgt = json.loads(Path(args.target).read_text(encoding="utf-8"))
        r = compare(tgt, m)
        print(f"\ndistance to target: {r['distance']:.3f}")
        for k, v in list(r["residuals"].items())[:8]:
            print(f"  {k:32} target {v['a']:>9.2f}  export {v['b']:>9.2f}  log {v['log_ratio']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
