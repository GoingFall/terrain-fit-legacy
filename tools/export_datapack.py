#!/usr/bin/env python3
"""Export a fitted parameter set as a 1.20.1 datapack (custom dimension + noise settings).

The datapack reuses vanilla noises (no custom noise registry needed) and only replaces the density
router, so it is easy to verify by eye and by measurement:

  /execute in fit:fit run tp @s 0 120 0
  python tools/terrain_metrics.py "<save>"        # measure what the game actually generated

What is expressible in density-function JSON: continent spline, massif mask, folded ridges, hills,
hill-gated spikes, river bands. What is NOT: the directional valley axis (needs a rotated frame,
which the DF language cannot express) -- it is dropped and reported.

Usage:
  python export_datapack.py --params preview/best_v7.json --out dist/fit-datapack
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model_v2 import Params2  # noqa: E402

TEMPLATE_URL = ("https://raw.githubusercontent.com/misode/mcmeta/1.20.1-data/"
                "data/minecraft/worldgen/noise_settings/overworld.json")
CACHE = Path("_tpl")


# ----------------------------------------------------------------- density helpers
def const(v: float):
    """Operands may be bare numbers; that is what vanilla's own density functions do."""
    return float(v)


def add(a: dict, b: dict) -> dict:
    return {"type": "minecraft:add", "argument1": a, "argument2": b}


def mul(a: dict, b: dict) -> dict:
    return {"type": "minecraft:mul", "argument1": a, "argument2": b}


def absf(a: dict) -> dict:
    return {"type": "minecraft:abs", "argument": a}


def clampf(a: dict, lo: float, hi: float) -> dict:
    return {"type": "minecraft:clamp", "input": a, "min": lo, "max": hi}


def noise(node_id: str, scale: float = 1.0, y_scale: float = 0.0) -> dict:
    return {"type": "minecraft:noise", "noise": node_id,
            "xz_scale": scale, "y_scale": y_scale}


def spline(coord, points: list[tuple[float, float]]) -> dict:
    """Piecewise-linear spline over a density function.

    Schema verified against vanilla's overworld/factor.json: every point is
    {"location": float, "value": <number or DF>, "derivative": float}.
    """
    return {"type": "minecraft:spline", "spline": {
        "coordinate": coord,
        "points": [{"location": float(x), "value": float(y), "derivative": 0.0}
                   for x, y in points],
    }}


def smoothstep_spline(coord, lo: float, hi: float, lo_v: float = 0.0, hi_v: float = 1.0) -> dict:
    """Spline step from lo to hi. Locations must be strictly increasing and inside (-1, 1),
    otherwise duplicate points are emitted and the pack fails to parse."""
    eps = 1e-3
    lo = max(-1.0 + eps, min(1.0 - 2 * eps, lo))
    hi = max(lo + eps, min(1.0 - eps, hi))
    pts = [(-1.0, lo_v), (lo, lo_v), (hi, hi_v), (1.0, hi_v)]
    dedup = []
    for x, y in pts:
        if dedup and abs(x - dedup[-1][0]) < 1e-6:
            dedup[-1] = (x, y)
        else:
            dedup.append((x, y))
    return spline(coord, dedup)


def y_axis() -> dict:
    """A density function that returns the block y (so `H - y` becomes a density)."""
    return {"type": "minecraft:y_clamped_gradient",
            "from_y": -64, "to_y": 320, "from_value": -64.0, "to_value": 320.0}


# ----------------------------------------------------------------- model -> DF
def height_df(p: Params2) -> dict:
    """Compose our model out of vanilla's climate fields.

    A density-function cannot invent new noise wavelengths without new noise registry entries, so
    the bands map onto vanilla's own fields (continents / erosion / ridges / vegetation /
    temperature). The *shapes* (splines, masks, gating, amplitudes) are ours; the wavelengths are
    vanilla's -- the export is an approximation, documented in the README.
    """
    CONT = "minecraft:overworld/continents"
    EROS = "minecraft:overworld/erosion"
    RIDG = "minecraft:overworld/ridges"
    FOLD = "minecraft:overworld/ridges_folded"
    TEMP = noise("minecraft:temperature", scale=0.25)      # raw noise entries, not DF files
    VEG = noise("minecraft:vegetation", scale=0.25)

    # continent -> vertical profile (the spline is our hypsometry curve)
    cont = spline(CONT, [(-1.0, 0.0), (-0.1, 0.42), (0.3, 0.62), (1.0, 1.0)])
    h = add(const(p.base_level), mul(const(p.relief_macro), cont))

    # mountain mask from vegetation, gating the big relief
    mask = smoothstep_spline(VEG, p.mountain_threshold, p.mountain_threshold + p.mountain_softness)

    # massif body
    massif = spline(EROS, [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)])
    h = add(h, mul(const(p.massif_amp), mul(massif, mask)))

    # ridge crests: vanilla already ships the folded ridge field
    sharp = spline(FOLD, [(-1.0, 0.0), (0.0, 0.15), (0.5, 0.35 ** p.ridge_sharp), (1.0, 1.0)])
    h = add(h, mul(const(p.ridge_amp), mul(sharp, mask)))

    # hills, and spikes gated by the hills (RTG)
    hill = spline(TEMP, [(-1.0, 0.0), (0.0, 0.5), (1.0, 1.0)])
    h = add(h, mul(const(p.hill_amp), add(mul(const(2.0), hill), const(-1.0))))
    spike = smoothstep_spline(TEMP, p.spike_threshold, 1.0)
    h = add(h, mul(const(p.spike_amp), mul(spike, hill)))

    # river band around the zero contour of erosion
    band = spline(absf(EROS), [(0.0, 1.0), (p.river_width, 0.0), (1.0, 0.0)])
    h = add(h, mul(const(-p.river_depth), band))
    return h


def final_density(p: Params2) -> dict:
    # density = H(x,z) - y  -> positive below the surface, negative above
    return clampf(add(height_df(p), mul(const(-1.0), y_axis())), -8.0, 8.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="best_*.json from fit_metrics")
    ap.add_argument("--namespace", default="fit")
    ap.add_argument("--out", default="dist/fit-datapack")
    ap.add_argument("--name", default="fit")
    args = ap.parse_args()

    best = json.loads(Path(args.params).read_text(encoding="utf-8"))
    p = Params2(**{**{f: getattr(Params2(), f) for f in Params2.__dataclass_fields__},
                   **best["params"]})

    CACHE.mkdir(exist_ok=True)
    tpl_path = CACHE / "overworld.json"
    if not tpl_path.exists():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        raw = opener.open(urllib.request.Request(TEMPLATE_URL, headers={"User-Agent": "dsh/1.0"}),
                          timeout=30).read()
        tpl_path.write_bytes(raw)
        print(f"cached template -> {tpl_path}")
    tpl = json.loads(tpl_path.read_text(encoding="utf-8"))

    settings = json.loads(json.dumps(tpl))          # deep copy
    settings["noise_router"]["final_density"] = final_density(p)
    settings["noise_router"]["initial_density_without_jaggedness"] = final_density(p)
    settings["sea_level"] = int(round(best["metrics"]["cover"]["water_pct"] * 0 + p.sea_level))

    out = Path(args.out)
    ns, name = args.namespace, args.name
    (out / "data" / ns / "worldgen" / "noise_settings").mkdir(parents=True, exist_ok=True)
    (out / "data" / ns / "worldgen" / "dimension_type").mkdir(parents=True, exist_ok=True)
    (out / "data" / ns / "dimension").mkdir(parents=True, exist_ok=True)

    (out / "pack.mcmeta").write_text(json.dumps({
        "pack": {"pack_format": 15,
                 "description": f"terrain-fit export ({best['distance']:.3f} dist) — /execute in {ns}:{name} run tp @s 0 140 0"}
    }, indent=2), encoding="utf-8")

    (out / "data" / ns / "worldgen" / "noise_settings" / f"{name}.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8")

    dt_path = CACHE / "dimension_type_overworld.json"
    if not dt_path.exists():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        raw = opener.open(urllib.request.Request(
            "https://raw.githubusercontent.com/misode/mcmeta/1.20.1-data/"
            "data/minecraft/dimension_type/overworld.json",
            headers={"User-Agent": "dsh/1.0"}), timeout=30).read()
        dt_path.write_bytes(raw)
    (out / "data" / ns / "worldgen" / "dimension_type" / f"{name}.json").write_text(
        dt_path.read_text(encoding="utf-8"), encoding="utf-8")

    (out / "data" / ns / "dimension" / f"{name}.json").write_text(json.dumps({
        "type": f"{ns}:{name}",
        "generator": {
            "type": "minecraft:noise",
            "settings": f"{ns}:{name}",
            "biome_source": {"type": "minecraft:multi_noise", "preset": "minecraft:overworld"},
        },
    }, indent=2), encoding="utf-8")

    print(f"-> {out}")
    print(f"   dimension: {ns}:{name}")
    print(f"   install:   copy the folder into <world>/datapacks/ and /reload")
    print(f"   enter:     /execute in {ns}:{name} run tp @s 0 140 0")
    print("   dropped (not expressible in density functions): directional valley axis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
