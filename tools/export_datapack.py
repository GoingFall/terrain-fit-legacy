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

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model_v2 import Params2  # noqa: E402

TEMPLATE_URL = ("https://raw.githubusercontent.com/misode/mcmeta/1.20.1-data/"
                "data/minecraft/worldgen/noise_settings/overworld.json")
CACHE = Path("_tpl")
SIZE = 512.0        # the generator's own domain, in blocks: wavelength = SIZE / freq
NS = "fit"          # set from --namespace in main()


def band_noise(freq: float, octaves: int = 1) -> tuple[dict, dict]:
    """One of our spectral bands -> (noise registry entry, density-function node).

    A vanilla noise entry gives features of 1/2^firstOctave blocks at xz_scale 1, so the octave is
    chosen from the target wavelength and xz_scale does the fine tuning. This is what makes the
    export keep the frequencies the fit chose instead of inheriting vanilla's.
    """
    import math
    feature = SIZE / max(freq, 1e-3)                 # blocks per feature
    octave = -int(round(math.log2(max(feature, 1.0))))
    scale = feature * (2.0 ** octave)
    entry = {"firstOctave": octave,
             "amplitudes": [1.0 / (2 ** i) for i in range(max(1, octaves))]}
    node = {"type": "minecraft:noise", "noise": None, "xz_scale": round(scale, 5), "y_scale": 0.0}
    return entry, node


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
def height_df(p: Params2) -> tuple[dict, dict]:
    """Compose our model out of vanilla's climate fields.

    A density-function cannot invent new noise wavelengths without new noise registry entries, so
    the bands map onto vanilla's own fields (continents / erosion / ridges / vegetation /
    temperature). The *shapes* (splines, masks, gating, amplitudes) are ours; the wavelengths are
    vanilla's -- the export is an approximation, documented in the README.
    """
    noises = {}

    def band(freq: float, octaves: int = 1):
        entry, node = band_noise(freq, octaves)
        name = f"band{len(noises)}"
        node["noise"] = f"{NS}:{name}"
        noises[name] = entry
        return node

    CONT = band(p.continent_freq, 2)
    EROS = band(p.massif_freq, 2)
    RIDG = band(p.ridge_freq, 2)
    TEMP = band(p.hill_freq, 2)
    VEG = band(p.mountain_freq, 2)

    # continent -> vertical profile (the spline is our hypsometry curve)
    cont = spline(CONT, [(-1.0, 0.45), (-0.4, 0.68), (0.0, 0.82), (0.5, 0.92), (1.0, 1.0)])
    h = add(const(p.base_level), mul(const(p.relief_macro), cont))

    # mountain mask from vegetation, gating the big relief
    mask = smoothstep_spline(VEG, p.mountain_threshold, p.mountain_threshold + p.mountain_softness)

    # massif body
    massif = spline(EROS, [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)])
    h = add(h, mul(const(p.massif_amp), mul(massif, mask)))

    # ridge crests: fold our own band (1 - |2|n| - 1|) then sharpen with a spline
    folded = add(1.0, mul(-1.0, absf(add(mul(2.0, absf(RIDG)), -1.0))))
    sharp = spline(folded, [(0.0, 0.0), (0.5, 1.0 - 0.35 ** p.ridge_sharp), (1.0, 1.0)])
    h = add(h, mul(const(p.ridge_amp), mul(sharp, mask)))

    # hills, and spikes gated by the hills (RTG)
    hill = spline(TEMP, [(-1.0, 0.0), (0.0, 0.5), (1.0, 1.0)])
    h = add(h, mul(const(p.hill_amp), add(mul(const(2.0), hill), const(-1.0))))
    spike = smoothstep_spline(TEMP, p.spike_threshold, 1.0)
    h = add(h, mul(const(p.spike_amp), mul(spike, hill)))

    # river band around the zero contour
    river = spline(absf(band(p.river_freq, 1)), [(0.0, 1.0), (p.river_width, 0.0), (1.0, 0.0)])
    h = add(h, mul(const(-p.river_depth), river))
    return h, noises


def final_density(p: Params2) -> tuple[dict, dict]:
    # density = H(x,z) - y  -> positive below the surface, negative above
    h, noises = height_df(p)
    return clampf(add(h, mul(const(-1.0), y_axis())), -8.0, 8.0), noises


def biome_source(spec: str) -> dict:
    """One biome -> minecraft:fixed; several -> minecraft:multi_noise with explicit parameters.

    multi_noise picks by climate, so a list is split across the temperature axis (cold to warm),
    which is the closest a datapack gets to "snow at altitude" without touching the height field.
    """
    names = [b.strip() for b in spec.split(",") if b.strip()]
    if len(names) == 1:
        return {"type": "minecraft:fixed", "biome": names[0]}
    full = {"min_inclusive": -1.0, "max_inclusive": 1.0}
    step = 2.0 / len(names)
    out = []
    for i, name in enumerate(names):
        lo = -1.0 + i * step
        out.append({
            "biome": name,
            "parameters": {
                "temperature": {"min_inclusive": round(lo, 4), "max_inclusive": round(lo + step, 4)},
                "humidity": full, "continentalness": full, "erosion": full,
                "weirdness": full, "depth": 0, "offset": 0,
            },
        })
    return {"type": "minecraft:multi_noise", "biomes": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="best_*.json from fit_metrics")
    ap.add_argument("--namespace", default="fit")
    ap.add_argument("--out", default="dist/fit-datapack")
    ap.add_argument("--name", default="fit")
    ap.add_argument("--target", help="metrics JSON: match its relief range and snow line")
    ap.add_argument("--biome", default="minecraft:snowy_plains",
                    help="one biome (fixed source) or a comma-separated list (multi_noise, "
                         "split by temperature)")
    ap.add_argument("--window", type=int, default=512,
                    help="measurement window in blocks, matched to the reference region")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="blocks to add after normalisation (calibrate from an in-game measurement)")
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

    global NS
    NS = args.namespace
    fd, band_noises = final_density(p)
    settings = json.loads(json.dumps(tpl))          # deep copy
    settings["noise_router"]["final_density"] = fd
    settings["noise_router"]["initial_density_without_jaggedness"] = fd
    settings["aquifers_enabled"] = False        # aquifers punch holes into a custom height field

    # --- normalise the height field and set the snow line, decided offline (no game needed)
    if args.target:
        import df_eval
        tgt = json.loads(Path(args.target).read_text(encoding="utf-8"))
        lo_t, hi_t = tgt["relief"]["min"], tgt["relief"]["max"]
        np_x = np_z = 512
        xs = (np.arange(np_x) + 0.5)
        zs = (np.arange(np_z) + 0.5)
        noise_registry = {f"{args.namespace}:{k}": v for k, v in band_noises.items()}
        ctx = df_eval.Ctx(noise_registry, xs, zs, np_x)
        hdf0, _ = height_df(p)
        h = np.asarray(df_eval.ev(hdf0, np.zeros((np_x, np_z)), ctx), dtype=np.float64)
        a = (hi_t - lo_t) / max(float(h.max() - h.min()), 1e-3)
        b = lo_t - a * float(h.min()) + args.offset
        print(f"   height normalisation: raw {h.min():.0f}..{h.max():.0f} -> "
              f"{lo_t:.0f}..{hi_t:.0f}  (x{a:.3f} +{b:.1f})")
        hdf, _ = height_df(p)
        scaled = add(mul(hdf, a), b)
        bounded = clampf(scaled, lo_t, hi_t)
        settings["noise_router"]["final_density"] = clampf(
            add(bounded, mul(const(-1.0), y_axis())), -8.0, 8.0)
        settings["noise_router"]["initial_density_without_jaggedness"] = settings["noise_router"]["final_density"]

        h2 = df_eval.solve_heights(settings["noise_router"]["final_density"], ctx,
                                   lo=-64.0, hi=320.0, iters=40)
        # the target keeps only a sliver of water at the very bottom; a quantile of the solved
        # distribution is not trustworthy (it is the offline noise, not the game's)
        settings["sea_level"] = int(round(lo_t)) + 1
        print(f"   sea level {settings['sea_level']} (target water {tgt['cover']['water_pct']:.2f}%)"
              f"  range {h2.min():.0f}..{h2.max():.0f}")

        # snow above the fitted snow line, before vanilla's biome surface rules
        snow_line = tgt.get("snow_line") or (lo_t + 0.6 * (hi_t - lo_t))
        rule = settings["surface_rule"]
        seq = rule.get("sequence") if isinstance(rule, dict) else None
        if seq is not None:
            seq.insert(0, {
                "type": "minecraft:condition",
                "if_true": {"type": "minecraft:y_above", "add_stone_depth": False,
                            "anchor": {"absolute": int(round(snow_line))},
                            "surface_depth_multiplier": 2},
                "then_run": {"type": "minecraft:block",
                             "result_state": {"Name": "minecraft:snow"}},
            })
            print(f"   snow surface rule: y > {int(round(snow_line))} -> snow_block")

    out = Path(args.out)
    ns, name = args.namespace, args.name
    (out / "data" / ns / "worldgen" / "noise_settings").mkdir(parents=True, exist_ok=True)
    (out / "data" / ns / "dimension_type").mkdir(parents=True, exist_ok=True)
    (out / "data" / ns / "dimension").mkdir(parents=True, exist_ok=True)

    # self-check: surface rule nodes must use the exact 1.20.1 field names
    REQUIRED = {
        "minecraft:block": {"result_state"},
        "minecraft:y_above": {"add_stone_depth", "anchor", "surface_depth_multiplier"},
        "minecraft:condition": {"if_true", "then_run"},
    }

    def check_rule(n, path="surface_rule"):
        if isinstance(n, dict):
            t = n.get("type")
            if t in REQUIRED:
                missing = REQUIRED[t] - set(n)
                if missing:
                    raise SystemExit(f"{path}: {t} missing {sorted(missing)}")
            for k, v in n.items():
                check_rule(v, f"{path}.{k}")
        elif isinstance(n, list):
            for i, v in enumerate(n):
                check_rule(v, f"{path}[{i}]")
    check_rule(settings["surface_rule"])
    print("   surface rule schema: ok")

    noise_dir = out / "data" / ns / "worldgen" / "noise"
    noise_dir.mkdir(parents=True, exist_ok=True)
    for band_name, entry in band_noises.items():
        (noise_dir / f"{band_name}.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")

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
    (out / "data" / ns / "dimension_type" / f"{name}.json").write_text(
        dt_path.read_text(encoding="utf-8"), encoding="utf-8")

    (out / "data" / ns / "dimension" / f"{name}.json").write_text(json.dumps({
        "type": f"{ns}:{name}",
        "generator": {
            "type": "minecraft:noise",
            "settings": f"{ns}:{name}",
            "biome_source": biome_source(args.biome),
        },
    }, indent=2), encoding="utf-8")

    half = args.window // 2
    manifest = {
        "dimension": f"{ns}:{name}",
        "biome": args.biome,
        "window": {"size": args.window, "x": [-half, half], "z": [-half, half]},
        "target": args.target,
        "offset": args.offset,
        "params": {k: v for k, v in vars(p).items() if not k.startswith("_")},
        "measure": f"python tools/terrain_metrics.py \"<world>/dimensions/{ns}/{name}\" "
                   f"--region r.-1.-1 --json ingame.json",
        "teleport": f"/execute in {ns}:{name} run tp @s 0 200 0",
    }
    (out / "fit-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"-> {out}")
    print(f"   dimension: {ns}:{name}")
    print(f"   install:   copy the folder into <world>/datapacks/ and /reload")
    print(f"   biome:     {args.biome} ({'fixed' if ',' not in args.biome else 'multi_noise by temperature'})")
    print(f"   window:    {args.window} blocks, same size as the reference region")
    print(f"   enter:     /execute in {ns}:{name} run tp @s 0 200 0")
    print("   dropped (not expressible in density functions): directional valley axis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
