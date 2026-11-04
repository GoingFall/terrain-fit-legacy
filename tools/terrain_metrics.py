#!/usr/bin/env python3
"""Terrain metrics for evaluation: the numbers you compare between a reference map and a
generated one.

Everything is derived from the same layers the viewer uses (ground / water / flags), so the
panel in the viewer and this CLI always agree.

Usage:
  python terrain_metrics.py "<save>" --region r.0.0 --json metrics.json
  python terrain_metrics.py "<save>"                       # whole map, printed table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from world_layers import collect_layers  # noqa: E402


def _neighbour(fn, h):
    """Apply fn(shifted) over the 8 neighbours, edge-padded (no wrap-around)."""
    out = None
    for dz in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dz == 0 and dx == 0:
                continue
            pad = np.pad(h, 1, mode="edge")
            shifted = pad[1 + dz : 1 + dz + h.shape[0], 1 + dx : 1 + dx + h.shape[1]]
            out = fn(out, shifted)
    return out


def ridge_density(h: np.ndarray, valid: np.ndarray, radius: int = 5) -> float:
    """Local maxima of the *smoothed* surface per km2 (major crests, not 1-block steps).

    Counting raw maxima on integer heights is meaningless (plateaus and single-block noise
    dominate), so the surface is box-blurred first; radius 5 keeps features of ~10 blocks.
    """
    smooth = h.astype(np.float32)
    for _ in range(radius):
        pad = np.pad(smooth, 1, mode="edge")
        smooth = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
                  + 4.0 * smooth) / 8.0
    higher = _neighbour(lambda acc, s: s if acc is None else np.maximum(acc, s), smooth)
    edge = valid.copy()
    edge[0, :] = edge[-1, :] = False
    edge[:, 0] = edge[:, -1] = False
    crest = edge & (smooth > higher + 1e-4)       # strictly higher than all 8 neighbours
    area_km2 = float(valid.sum()) / 1e6           # 1 column == 1 m2
    return float(crest.sum() / area_km2) if area_km2 else 0.0


def spectrum(ground: np.ndarray, valid: np.ndarray, max_side: int = 1024) -> dict:
    """Radial power spectrum of the height field, binned by wavelength.

    Answers "how busy is the surface at each scale" — the thing slope percentiles cannot see.
    Returns energy share per band plus the log-log slope of power vs frequency.
    """
    ys, xs = np.where(valid)
    if xs.size < 64:
        return {}
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = ground[y0:y1, x0:x1].astype(np.float64)
    mask = valid[y0:y1, x0:x1]
    crop = np.where(mask, crop, float(np.median(crop[mask])))
    crop -= crop.mean()
    h, w = crop.shape
    step = max(1, int(np.ceil(max(h, w) / max_side)))
    crop = crop[::step, ::step]
    h, w = crop.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    spec = np.abs(np.fft.rfft2(crop * win)) ** 2
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    freq = np.hypot(fy, fx)                             # cycles per block
    with np.errstate(divide="ignore"):
        wl = 1.0 / freq
    total = spec.sum() or 1.0

    edges = [(None, 512), (512, 256), (256, 128), (128, 64), (64, 32),
             (32, 16), (16, 8), (8, 0)]
    bands = {}
    for lo, hi in edges:
        if lo is None:
            m = wl >= hi
            name = f">={hi}"
        elif hi == 0:
            m = wl < lo
            name = f"<{lo}"
        else:
            m = (wl >= hi) & (wl < lo)
            name = f"{hi}-{lo}"
        bands[name] = float(spec[m].sum() / total * 100)

    # log-log slope over the resolved range (exclude DC and the noise floor)
    sel = (freq > 1.0 / 512) & (freq < 1.0 / 8)
    if sel.any():
        lf = np.log10(freq[sel])
        lp = np.log10(np.maximum(spec[sel], 1e-20))
        slope = float(np.polyfit(lf, lp, 1)[0])
    else:
        slope = 0.0
    return {"bands_pct": bands, "spectral_slope": slope}


def hypsometry(ground: np.ndarray, valid: np.ndarray, bins: int = 16) -> dict:
    """Elevation-area curve: a basin keeps most of its area low, a plateau keeps it mid-high.

    This is the cheapest descriptor that separates "bowl" from "hills" from "mesa" — the shape
    information that slope percentiles and spectra both throw away.
    """
    h = ground[valid]
    if h.size == 0:
        return {}
    lo, hi = float(h.min()), float(h.max())
    t = (h - lo) / max(hi - lo, 1e-6)
    counts, _ = np.histogram(t, bins=bins, range=(0.0, 1.0))
    return {
        "integral": float(t.mean()),                  # 0.5 = symmetric, <0.5 = basin, >0.5 = mesa
        "p25": float(np.percentile(t, 25)),
        "p50": float(np.percentile(t, 50)),
        "p75": float(np.percentile(t, 75)),
        "bins_pct": [float(c / h.size * 100) for c in counts],
    }


def anisotropy(ground: np.ndarray, valid: np.ndarray, max_side: int = 1024) -> dict:
    """Directional energy in the mid-frequency band: ring ridges are isotropic, a wall is not."""
    ys, xs = np.where(valid)
    if xs.size < 64:
        return {}
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = ground[y0:y1, x0:x1].astype(np.float64)
    mask = valid[y0:y1, x0:x1]
    crop = np.where(mask, crop, float(np.median(crop[mask])))
    crop -= crop.mean()
    step = max(1, int(np.ceil(max(crop.shape) / max_side)))
    crop = crop[::step, ::step]
    h, w = crop.shape
    spec = np.abs(np.fft.rfft2(crop * np.outer(np.hanning(h), np.hanning(w)))) ** 2
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    freq = np.hypot(fy, fx)
    with np.errstate(divide="ignore"):
        wl = 1.0 / freq
    band = (wl >= 64) & (wl <= 512)
    if not band.any():
        return {}
    ang = np.degrees(np.arctan2(fy, fx)) % 180.0
    sectors = np.floor(ang / 22.5).astype(int) % 8
    energy = np.zeros(8)
    for i in range(8):
        m = band & (sectors == i)
        energy[i] = spec[m].sum()
    tot = energy.sum() or 1.0
    share = energy / tot
    return {
        "ratio": float(share.max() / max(share.min(), 1e-6)),
        "angle_deg": float((int(np.argmax(share)) + 0.5) * 22.5),
        "sectors_pct": [float(v * 100) for v in share],
    }


def compute(ground: np.ndarray, water: np.ndarray, flags: np.ndarray,
            valid: np.ndarray | None = None) -> dict:
    if valid is None:
        valid = ground > -63
    h = ground[valid]
    if h.size == 0:
        return {"columns": 0}

    gy, gx = np.gradient(ground)
    slope = np.hypot(gx, gy)[valid]
    denom = max(int(valid.sum()), 1)          # percentages are over valid columns, not the grid
    wet = (water > ground + 0.5) & valid
    snow = ((flags & 1) > 0) & valid
    ice = ((flags & 2) > 0) & valid
    canopy = ((flags & 4) > 0) & valid

    return {
        "columns": int(valid.size),
        "valid_columns": int(h.size),
        "relief": {
            "min": float(h.min()), "max": float(h.max()),
            "span": float(h.max() - h.min()), "mean": float(h.mean()),
        },
        "slope": {
            "p50": float(np.percentile(slope, 50)),
            "p90": float(np.percentile(slope, 90)),
            "p99": float(np.percentile(slope, 99)),
            "max": float(slope.max()),
            "gt2_pct": float((slope > 2).mean() * 100),
            "gt4_pct": float((slope > 4).mean() * 100),
            "flat_pct": float((slope < 0.3).mean() * 100),
        },
        "cover": {
            "water_pct": float(wet.sum() / denom * 100),
            "snow_pct": float(snow.sum() / denom * 100),
            "ice_pct": float(ice.sum() / denom * 100),
            "canopy_pct": float(canopy.sum() / denom * 100),
        },
        "snow_line": float(np.median(ground[snow])) if snow.any() else None,
        "ridge_per_km2": ridge_density(ground, valid),
        "spectrum": spectrum(ground, valid),
        "hypsometry": hypsometry(ground, valid),
        "anisotropy": anisotropy(ground, valid),
    }


def summary_line(m: dict) -> str:
    if not m.get("valid_columns"):
        return "no data"
    r, s, c = m["relief"], m["slope"], m["cover"]
    return (f"relief {r['span']:.0f} | slope {s['p50']:.2f}/{s['p90']:.2f}/{s['p99']:.2f}"
            f" | water {c['water_pct']:.1f}% | snow {c['snow_pct']:.1f}%"
            f" | ridges {m['ridge_per_km2']:.0f}/km2")


# ---------------------------------------------------------------- comparison

WEIGHTS = {
    "relief.span": 1.0,
    "slope.p50": 1.0,
    "slope.p90": 1.0,
    "slope.p99": 1.5,          # the tail is what makes hand-made terrain look hand-made
    "slope.gt2_pct": 1.0,
    "slope.gt4_pct": 1.0,
    "slope.flat_pct": 0.5,
    "cover.water_pct": 1.0,
    "cover.snow_pct": 0.5,
    "ridge_per_km2": 1.0,
    # radial spectrum: the target has almost no energy below 128 blocks, and a fit that adds
    # 64-128 "small ridges" looks busy even when every slope percentile matches
    "spectrum.bands_pct.256-512": 0.8,
    "spectrum.bands_pct.128-256": 0.8,
    "spectrum.bands_pct.64-128": 1.0,
    "spectrum.bands_pct.32-64": 1.0,
    # structural descriptors: where the area sits vertically, and whether ridges form rings
    "hypsometry.integral": 1.2,
    "hypsometry.p25": 0.6,
    "hypsometry.p50": 0.6,
    "hypsometry.p75": 0.6,
    "anisotropy.ratio": 0.8,
}
EPS = 0.05
EPS_PCT = 0.5          # for percentage metrics that can be near zero


def _get(m: dict, path: str) -> float:
    cur = m
    for part in path.split("."):
        cur = cur[part]
    return float(cur)


def compare(a: dict, b: dict) -> dict:
    """Per-metric log-ratio and a single weighted distance (0 = identical)."""
    rows, total, wsum = {}, 0.0, 0.0
    for key, w in WEIGHTS.items():
        try:
            va, vb = _get(a, key), _get(b, key)
        except (KeyError, TypeError):
            continue                     # metric missing on one side (older JSON) -> skip
        if va is None or vb is None:
            continue
        eps = EPS_PCT if ("bands_pct" in key or "sectors_pct" in key) else EPS
        la = np.log(max(va, eps))
        lb = np.log(max(vb, eps))
        d = abs(la - lb)
        rows[key] = {"a": round(va, 4), "b": round(vb, 4),
                     "log_ratio": round(float(la - lb), 4), "weight": w}
        total += w * d
        wsum += w
    return {"distance": round(float(total / max(wsum, 1e-9)), 4),
            "residuals": dict(sorted(rows.items(), key=lambda kv: -abs(kv[1]["log_ratio"])))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("--region")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                    help="compare two metrics files instead of measuring a world")
    args = ap.parse_args()

    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text(encoding="utf-8"))
        b = json.loads(Path(args.compare[1]).read_text(encoding="utf-8"))
        res = compare(a, b)
        print(f"distance = {res['distance']:.4f}   (0 = identical)")
        for k, v in res["residuals"].items():
            print(f"  {k:20} {v['a']:>10.3f} vs {v['b']:>10.3f}   "
                  f"log-ratio {v['log_ratio']:+.3f}  w={v['weight']}")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        return 0

    ground, water, canopy, flags, sid, x0, z0 = collect_layers(
        Path(args.world), args.region, workers=args.workers, fill_missing=False)
    m = compute(ground, water, flags)
    m["origin"] = [int(x0), int(z0)]
    m["region"] = args.region or "all"

    print(summary_line(m))
    for group in ("relief", "slope", "cover"):
        print(f"  {group}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                         for k, v in m[group].items()))
    print(f"  snow_line={m['snow_line']}  ridge_per_km2={m['ridge_per_km2']:.1f}")
    if m.get("hypsometry"):
        hp = m["hypsometry"]
        print(f"  hypsometry: integral {hp['integral']:.3f}  p25/50/75 "
              f"{hp['p25']:.2f}/{hp['p50']:.2f}/{hp['p75']:.2f}")
    if m.get("anisotropy"):
        an = m["anisotropy"]
        print(f"  anisotropy: ratio {an['ratio']:.2f}  dominant {an['angle_deg']:.0f} deg")
    if m.get("spectrum"):
        sp = m["spectrum"]
        print(f"  spectrum: slope {sp['spectral_slope']:+.3f}")
        print("    " + "  ".join(f"{k}:{v:.1f}%" for k, v in sp["bands_pct"].items()))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"  -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
