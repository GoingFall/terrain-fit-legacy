#!/usr/bin/env python3
"""Fitting loop against the mod: write config -> run Forge server -> generate -> measure.

The mod reads config/terrainfit.json at startup, so an iteration is a file write plus a server run
(~80 s) instead of a Gradle build.

Usage:
  python fit_loop_mod.py --target preview/m_r00.json
  python fit_loop_mod.py --target preview/m_r00.json --set reliefMacro=440 ridgeAmp=70
  python fit_loop_mod.py --target preview/m_r00.json --search reliefMacro=300:600:3
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from terrain_metrics import compare  # noqa: E402

FORGE = Path("_forgeserver")
ARGS = "libraries/net/minecraftforge/forge/1.20.1-47.4.28/win_args.txt"
CONFIG = FORGE / "config" / "terrainfit.json"
PY = sys.executable

DEFAULTS = {
    "seed": 1337, "baseLevel": 110.0, "reliefMacro": 180.0, "continentFreq": 1.9,
    "continentGamma": 1.7, "massifFreq": 4.0, "massifAmp": 40.0, "mountainFreq": 2.3,
    "mountainThreshold": 0.20, "mountainSoftness": 0.30, "ridgeFreq": 8.5, "ridgeAmp": 55.0,
    "ridgeSharp": 1.5, "hillFreq": 11.0, "hillAmp": 16.0, "spikeFreq": 21.0, "spikeAmp": 26.0,
    "spikeThreshold": 0.45, "erosionFreq": 3.2, "erosionFloor": 0.30, "riverFreq": 2.6,
    "riverDepth": 16.0, "riverWidth": 0.09, "seaLevel": 74, "snowLine": 193.0,
    "snowQuantile": 0.40,
}


def write_config(params: dict) -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(params, indent=2), encoding="utf-8")


def run_server(area: int, ox: int, oz: int, timeout: int = 600) -> None:
    world = FORGE / "world"
    if world.exists():
        import shutil
        shutil.rmtree(world, ignore_errors=True)
    proc = subprocess.Popen(["java", "-Xmx2G", "@" + ARGS, "nogui"], cwd=FORGE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    q: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: [q.put(l) for l in proc.stdout] or q.put(""), daemon=True).start()

    region = FORGE / "world" / "dimensions" / "fit" / "fit" / "region"
    half = max(16, area // 2)
    cmds = []
    for dx in (0, 1):
        for dz in (0, 1):
            x1, z1 = ox - half + dx * half, oz - half + dz * half
            cmds.append(f"execute in fit:fit run forceload add {x1} {z1} {x1 + half - 1} {z1 + half - 1}")
    t0 = time.time()
    started = False
    quiet = time.time()
    last = -1
    try:
        while True:
            try:
                line = q.get(timeout=0.5)
            except queue.Empty:
                line = None
            if line == "":
                break
            if line and "Done (" in line and not started:
                started = True
                for c in cmds:
                    proc.stdin.write(c + "\n")
                proc.stdin.flush()
                quiet = time.time()
            if started:
                size = sum(f.stat().st_size for f in region.glob("*.mca")) if region.exists() else 0
                if size != last:
                    last = size
                    quiet = time.time()
                elif time.time() - quiet > 20:
                    break
            if time.time() - t0 > timeout:
                print("   timeout")
                break
    finally:
        try:
            proc.stdin.write("stop\n")
            proc.stdin.flush()
            proc.wait(timeout=90)
        except Exception:
            proc.kill()


def measure() -> dict:
    world = FORGE / "world" / "dimensions" / "fit" / "fit"
    subprocess.call([PY, "tools/terrain_metrics.py", str(world), "--workers", "1",
                     "--json", "mod_ingame.json"], stdout=subprocess.DEVNULL)
    return json.loads(Path("mod_ingame.json").read_text(encoding="utf-8"))


def evaluate(params: dict, target: dict, area: int, ox: int, oz: int) -> tuple[float, dict]:
    write_config(params)
    run_server(area, ox, oz)
    got = measure()
    return compare(target, got)["distance"], got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--area", type=int, default=512)
    ap.add_argument("--origin", default="-1024,-1024")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--search", nargs="*", default=[], metavar="KEY=LO:HI:STEPS")
    args = ap.parse_args()

    ox, oz = (int(v) for v in args.origin.split(","))
    target = json.loads(Path(args.target).read_text(encoding="utf-8"))
    params = dict(DEFAULTS)
    if CONFIG.exists():
        params.update(json.loads(CONFIG.read_text(encoding="utf-8")))

    for kv in args.set:
        k, _, v = kv.partition("=")
        params[k] = int(v) if k in ("seed", "seaLevel") else float(v)

    trials = [dict(params)]
    if args.search:
        key, rng = args.search[0].split("=")
        lo, hi, steps = (float(x) for x in rng.split(":"))
        trials = []
        for i in range(int(steps)):
            p = dict(params)
            p[key] = lo + (hi - lo) * i / max(1, steps - 1)
            trials.append(p)

    best = (1e9, None, None)
    for i, p in enumerate(trials):
        dist, got = evaluate(p, target, args.area, ox, oz)
        m = got
        print(f"[{i}] d={dist:.3f}  relief {m['relief']['span']:5.0f}  "
              f"p50/p90 {m['slope']['p50']:.2f}/{m['slope']['p90']:.2f}  "
              f"flat {m['slope']['flat_pct']:5.1f}%  snow {m['cover']['snow_pct']:5.1f}%")
        if dist < best[0]:
            best = (dist, p, m)

    dist, p, m = best
    print(f"\nbest distance {dist:.3f}")
    write_config(p or params)
    print(f"config written to {CONFIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
