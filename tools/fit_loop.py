#!/usr/bin/env python3
"""One iteration of the fitting loop, entirely offline: export -> validate -> generate -> measure.

This is the loop that used to require launching the game twice per change:

  1. export a datapack from a parameter set (tools/export_datapack.py)
  2. parse it in a real 1.20.1 server (tools/server_probe.py --mode validate)
  3. generate a fixed 512-block area with it (--mode generate)
  4. measure what the game produced (tools/terrain_metrics.py) and print the distance

Usage:
  python fit_loop.py --params preview/best_v7.json --target preview/m_r00.json --offset 39
  python fit_loop.py --params best.json --target m.json --tweak continent_gain=1.4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from terrain_metrics import compare  # noqa: E402

ROOT = Path(__file__).parent.parent
PY = sys.executable


def sh(cmd: list[str], **kw) -> int:
    print("$ " + " ".join(str(c) for c in cmd))
    return subprocess.call(cmd, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", default="dist/fit-datapack")
    ap.add_argument("--area", type=int, default=128)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--biome", default="minecraft:snowy_plains")
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()

    print("== 1/3 export")
    rc = sh([PY, "tools/export_datapack.py", "--params", args.params, "--out", args.out,
             "--target", args.target, "--offset", str(args.offset), "--biome", args.biome])
    if rc:
        return rc

    if args.skip_generate:
        print("\n== 2/2 validate in a real server")
        return sh([PY, "tools/server_probe.py", "--pack", args.out, "--mode", "validate"])

    print("\n== 2/3 generate (a bad pack fails at boot, so this also validates)")
    rc = sh([PY, "tools/server_probe.py", "--pack", args.out, "--mode", "generate",
             "--area", str(args.area), "--biome", args.biome])
    if rc:
        print("PACK REJECTED - fix the pack")
        return rc

    print("\n== 3/3 measure")
    world = Path("_server/run/world")
    subprocess.call([PY, "tools/terrain_metrics.py", str(world), "--workers", "1",
                     "--json", "ingame.json"])
    got = json.loads(Path("ingame.json").read_text(encoding="utf-8"))
    tgt = json.loads(Path(args.target).read_text(encoding="utf-8"))
    res = compare(tgt, got)
    print(f"\ndistance to target: {res['distance']:.3f}")
    for k, v in list(res["residuals"].items())[:8]:
        print(f"  {k:32} target {v['a']:>9.2f}  game {v['b']:>9.2f}  log {v['log_ratio']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
