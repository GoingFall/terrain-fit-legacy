#!/usr/bin/env python3
"""Load a datapack in a real 1.20.1 server, headless, and report exactly what the game says.

The client and the vanilla server use the same DataFixerUpper codecs, so a pack that parses here
parses there. This replaces "launch the game and wait" with a ~15 second check, and in generate
mode it also produces terrain that tools/terrain_metrics.py can measure.

Usage:
  python server_probe.py --pack dist/fit-datapack --mode validate
  python server_probe.py --pack dist/fit-datapack --mode generate --area 512

Needs _server/server.jar (vanilla 1.20.1) and Java 17.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

SERVER = Path("_server/server.jar")
WORK = Path("_server/run")

ERROR_MARKERS = (
    "Registry loading errors", "Failed to parse", "Couldn't load", "Caused by",
    "Not a string", "No key", "Unknown", "not a valid",
)


def prepare(pack: Path, overwrite_overworld: bool, biome: str) -> Path:
    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / "world" / "datapacks" / "probe").mkdir(parents=True)
    (WORK / "eula.txt").write_text("eula=true\n", encoding="utf-8")
    (WORK / "server.properties").write_text(
        "level-type=minecraft:normal\n"
        "online-mode=false\n"
        "spawn-protection=0\n"
        "view-distance=10\n"
        "simulation-distance=6\n"
        "max-tick-time=-1\n"
        "sync-chunk-writes=false\n"
        "level-seed=1\n",
        encoding="utf-8")

    dst = WORK / "world" / "datapacks" / "probe"
    for f in pack.rglob("*"):
        if f.is_file():
            target = dst / f.relative_to(pack)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)

    if overwrite_overworld:
        settings = next(dst.glob("data/*/worldgen/noise_settings/*.json"))
        ns, name = settings.parents[2].name, settings.stem
        over = dst / "data" / "minecraft" / "dimension" / "overworld.json"
        over.parent.mkdir(parents=True, exist_ok=True)
        over.write_text(json.dumps({
            "type": "minecraft:overworld",
            "generator": {
                "type": "minecraft:noise",
                "settings": f"{ns}:{name}",
                "biome_source": {"type": "minecraft:fixed", "biome": biome},
            },
        }, indent=2), encoding="utf-8")
        print(f"   overworld overridden with {ns}:{name} ({biome})")
    return WORK


def run(timeout: int, commands: list[str] | None) -> tuple[int, str]:
    cmd = ["java", "-Xmx2G", "-jar", str(SERVER.resolve()), "--nogui"]
    proc = subprocess.Popen(cmd, cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    t0 = time.time()
    log: list[str] = []
    started = sent = False
    quiet_since = time.time()
    last_size = -1
    try:
        for line in iter(proc.stdout.readline, ""):
            log.append(line.rstrip())
            if not started and any("Done (" in l for l in log[-40:]):
                started = True
                quiet_since = time.time()
                print(f"   server up in {time.time() - t0:.1f}s")
                if commands and not sent:
                    for c in commands:
                        proc.stdin.write(c + "\n")
                    proc.stdin.flush()
                    sent = True
                    print(f"   sent {len(commands)} forceload command(s)")
            if started:
                region = WORK / "world" / "region"
                size = sum(f.stat().st_size for f in region.glob("*.mca")) if region.exists() else 0
                if size != last_size:
                    last_size = size
                    quiet_since = time.time()
                elif time.time() - quiet_since > 30:
                    break
            if time.time() - t0 > timeout:
                print("   timeout hit")
                break
    finally:
        try:
            proc.stdin.write("stop\n")
            proc.stdin.flush()
            proc.wait(timeout=120)
        except Exception:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
    return proc.returncode, "\n".join(log)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--mode", choices=["validate", "generate"], default="validate")
    ap.add_argument("--area", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--biome", default="minecraft:snowy_plains")
    args = ap.parse_args()

    if not SERVER.exists():
        print(f"missing {SERVER}")
        return 1

    prepare(Path(args.pack), args.mode == "generate", args.biome)
    commands = None
    if args.mode == "generate":
        n = max(1, args.area // 16 // 2)
        commands = [
            f"forceload add {-n} {-n} -1 -1",
            f"forceload add {-n} 0 -1 {n - 1}",
            f"forceload add 0 {-n} {n - 1} -1",
            f"forceload add 0 0 {n - 1} {n - 1}",
        ]

    code, log = run(args.timeout, commands)
    problems = [l for l in log.splitlines() if any(k in l for k in ERROR_MARKERS)]
    print(f"exit={code}  log lines={len(log.splitlines())}")
    if problems:
        print("\n--- what the game complained about ---")
        for l in problems[:25]:
            print("  " + l[:230])
        return 1

    print("PACK PARSED: no registry or codec errors")
    if args.mode == "generate":
        region = WORK / "world" / "region"
        files = list(region.glob("*.mca")) if region.exists() else []
        size = sum(f.stat().st_size for f in files) // 1024 // 1024
        print(f"generated {len(files)} region file(s), {size} MB -> {region}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
