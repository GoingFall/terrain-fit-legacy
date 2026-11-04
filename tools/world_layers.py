#!/usr/bin/env python3
"""Split a world's columns into three separate layers: ground / water / canopy.

Vectorised: each 16^3 section is decoded with numpy bit-ops (one pass over 4096 indices),
never a per-block Python loop. That is what makes a whole region take seconds instead of
minutes.

A single "top non-air block" height field conflates three different things:
  * ground  -- the terrain surface (stone/dirt/grass/sand...)
  * water   -- a fluid surface, which can sit above or below the ground
  * canopy  -- leaves/logs/plants, which belong *on* the ground, not *as* the ground
"""
from __future__ import annotations

import os
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from world_heightmap import iter_chunks, read_nbt  # noqa: E402

# block id -> colour table, produced by tools/block_palette.py from the real textures
_PAL = Path(__file__).parent.parent / "_palette" / "palette.json"
PALETTE_NAMES: list[str] = []
NAME2ID: dict[str, int] = {}
if _PAL.exists():
    _data = json.loads(_PAL.read_text(encoding="utf-8"))
    PALETTE_NAMES = sorted(_data["colors"].keys())
    NAME2ID = {n: i for i, n in enumerate(PALETTE_NAMES)}
UNKNOWN_ID = 255

GROUND, WATER, CANOPY, AIR, SNOW, ICE = 0, 1, 2, 3, 4, 5

WATER_EXACT = {"water", "kelp", "kelp_plant", "seagrass", "tall_seagrass",
               "bubble_column", "sea_pickle"}
ICE_EXACT = {"ice", "packed_ice", "blue_ice", "frosted_ice"}
SNOW_EXACT = {"snow", "snow_block", "powder_snow"}
SNOW_IDS = np.array([NAME2ID[n] for n in sorted(SNOW_EXACT) if n in NAME2ID], dtype=np.uint8)
# ground blocks whose names contain a plant word and must NOT be treated as vegetation
GROUND_EXACT = {"grass_block", "dirt", "coarse_dirt", "rooted_dirt", "podzol", "mycelium",
                "dirt_path", "farmland", "mud", "packed_mud", "moss_block", "moss_carpet",
                "clay", "gravel", "sand", "red_sand", "soul_sand", "soul_soil",
                "short_grass", "tall_grass"}          # last two are removed right below
GROUND_EXACT -= {"short_grass", "tall_grass"}
PLANT_EXACT = {"grass", "short_grass", "tall_grass", "fern", "large_fern", "dead_bush", "bush",
               "sugar_cane", "sweet_berry_bush", "cactus", "bamboo", "bamboo_sapling",
               "lily_pad", "brown_mushroom", "red_mushroom", "vine", "glow_lichen",
               "wheat", "carrots", "potatoes", "beetroots", "melon_stem",
               "pumpkin_stem", "torchflower", "pitcher_plant", "pink_petals",
               "sunflower", "lilac", "rose_bush", "peony", "azalea",
               "flowering_azalea", "big_dripleaf", "small_dripleaf", "twisting_vines",
               "weeping_vines", "cave_vines", "cave_vines_plant", "chorus_plant"}
DECOR_EXACT = {"lever", "torch", "wall_torch", "soul_torch", "soul_wall_torch", "ladder",
               "chain", "lantern", "soul_lantern", "flower_pot", "cauldron", "campfire",
               "soul_campfire", "sign", "wall_sign", "chest", "barrel", "hopper", "anvil",
               "grindstone", "stonecutter", "composter", "bell", "rail", "powered_rail"}
FLOWER_EXACT = {"poppy", "dandelion", "blue_orchid", "allium", "azure_bluet",
                "oxeye_daisy", "cornflower", "lily_of_the_valley", "wither_rose"}
FLOWER_SUFFIX = ("_tulip", "_daisy", "_poppy", "_orchid", "_allium", "_bluet",
                 "_lily", "_flower", "_sapling", "_coral", "_coral_fan", "_coral_wall_fan")


def kind_of(name: str) -> int:
    n = name.split(":", 1)[-1]
    if n in ("air", "cave_air", "void_air"):
        return AIR
    if n in GROUND_EXACT:
        return GROUND
    if n in WATER_EXACT:
        return WATER
    if n in ICE_EXACT:
        return ICE
    if n in SNOW_EXACT:
        return SNOW
    # vegetation: suffix rules are safe, "grass_block" is already handled above
    if (n.endswith("_fence") or n.endswith("_fence_gate") or n.endswith("_wall")
            or n.endswith("_leaves") or n.endswith("_log") or n.endswith("_wood")
            or n.endswith("_stem") or n.endswith("_hyphae") or n.endswith("_wart_block")):
        return CANOPY
    if n in PLANT_EXACT or n in FLOWER_EXACT or n in DECOR_EXACT or n.endswith(FLOWER_SUFFIX):
        return CANOPY
    # water/ice/snow families by suffix
    if n.endswith("_ice"):
        return ICE
    if n.endswith("_snow"):
        return SNOW
    return GROUND


def chunk_layers(root: dict):
    """-> (int16 (6,16,16) top-y per kind, uint8 (16,16) surface block id)."""
    out = np.full((6, 16, 16), -70, dtype=np.int16)
    best_y = np.full((16, 16), -999, dtype=np.int16)
    best_sid = np.full((16, 16), UNKNOWN_ID, dtype=np.uint8)
    for sec in root.get("sections") or []:
        bs = sec.get("block_states")
        if not bs:
            continue
        palette = [p.get("Name", "minecraft:air") for p in bs.get("palette") or []]
        if not palette:
            continue
        kinds = np.fromiter((kind_of(n) for n in palette), dtype=np.int8, count=len(palette))
        if (kinds == AIR).all():
            continue
        y0 = int(sec.get("Y", 0)) * 16

        data = bs.get("data")
        if data:
            bits = max(4, (len(palette) - 1).bit_length())
            per_long = 64 // bits
            words = np.asarray([int(v) & 0xFFFFFFFFFFFFFFFF for v in data], dtype=np.uint64)
            shifts = (np.arange(per_long, dtype=np.uint64) * np.uint64(bits))
            idx = ((words[:, None] >> shifts[None, :]) & np.uint64((1 << bits) - 1))
            idx = idx.ravel()[:4096].astype(np.intp)
        else:
            idx = np.zeros(4096, dtype=np.intp)

        block_kind = kinds[idx].astype(np.int8).reshape(16, 16, 16)      # (y, z, x)
        ys = (np.arange(16, dtype=np.int16) + np.int16(y0))[:, None, None]

        # surface id: the topmost non-air block in each column of this section
        if PALETTE_NAMES:
            sid_arr = np.array([NAME2ID.get(n.split(":", 1)[-1], UNKNOWN_ID) for n in palette],
                               dtype=np.uint8)
            pid = sid_arr[idx].reshape(16, 16, 16)
            nonair = (block_kind != AIR) & (block_kind != CANOPY) & (block_kind != WATER)
            yfill = np.where(nonair, ys, np.int16(-999))
            iy = yfill.argmax(axis=0)                                    # (z, x)
            ytop = yfill.max(axis=0)
            sid_top = np.take_along_axis(pid, iy[None, :, :], axis=0)[0]
            better = ytop > best_y
            best_sid = np.where(better, sid_top, best_sid).astype(np.uint8)
            best_y = np.where(better, ytop, best_y).astype(np.int16)

        for k in (GROUND, WATER, CANOPY, SNOW, ICE):
            mask = block_kind == k
            if not mask.any():
                continue
            top = np.where(mask, ys, np.int16(-70)).max(axis=0)          # (z, x)
            np.maximum(out[k], top, out=out[k])
    return out, best_sid


def _decode_region(job):
    """Worker: decode one region file into a (6, 512, 512) int16 buffer.

    Module-level so it can be pickled by ProcessPoolExecutor under spawn (Windows).
    """
    path_str, _rx, _rz = job
    buf = np.full((6, 512, 512), -70, dtype=np.int16)
    sid = np.full((512, 512), UNKNOWN_ID, dtype=np.uint8)
    n = 0
    for slot, raw in iter_chunks(Path(path_str)):
        try:
            root = read_nbt(raw)
        except Exception:
            continue
        cx = slot & 31
        cz = slot >> 5
        lay, sid_c = chunk_layers(root)
        buf[:, cz * 16 : cz * 16 + 16, cx * 16 : cx * 16 + 16] = lay
        sid[cz * 16 : cz * 16 + 16, cx * 16 : cx * 16 + 16] = sid_c
        n += 1
    return buf, sid, n


def collect_layers(world: Path, only_region: str | None = None, verbose: bool = False,
                   workers: int | None = None, fill_missing: bool = True):
    """-> (ground, water, canopy, x0, z0) as float32 arrays."""
    mcae = []
    for mca in sorted(world.glob("region/*.mca")):
        if only_region and mca.stem != only_region:
            continue
        try:
            rx, rz = (int(v) for v in mca.stem.split(".")[1:3])
        except Exception:
            continue
        mcae.append((mca, rx, rz))
    if not mcae:
        raise SystemExit("no region files")

    rx0 = min(r for _, r, _ in mcae)
    rz0 = min(z for _, _, z in mcae)
    rx1 = max(r for _, r, _ in mcae) + 1
    rz1 = max(z for _, _, z in mcae) + 1
    H, W = (rz1 - rz0) * 512, (rx1 - rx0) * 512
    buf = np.full((6, H, W), -70, dtype=np.int16)
    sid_buf = np.full((H, W), UNKNOWN_ID, dtype=np.uint8)

    t0 = time.time()
    chunks = 0
    if workers is None:
        workers = min(8, os.cpu_count() or 1, len(mcae))
    if workers > 1 and len(mcae) > 1:
        jobs = [(str(m), r, z) for m, r, z in mcae]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for (mca, rx, rz), (r_buf, r_sid, n) in zip(mcae, ex.map(_decode_region, jobs)):
                ox, oz = (rx - rx0) * 512, (rz - rz0) * 512
                buf[:, oz : oz + 512, ox : ox + 512] = r_buf
                sid_buf[oz : oz + 512, ox : ox + 512] = r_sid
                chunks += n
        if verbose:
            print(f"   decoded {chunks} chunks in {time.time() - t0:.1f}s "
                  f"({workers} workers, {len(mcae)} regions)", file=sys.stderr)
    else:
        for mca, rx, rz in mcae:
            for slot, raw in iter_chunks(mca):
                try:
                    root = read_nbt(raw)
                except Exception:
                    continue
                lay, sid_c = chunk_layers(root)
                cx = (rx - rx0) * 32 + (slot & 31)
                cz = (rz - rz0) * 32 + (slot >> 5)
                buf[:, cz * 16 : cz * 16 + 16, cx * 16 : cx * 16 + 16] = lay
                sid_buf[cz * 16 : cz * 16 + 16, cx * 16 : cx * 16 + 16] = sid_c
                chunks += 1
        if verbose:
            print(f"   decoded {chunks} chunks in {time.time() - t0:.1f}s (serial)",
                  file=sys.stderr)

    def layer(k):
        a = buf[k].astype(np.float32)
        a[a <= -70] = -1e4
        return a

    g_raw, w_raw, c_raw = layer(GROUND), layer(WATER), layer(CANOPY)
    s_raw, i_raw = layer(SNOW), layer(ICE)

    # snow sits *on* the ground; ice only counts as "water" when it actually caps water
    # (a glacier / ice sheet on land must become terrain, otherwise it paints a wall)
    ground = np.maximum(g_raw, s_raw)
    ice_on_water = (i_raw > w_raw) & (i_raw <= w_raw + 3.0)
    ice_on_land = (i_raw > w_raw) & ~ice_on_water
    ground = np.maximum(ground, np.where(ice_on_land, i_raw, -1e4))
    water = np.where(ice_on_water, i_raw, w_raw)
    canopy = c_raw

    flags = np.zeros(ground.shape, dtype=np.uint8)
    flags[s_raw > g_raw] += 1          # bit0: snow layer above the ground
    flags[ice_on_water] += 2           # bit1: ice (only where it caps water)
    flags[c_raw > ground] += 4         # bit2: canopy

    if fill_missing:
        missing = ground <= -70
        if missing.any():
            ground[missing] = float(np.median(ground[~missing]))
            water[missing] = -1e4
    for a in (water, canopy):
        a[a <= -70] = -1e4             # absent -> far below ground
    water[ground <= -63] = -1e4        # no terrain -> no water surface (map borders / holes)
    snow_surface = np.isin(sid_buf, SNOW_IDS)
    flags[snow_surface] |= 1
    return ground, water, canopy, flags, sid_buf, rx0 * 512, rz0 * 512


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    wk = 1 if "--serial" in sys.argv else None
    world = Path(args[0] if args else ".")
    region = args[1] if len(args) > 1 else None
    t = time.time()
    g, w, c, f, x0, z0 = collect_layers(world, region, verbose=True, workers=wk)
    dt = time.time() - t
    px = g.size
    print(f"ground {g.min():.0f}..{g.max():.0f}   water cols={(w > g + 0.5).mean():.1%}"
          f"   snow={(f & 1).mean():.1%}  ice={(f & 2).mean():.1%}  canopy={(f & 4).mean():.1%}")
    print(f"total {dt:.1f}s for {px:,} columns  ({px / dt / 1e3:.0f}k cols/s)")
