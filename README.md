# terrain-fit

**Tooling to fit procedural Minecraft terrain generation to hand-authored reference maps.**

Read a world save → reconstruct its heightfield, water, canopy and surface-block ids → look at it
in a self-contained 3D viewer → measure its terrain statistics → tune a noise-based generator until
it produces the same *kind* of landscape.

This is the first public snapshot: the **viewer and measurement pipeline are usable**; the
parameter-fitting loop is prototyped but not solved yet (see [Roadmap](#roadmap)).

> 一个 Minecraft 地形工具链：读取存档 → 重建高度场/水面/树冠/地表方块 → 在单文件 3D 查看器里查看
> → 量化地形统计 → 调整程序化生成器参数去拟合参考地图。当前版本：**查看器与度量管线可用**，
> 参数拟合只做了探索（见路线图）。

---

## Why

Scenic Minecraft maps (the ones in study/comfy videos) are usually **hand-painted heightfields**
(WorldPainter / WorldMachine), not generated. You cannot reproduce them with a noise generator —
but you *can* measure what makes them look the way they do, and iterate a generator toward those
measurements. This repo is the measuring + viewing half of that loop.

## What works today

| Capability | Where |
|---|---|
| Decode a world's region files into ground / water / canopy layers, with surface block ids | `tools/world_layers.py` |
| Height map PNG + slope statistics for any save or region | `tools/world_heightmap.py` |
| List which block ids exist in a world (find out whether a map uses modded blocks) | `tools/scan_world_blocks.py` |
| Render a save as a still image (same framing/lighting as the viewer) | `tools/world_preview.py` |
| **Interactive single-file 3D viewer** (WebGL1, no server, no dependencies) | `tools/world_viewer.py` |
| Real per-block colours from vanilla 1.20.1 textures (JEI-style: sample sprite + biome tint) | `tools/block_palette.py` |
| **Terrain metrics panel** (relief, slope percentiles, water/snow cover, ridge density) + JSON export | `tools/terrain_metrics.py` |
| **Metric distance + batch table** (compare maps numerically, no images) | `tools/terrain_metrics.py --compare`, `tools/batch_metrics.py` |
| Single-target parameter search (minimises the same metric distance) | `tools/fit_metrics.py --target metrics.json --random N` |
| Procedural terrain preview (parameters → picture, no game needed) | `tools/terrain_preview.py` |

## Quick start

```bash
# 1) any Minecraft save: build an interactive viewer (single .html, opens in a browser)
python tools/world_viewer.py "<path-to-save>" --out preview/viewer.html

#    or one region only (faster, more detail per pixel)
python tools/world_viewer.py "<path-to-save>" --region r.0.0 --out preview/region.html

# 2) measure a map
python tools/terrain_metrics.py "<path-to-save>" --region r.0.0 --json metrics.json
python tools/world_heightmap.py "<path-to-save>" -o heightmap.png --stats
python tools/world_layers.py "<path-to-save>" r.0.0        # layers + snow/ice/canopy coverage

# 3) build the colour table from vanilla textures (fetched once, cached)
python tools/block_palette.py

# 4) compare a procedural build against the target's slope distribution
python tools/fit_metrics.py ridge_amp=150 slope_limit=3
```

No mods, no game launch, no server: everything reads region files directly.

## Viewer

Open the generated `.html`:

| Input | Action |
|---|---|
| drag | orbit (yaw and pitch at the same angular rate) |
| wheel | zoom (fov) |
| `W` `A` `S` `D` | move along the current heading (pitch is ignored) |
| `space` / `shift` | up / down |
| `F` | toggle free flight (disables terrain collision) |
| `R` | jump to the auto-picked "most open" vantage point |
| `[` `]` or the slider | movement speed in blocks/second |

Debug parameters: `?x= &y= &z= &yaw= &pitch=` (fixed camera), `?nowater=1`, `?dbg=1` (hit-distance
heat map), `?fade=1` (soften the edge of missing data).

### How it renders

* heightfield raymarching in a fragment shader, adaptive "safe stepping" (step length derived from
  the vertical clearance, so grazing angles do not skip ridges)
* one RGBA texture carries: 14-bit height + 2-bit surface class (R,G), water surface (B), and an
  always-opaque alpha (storing data in alpha breaks under canvas premultiplication)
* a second texture carries the per-column surface block id, a third is a 256×2 palette
  (top faces / side faces), a fourth is a 256×256 texture atlas used only within 240 blocks
* only the largest connected island of valid chunks is loaded — map authors ring their work with
  empty chunks, and anything outside that ring is not part of the map

## Metrics (the evaluator's yardstick)

`tools/terrain_metrics.py` computes, from the same layers the viewer uses:

| Metric | Meaning |
|---|---|
| `relief.{min,max,span,mean}` | height range of the valid columns |
| `slope.{p50,p90,p99,max,gt2_pct,gt4_pct,flat_pct}` | gradient magnitude distribution (blocks per block) |
| `cover.{water,snow,ice,canopy}_pct` | surface composition |
| `snow_line` | median height of snow-covered columns |
| `ridge_per_km2` | local maxima of the surface *smoothed* with radius 5 (counting raw 1-block maxima is meaningless) |

The viewer shows the same summary in its HUD, and `--metrics-json` writes it out.

### Comparing maps numerically

```bash
# distance between two metric files (0 = identical)
python tools/terrain_metrics.py x --compare reference.json generated.json
# -> distance = 1.3528, plus per-metric residuals sorted by size

# a whole folder of saves -> one table (JSON + TSV) for batch runs
python tools/batch_metrics.py "<saves-dir>" --region r.0.0 --out table.json --tsv table.tsv
```

The distance is a weighted mean of `|ln(a/b)|` over the metrics, so it is scale-relative and every
metric contributes comparably. The residuals say *which* dimension is off (water cover? cliff
share? ridge density?), which is what a parameter search needs to steer on. Images are for humans;
this table is what a program should compare.

Current weights: `slope.p99` 1.5, `relief.span` / `slope.p50` / `slope.p90` / `slope.gt2_pct` /
`slope.gt4_pct` / `cover.water_pct` / `ridge_per_km2` 1.0, `cover.snow_pct` / `slope.flat_pct` 0.5.

## Findings worth keeping

* **Hand-made maps are smooth, procedural ones are spiky.** Measured on a reference map: slope
  p50 0.71 / p90 1.50 / p99 2.50, max 9. A naive ridged-noise generator: p50 0.36 / p90 2.49 /
  p99 11.05, max 23. The generator is *flatter on average yet far spikier*.
* **Single-scale slope distribution is not enough** to describe a landscape. Fitting it made the
  numbers match while the picture became bland — the reference has *medium slopes with high feature
  density*, and blurring/slope-limiting trades one for the other. Next step is a multi-scale metric
  (gradient at blur radii 1/4/16 plus ridge density).
* Map authors often **substitute blocks by colour** (green terracotta instead of grass). A palette
  built from real textures makes such maps look right, but it also means the surface block ids are
  not a reliable guide to "what this landscape would be made of" in a generated world.

## Layout

```
tools/        the pipeline (see table above)
reports/      written-up findings: mod identification, terrain mod options, fitting notes
preview/      small reference images (viewers themselves are gitignored, rebuild them)
```

## Requirements

Python 3.11+ with `numpy` and `Pillow`. Nothing else. Multi-process region decoding uses the
standard library; large maps scale with CPU cores (`--workers N`).

## First result: a fitted generator that ships as a datapack

`tools/export_datapack.py` turns a fitted parameter set into a 1.20.1 datapack with its own
dimension, so the generated terrain can be walked in and measured:

```bash
python tools/fit_metrics.py --model v2 --target metrics.json --random 50 --refine 3 --best best.json
python tools/export_datapack.py --params best.json --out dist/fit-datapack
# copy dist/fit-datapack (or the zip) into <world>/datapacks/, /reload, then:
#   /execute in fit:fit run tp @s 0 140 0
python tools/terrain_metrics.py "<that world>"        # measure what the game actually generated
```

Fitted against the reference region (`r.0.0` of the map used during development), distance 0.089:

| metric | target | fit |
|---|---|---|
| relief span | 238 | 193 |
| slope p50/p90/p99 | 0.71/1.50/2.50 | 0.71/1.39/2.57 |
| water / snow | 0.59% / 63.9% | 0.55% / 61.4% |
| ridge per km2 | 278 | 278 |
| spectral bands >=512 / 256-512 / 128-256 / 64-128 | 24.2 / 35.9 / 37.7 / 1.7 % | 23.8 / 34.1 / 39.8 / 1.6 % |

### Validating and generating without launching the game

`tools/server_probe.py` runs a real vanilla 1.20.1 server headless with the pack installed. The
client and the server share the same DataFixerUpper codecs, so a pack that parses there parses on
your machine — and in generate mode it can pre-generate a fixed area that `terrain_metrics.py` then
measures:

```bash
python tools/server_probe.py --pack dist/fit-datapack --mode validate      # ~15 s
python tools/server_probe.py --pack dist/fit-datapack --mode generate --area 512
python tools/terrain_metrics.py _server/run/world                           # what the game made
```

Requires `_server/server.jar` (vanilla 1.20.1) and Java 17. Every schema mistake that used to cost a
game launch now costs 15 seconds and prints the exact codec error.

### Checking the export without launching the game

`tools/df_eval.py` evaluates the exported density function itself: the surface is where the density
crosses zero, and every wrapper the router adds (squeeze, min with the cave noise) is monotonic and
zero-preserving, so the height field can be solved offline and compared with the target.

```bash
python tools/df_eval.py --datapack dist/fit-datapack --target metrics.json
```

That turned three game-length iterations into seconds, and caught: terrain clipped at y=320, an
absolute height range that put the whole world above sea level, and a missing snow surface rule.

What the datapack cannot express: the **directional valley axis** (a density function has no rotated
frame) and custom noise wavelengths — the export maps the bands onto vanilla's own climate fields
(`continents`, `erosion`, `ridges_folded`, `temperature`, `vegetation`), so it reproduces the shapes
and amplitudes of the model but not its exact frequencies. Treat the export as the pipeline proof,
not as a byte-identical clone of the preview.

Verified so far: JSON structure, node-type and reference validity against the 1.20.1 vanilla data.
Not verified: an actual in-game load (needs a Minecraft client).

## Roadmap

1. **A generator model that can represent the target, then fit it.** A random search over the
   current single-ring generator reached distance 0.40 against a snow-alpine reference, but the
   render is still wrong (relief 130 vs 238, cliff share 0 vs 0.14): the model produces plateaus
   with spikes, so more search will not help. Next: add a macroscale relief term and a sparse
   cliff term, then re-fit.
2. **Datapack export**: turn the fitted parameters into a `noise_settings` JSON that runs in-game
   (1.20.1), then verify the real terrain against the preview with `world_heightmap.py`.
3. **Temporal denoising (TAA)** in the viewer to remove the residual grazing-angle noise.
4. Volumetric (voxel) mode, so terrain does not look hollow from below.

## Notes

* The reference map used during development is not redistributed here; download links and analysis
  live in `reports/`.
* No vanilla assets are committed either: `tools/block_palette.py` fetches the 16×16 textures once
  and caches them locally.
