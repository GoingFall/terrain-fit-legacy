# terrain-fit mod (Forge 1.20.1)

A minimal `ChunkGenerator` that fills terrain from the fitted height model instead of vanilla's
noise router. The model lives in `FitTerrain.java` as plain arithmetic, and
`tools/fit_model.py` is a Python mirror of it: both were checked to produce identical heights on a
64x64 grid (0 differences over 4096 columns), so a preview and the game cannot drift apart.

Build:

    cd mod && ./gradlew build          # needs JDK 17
    cp build/libs/fit-1.0.0.jar <instance>/mods/

Use: the mod adds the dimension `fit:fit` (`/execute in fit:fit run tp @s 0 150 0`) with a fixed
snowy_plains biome, sea level 74 and a snow line at y=193.

Status: builds, loads in a Forge server, registers `fit:heightmap`, and generates chunks. First
measurement of the generated area: relief 54 (the model predicts 85 in that window), min 102 vs 101,
max 156 vs 186 -- so the fill is faithful at the low end but truncates the peaks somewhere.
