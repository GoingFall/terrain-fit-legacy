package com.terrainfit.fit;

import com.mojang.serialization.Codec;
import com.mojang.serialization.codecs.RecordCodecBuilder;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.WorldGenRegion;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.util.RandomSource;
import net.minecraft.world.level.LevelHeightAccessor;
import net.minecraft.world.level.NoiseColumn;
import net.minecraft.world.level.StructureManager;
import net.minecraft.world.level.biome.BiomeManager;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.ChunkGenerator;
import net.minecraft.world.level.chunk.LevelChunkSection;
import net.minecraft.world.level.levelgen.GenerationStep;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.blending.Blender;

/**
 * Minimal viable generator: fill every column from the fitted height model.
 *
 * No vanilla noise, no density functions — the same arithmetic the preview renders, so what the
 * datapack could not reproduce is reproduced here exactly.
 */
public class HeightmapChunkGenerator extends ChunkGenerator {
    public static final Codec<HeightmapChunkGenerator> CODEC = RecordCodecBuilder.create(inst ->
            inst.group(
                    net.minecraft.world.level.biome.BiomeSource.CODEC
                            .fieldOf("biome_source").forGetter(g -> g.biomeSource),
                    Codec.INT.fieldOf("sea_level").orElse(FitTerrain.SEA_LEVEL)
                            .forGetter(g -> g.seaLevel),
                    Codec.DOUBLE.fieldOf("snow_line").orElse(193.0)
                            .forGetter(g -> g.snowLine)
            ).apply(inst, HeightmapChunkGenerator::new));

    private final int seaLevel;
    private final double snowLine;

    public HeightmapChunkGenerator(net.minecraft.world.level.biome.BiomeSource biomeSource,
                                   int seaLevel, double snowLine) {
        super(biomeSource);
        this.seaLevel = seaLevel;
        this.snowLine = snowLine;
    }

    @Override
    protected Codec<? extends ChunkGenerator> codec() {
        return CODEC;
    }

    @Override
    public int getGenDepth() {
        return 384;
    }

    @Override
    public int getSeaLevel() {
        return seaLevel;
    }

    @Override
    public int getMinY() {
        return -64;
    }

    @Override
    public int getBaseHeight(int x, int z, Heightmap.Types type, LevelHeightAccessor level,
                             RandomState random) {
        double h = FitTerrain.height(x, z);
        int top = (int) Math.floor(h);
        int sea = Math.max(top, seaLevel);
        if (type == Heightmap.Types.OCEAN_FLOOR_WG) {
            return Math.max(top, getMinY());
        }
        return Math.max(sea, getMinY());
    }

    @Override
    public NoiseColumn getBaseColumn(int x, int z, LevelHeightAccessor level, RandomState random) {
        int height = getBaseHeight(x, z, Heightmap.Types.WORLD_SURFACE_WG, level, random);
        BlockState[] states = new BlockState[level.getHeight()];
        for (int i = 0; i < states.length; i++) {
            int y = getMinY() + i;
            states[i] = y <= height ? Blocks.STONE.defaultBlockState() : Blocks.AIR.defaultBlockState();
        }
        return new NoiseColumn(getMinY(), states);
    }

    @Override
    public CompletableFuture<ChunkAccess> fillFromNoise(Executor executor, Blender blender,
                                                        RandomState random,
                                                        StructureManager structureManager,
                                                        ChunkAccess chunk) {
        BlockState stone = Blocks.STONE.defaultBlockState();
        BlockState dirt = Blocks.DIRT.defaultBlockState();
        BlockState grass = Blocks.GRASS_BLOCK.defaultBlockState();
        BlockState snow = Blocks.SNOW.defaultBlockState();
        BlockState water = Blocks.WATER.defaultBlockState();

        Heightmap surface = chunk.getOrCreateHeightmapUnprimed(Heightmap.Types.WORLD_SURFACE_WG);
        Heightmap floor = chunk.getOrCreateHeightmapUnprimed(Heightmap.Types.OCEAN_FLOOR_WG);
        int minY = getMinY();
        int maxY = minY + getGenDepth();
        int x0 = chunk.getPos().getMinBlockX();
        int z0 = chunk.getPos().getMinBlockZ();

        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                int wx = x0 + lx, wz = z0 + lz;
                double hf = FitTerrain.height(wx, wz);
                int top = (int) Math.floor(hf);
                for (int y = minY; y < maxY; y++) {
                    BlockState state = null;
                    if (y <= top) {
                        int depth = top - y;
                        if (depth == 0) {
                            state = (top > snowLine) ? snow : grass;
                        } else if (depth < 4) {
                            state = (top > snowLine) ? stone : dirt;
                        } else {
                            state = stone;
                        }
                    } else if (y <= seaLevel) {
                        state = water;
                    }
                    if (state != null) {
                        set(chunk, lx, y, lz, state);
                        floor.update(lx, y, lz, state);
                        surface.update(lx, y, lz, state);
                    }
                }
            }
        }
        return CompletableFuture.completedFuture(chunk);
    }

    private static void set(ChunkAccess chunk, int lx, int y, int lz, BlockState state) {
        LevelChunkSection sec = chunk.getSection(chunk.getSectionIndex(y));
        sec.setBlockState(lx, y & 15, lz, state);
    }

    @Override
    public void buildSurface(WorldGenRegion region, StructureManager structureManager,
                             RandomState random, ChunkAccess chunk) {
        // surfaces are written in fillFromNoise
    }

    @Override
    public void applyCarvers(WorldGenRegion region, long seed, RandomState random,
                             BiomeManager biomeManager, StructureManager structureManager,
                             ChunkAccess chunk, GenerationStep.Carving step) {
        // no caves
    }

    @Override
    public void spawnOriginalMobs(WorldGenRegion region) {
        // no extra mob spawning
    }

    @Override
    public void addDebugScreenInfo(java.util.List<String> lines, RandomState random,
                                      BlockPos pos) {
        lines.add("terrain-fit: model v2");
    }
}
