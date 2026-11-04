package com.terrainfit.fit;

import com.mojang.serialization.Codec;
import net.minecraft.core.registries.Registries;
import net.minecraft.world.level.chunk.ChunkGenerator;
import net.minecraftforge.eventbus.api.IEventBus;
import net.minecraftforge.fml.javafmlmod.FMLJavaModLoadingContext;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.registries.DeferredRegister;
import net.minecraftforge.registries.RegistryObject;

/** Registers the terrain-fit chunk generator type. */
@Mod(FitMod.MODID)
public class FitMod {
    public static final String MODID = "fit";

    public static final DeferredRegister<Codec<? extends ChunkGenerator>> CHUNK_GENERATORS =
            DeferredRegister.create(Registries.CHUNK_GENERATOR, MODID);

    public static final RegistryObject<Codec<? extends ChunkGenerator>> HEIGHTMAP =
            CHUNK_GENERATORS.register("heightmap", () -> HeightmapChunkGenerator.CODEC);

    @SuppressWarnings("removal")
    public FitMod() {
        // 47.4.x still instantiates the mod class with a no-arg constructor
        IEventBus bus = FMLJavaModLoadingContext.get().getModEventBus();
        CHUNK_GENERATORS.register(bus);
    }
}
