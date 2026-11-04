package com.terrainfit.fit;

/**
 * The fitted terrain model, in plain Java with no Minecraft dependency so it can be unit-tested
 * against the Python implementation (tools/fit_model.py) and produce identical numbers.
 *
 * noise2: deterministic 2D value noise, bilinear + smoothstep, seeded by a 32-bit hash. The Python
 * side implements exactly the same arithmetic, so a preview and the game agree per column.
 */
public final class FitTerrain {
    public static final int SEED = 1337;

    // fitted parameters (best_v7, model v2)
    public static final double BASE_LEVEL = 110.0;
    public static final double RELIEF_MACRO = 180.0;
    public static final double CONTINENT_FREQ = 1.9;
    public static final double CONTINENT_GAMMA = 1.7;
    public static final double MASSIF_FREQ = 4.0;
    public static final double MASSIF_AMP = 40.0;
    public static final double MOUNTAIN_FREQ = 2.3;
    public static final double MOUNTAIN_THRESHOLD = 0.20;
    public static final double MOUNTAIN_SOFTNESS = 0.30;
    public static final double RIDGE_FREQ = 8.5;
    public static final double RIDGE_AMP = 55.0;
    public static final double RIDGE_SHARP = 1.5;
    public static final double HILL_FREQ = 11.0;
    public static final double HILL_AMP = 16.0;
    public static final double SPIKE_FREQ = 21.0;
    public static final double SPIKE_AMP = 26.0;
    public static final double SPIKE_THRESHOLD = 0.45;
    public static final double EROSION_FREQ = 3.2;
    public static final double EROSION_FLOOR = 0.30;
    public static final double RIVER_FREQ = 2.6;
    public static final double RIVER_DEPTH = 16.0;
    public static final double RIVER_WIDTH = 0.09;
    public static final int SEA_LEVEL = 74;
    public static final double SNOW_QUANTILE = 0.40;

    private FitTerrain() {}

    // ---------------------------------------------------------------- hashing + noise
    private static int hash(int x, int z, int seed) {
        int h = x * 374761393 + z * 668265263 + seed * 1442695041;
        h = (h ^ (h >>> 13)) * 1274126177;
        return h ^ (h >>> 16);
    }

    private static double rand01(int x, int z, int seed) {
        return (hash(x, z, seed) & 0x7fffffff) / (double) 0x7fffffff;
    }

    private static double fade(double t) {
        return t * t * (3.0 - 2.0 * t);
    }

    /** value noise in [-1, 1]; wavelength = 1/freq blocks. */
    public static double noise2(double x, double z, double freq, int seed, int octaves) {
        double sum = 0.0, amp = 1.0, norm = 0.0, f = freq;
        for (int o = 0; o < octaves; o++) {
            double px = x * f, pz = z * f;
            int x0 = (int) Math.floor(px), z0 = (int) Math.floor(pz);
            double tx = fade(px - x0), tz = fade(pz - z0);
            double v00 = rand01(x0, z0, seed + o * 7919);
            double v10 = rand01(x0 + 1, z0, seed + o * 7919);
            double v01 = rand01(x0, z0 + 1, seed + o * 7919);
            double v11 = rand01(x0 + 1, z0 + 1, seed + o * 7919);
            double a = v00 + (v10 - v00) * tx;
            double b = v01 + (v11 - v01) * tx;
            sum += amp * ((a + (b - a) * tz) * 2.0 - 1.0);
            norm += amp;
            amp *= 0.5;
            f *= 2.0;
        }
        return sum / norm;
    }

    // ---------------------------------------------------------------- model
    private static double smooth(double t) {
        t = Math.max(0.0, Math.min(1.0, t));
        return t * t * (3.0 - 2.0 * t);
    }

    private static double folded(double n) {
        return 1.0 - Math.abs(2.0 * Math.abs(n) - 1.0);
    }

    /** surface height in blocks for world column (x, z). */
    public static double height(double x, double z) {
        double c = noise2(x, z, CONTINENT_FREQ / 512.0, SEED, 2);
        double t = Math.pow(Math.max(0.0, Math.min(1.0, c * 0.5 + 0.5)), CONTINENT_GAMMA);
        double h = BASE_LEVEL + RELIEF_MACRO * t;

        double m = noise2(x, z, MOUNTAIN_FREQ / 512.0, SEED + 11, 2);
        double mountain = smooth((m - MOUNTAIN_THRESHOLD) / MOUNTAIN_SOFTNESS);

        double e = noise2(x, z, EROSION_FREQ / 512.0, SEED + 22, 2) * 0.5 + 0.5;
        double erosion = EROSION_FLOOR + (1.0 - EROSION_FLOOR) * smooth(e);

        double ridge = 0.0, amp = 1.0, f = RIDGE_FREQ / 512.0;
        for (int o = 0; o < 2; o++) {
            ridge += folded(noise2(x, z, f, SEED + 33, 1)) * amp;
            amp *= 0.5;
            f *= 2.0;
        }
        ridge = Math.pow(ridge / 1.5, RIDGE_SHARP);

        double massif = noise2(x, z, MASSIF_FREQ / 512.0, SEED + 44, 2);
        h += massif * mountain * MASSIF_AMP * erosion;
        h += mountain * ridge * RIDGE_AMP * erosion;

        double hill = smooth(noise2(x, z, HILL_FREQ / 512.0, SEED + 55, 2) * 0.5 + 0.5);
        h += (hill * 2.0 - 1.0) * HILL_AMP * erosion;

        double spike = smooth((noise2(x, z, SPIKE_FREQ / 512.0, SEED + 66, 2) * 0.5 + 0.5
                - SPIKE_THRESHOLD) / (1.0 - SPIKE_THRESHOLD));
        h += hill * spike * SPIKE_AMP;

        double rv = noise2(x, z, RIVER_FREQ / 512.0, SEED + 77, 2);
        double river = 1.0 - smooth(Math.abs(rv) / RIVER_WIDTH);
        h -= river * RIVER_DEPTH;

        return h;
    }

    /** dump a grid as "x z height" lines; used to check the Python implementation against this one. */
    public static void main(String[] args) {
        int size = args.length > 0 ? Integer.parseInt(args[0]) : 64;
        StringBuilder sb = new StringBuilder();
        for (int z = 0; z < size; z++) {
            for (int x = 0; x < size; x++) {
                sb.append(x).append(' ').append(z).append(' ')
                  .append(String.format("%.6f", height(x, z))).append('\n');
            }
        }
        System.out.print(sb);
    }
}
