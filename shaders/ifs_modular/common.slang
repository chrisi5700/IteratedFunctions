module common;

public struct Particle {
    public float3 position;  // World position (z will be 0 for 2D)
    float _padding1;  // Alignment
    public float4 color;     // RGBA color
};

public struct IFSParams {
    public uint iterationCount;      // Number of iterations per dispatch
    public uint particleCount;       // Total particles
    public float scale;              // Global scale factor
    public uint randomSeed;          // Seed for randomization
};

