// IFS Compute Shader - Sierpinski Triangle
// Applies affine transformations iteratively to particles

struct Particle {
    float2 position;  // Current position
    float2 _padding;  // Align to 16 bytes
};

RWStructuredBuffer<Particle> particles;

struct IFSParams {
    uint iterationCount;      // Number of iterations per dispatch
    uint particleCount;       // Total particles
    float scale;              // Global scale factor
    uint randomSeed;          // Seed for randomization
};

[[vk::binding(1, 0)]]
ConstantBuffer<IFSParams> params;

// Wang hash - fast, good distribution for graphics
uint wang_hash(uint seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

float hash_to_float(uint hash_val) {
    return float(hash_val) / 4294967296.0;
}

// Sierpinski triangle affine transforms
float2 applyTransform(float2 pos, int transformIndex) {
    // T1: Scale 0.5, no translation (bottom-left)
    if (transformIndex == 0) {
        return pos * 0.5;
    }
    // T2: Scale 0.5, translate right (bottom-right)
    else if (transformIndex == 1) {
        return pos * 0.5 + float2(0.5, 0.0);
    }
    // T3: Scale 0.5, translate to apex (top)
    else {
        return pos * 0.5 + float2(0.25, 0.433); // 0.433 ≈ sqrt(3)/4
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 GlobalInvocationID : SV_DispatchThreadID) {
    uint index = GlobalInvocationID.x;
    if (index >= params.particleCount)
        return;

    float2 pos = particles[index].position;

    // Apply N iterations of random transform selection
    for (uint iter = 0; iter < params.iterationCount; iter++) {
        // Generate pseudo-random number for transform selection
        uint seed = params.randomSeed + index * 1000u + iter;
        float randVal = hash_to_float(wang_hash(seed));

        // Select transform (0, 1, or 2 with equal probability)
        int transformIndex = int(floor(randVal * 3.0));
        transformIndex = clamp(transformIndex, 0, 2);

        // Apply selected transform
        pos = applyTransform(pos, transformIndex);
    }

    // Apply global scale (center around 0.5, scale, then re-center)
    pos = (pos - 0.5) * params.scale + 0.5;

    // Write back
    particles[index].position = pos;
}
