// Sierpinski Triangle 2D - IFS Compute Shader
// Generates classic Sierpinski triangle fractal using 3 affine transforms

// Unified particle structure (matches include/ifs/ParticleData.hpp)
struct Particle {
    float3 position;  // World position (z will be 0 for 2D)
    float _padding1;  // Alignment
    float4 color;     // RGBA color
};

[[vk::binding(0, 0)]]
RWStructuredBuffer<Particle> particles;

struct IFSParams {
    uint iterationCount;      // Number of iterations per dispatch
    uint particleCount;       // Total particles
    float scale;              // Global scale factor
    uint randomSeed;          // Seed for randomization
};

[[vk::binding(1, 0)]]
ConstantBuffer<IFSParams> params;

// Wang hash - fast pseudo-random number generator
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
// Three transforms corresponding to the three vertices of an equilateral triangle
float2 applyTransform(float2 pos, int transformIndex) {
    // Transform 0: Bottom-left vertex (0, 0)
    // Maps to bottom-left half of triangle
    if (transformIndex == 0) {
        return pos * 0.5;
    }
    // Transform 1: Bottom-right vertex (1, 0)
    // Maps to bottom-right half of triangle
    else if (transformIndex == 1) {
        return pos * 0.5 + float2(0.5, 0.0);
    }
    // Transform 2: Top vertex (0.5, 0.866)
    // Maps to top half of triangle (0.866 ≈ sqrt(3)/2 for equilateral)
    else {
        return pos * 0.5 + float2(0.25, 0.433);
    }
}

// Generate color based on position (gradient)
float4 compute_color(float2 pos) {
    // Rainbow gradient based on X position
    float t = pos.x;
    float3 rgb;
    rgb.r = 0.5 + 0.5 * cos(6.28318 * (t + 0.0));
    rgb.g = 0.5 + 0.5 * cos(6.28318 * (t + 0.33));
    rgb.b = 0.5 + 0.5 * cos(6.28318 * (t + 0.67));
    return float4(rgb, 1.0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 GlobalInvocationID : SV_DispatchThreadID) {
    uint index = GlobalInvocationID.x;
    if (index >= params.particleCount)
        return;

    // Get current position (only use x,y since this is 2D)
    float2 pos = particles[index].position.xy;

    // Apply N iterations of random IFS transform
    for (uint iter = 0; iter < params.iterationCount; iter++) {
        // Generate pseudo-random transform selection
        uint seed = params.randomSeed + index * 1000u + iter;
        float randVal = hash_to_float(wang_hash(seed));

        // Select one of 3 transforms with equal probability
        int transformIndex = int(floor(randVal * 3.0));
        transformIndex = clamp(transformIndex, 0, 2);

        // Apply selected affine transform
        pos = applyTransform(pos, transformIndex);
    }

    // Apply global scale (center around 0.5, scale, then re-center)
    pos = (pos - 0.5) * params.scale + 0.5;

    // Write back to particle buffer
    particles[index].position = float3(pos.x, pos.y, 0.0);  // z=0 for 2D
    particles[index].color = compute_color(pos);
}
