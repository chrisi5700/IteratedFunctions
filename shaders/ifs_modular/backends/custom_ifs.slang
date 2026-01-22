// Sierpinski Triangle 2D - IFS Compute Shader
// Generates classic Sierpinski triangle fractal using 3 affine transforms

// Unified particle structure (matches include/ifs/ParticleData.hpp)
import ifs_modular.common;



[[vk::binding(0, 0)]]
RWStructuredBuffer<Particle> particles;

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


float3 applyTransform(float3 pos, float rand)
{
    if (rand < 0.01)
    {
        // Stem
        return float3(
            0.0,
            0.16 * pos.y,
            pos.z
        );
    }
    else if (rand < 0.86)
    {
        // Successively smaller leaflets
        return float3(
            0.85 * pos.x + 0.04 * pos.y,
           -0.04 * pos.x + 0.85 * pos.y + 1.6,
            pos.z
        );
    }
    else if (rand < 0.93)
    {
        // Left leaflet
        return float3(
            0.20 * pos.x - 0.26 * pos.y,
            0.23 * pos.x + 0.22 * pos.y + 1.6,
            pos.z
        );
    }
    else
    {
        // Right leaflet
        return float3(
           -0.15 * pos.x + 0.28 * pos.y,
            0.26 * pos.x + 0.24 * pos.y + 0.44,
            pos.z
        );
    }
}

float4 compute_color(float3 pos)
{
    // Normalize height roughly into [0,1]
    float h = saturate(pos.y / 10.0);

    float3 rgb;
    rgb.r = lerp(0.05, 0.2, h);
    rgb.g = lerp(0.2,  0.9, h);
    rgb.b = lerp(0.05, 0.3, h);

    return float4(rgb, 1.0);
}



[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 GlobalInvocationID : SV_DispatchThreadID) {
    uint index = GlobalInvocationID.x;
    if (index >= params.particleCount)
        return;

    float3 pos = 0;

    // Apply N iterations of random IFS transform
    for (uint iter = 0; iter < params.iterationCount; iter++) {
        // Generate pseudo-random transform selection
        uint seed = params.randomSeed + index * 1000u + iter;
        float randVal = hash_to_float(wang_hash(seed));

        // Apply selected affine transform
        pos = applyTransform(pos, randVal);
    }

    // Apply global scale (center around 0.5, scale, then re-center)
    pos = (pos - 0.5) * params.scale + 0.5;

    // Write back to particle buffer
    particles[index].position = float3(pos.x, pos.y, pos.z);  // z=0 for 2D
    particles[index].color = compute_color(pos);
}
