// Particle Renderer - Fragment Shader
// Simple particle rendering with per-particle colors

struct FragmentInput {
    [[vk::location(0)]] float4 color : COLOR0;
};

[shader("fragment")]
float4 main(FragmentInput input) : SV_Target {
    // Output particle color directly
    // Could add circular fade-out for better particle appearance
    return input.color;
}
