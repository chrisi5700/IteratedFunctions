// IFS Particle Fragment Shader
// Simple color output for particles

struct FragmentInput {
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

struct ColorParams {
    float4 color;         // Base color
};

[[vk::binding(2, 0)]]
ConstantBuffer<ColorParams> colorParams;

[shader("fragment")]
float4 main(FragmentInput input) : SV_Target {
    // Simple gradient based on Y position
    float t = input.uv.y;
    float3 color1 = float3(0.2, 0.6, 1.0);  // Blue
    float3 color2 = float3(1.0, 0.4, 0.2);  // Orange
    float3 color = lerp(color1, color2, t);

    return float4(color * colorParams.color.rgb, colorParams.color.a);
}
