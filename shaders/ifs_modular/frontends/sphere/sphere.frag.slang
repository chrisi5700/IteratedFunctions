// Sphere Fragment Shader - Simple Blinn-Phong shading

struct ViewParams {
    column_major float4x4 view_projection; // GLM is in col major while slang is in col major!!
    float3 camera_pos;
    float sphere_radius;
    float3 light_dir;
    float padding;
};

// Uniforms
[[vk::binding(0, 0)]]
ConstantBuffer<ViewParams> view_params;

// Input from vertex shader
struct FragmentInput {
    float3 world_pos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float3 view_dir : TEXCOORD2;
    float4 color : TEXCOORD3;  // Particle color from backend
};

// Output
struct FragmentOutput {
    float4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput main(FragmentInput input) {
    FragmentOutput output;

    // Normalize interpolated vectors
    float3 N = normalize(input.normal);
    float3 V = normalize(input.view_dir);
    float3 L = normalize(view_params.light_dir);

    // Base color from particle (computed by backend)
    float3 base_color = input.color.rgb;

    // Ambient
    float3 ambient = 0.3 * base_color;

    // Diffuse
    float diff = max(dot(N, L), 0.0);
    float3 diffuse = diff * base_color;

    // Specular (Blinn-Phong)
    float3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 32.0);
    float3 specular = float3(0.2) * spec;  // Slightly less specular to preserve color

    // Final color
    float3 color = ambient + diffuse + specular;

    output.color = float4(color, input.color.a);

    return output;
}
