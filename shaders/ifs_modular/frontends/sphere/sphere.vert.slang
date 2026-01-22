// Sphere Vertex Shader - Instanced rendering
// Each instance is one particle, rendered as a sphere
import ifs_modular.common;


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

// Particle positions (storage buffer)
[[vk::binding(1, 0)]]
StructuredBuffer<Particle> particles;

// Vertex input (sphere mesh)
struct VertexInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
};

// Output to fragment shader
struct VertexOutput {
    float4 position : SV_Position;
    float3 world_pos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float3 view_dir : TEXCOORD2;
    float4 color : TEXCOORD3;  // Particle color from backend
};

[shader("vertex")]
VertexOutput main(VertexInput input, uint instance_id : SV_InstanceID) {
    VertexOutput output;

    // Get particle position and color
    Particle particle = particles[instance_id];
    float3 particle_pos = particle.position;

    // Scale sphere mesh by radius and translate to particle position
    float3 world_pos = particle_pos + input.position * view_params.sphere_radius;

    // Transform to clip space
    output.position = mul(view_params.view_projection, float4(world_pos, 1.0));
    output.world_pos = world_pos;
    output.normal = input.normal;  // Sphere normals are world-space for unit sphere
    output.view_dir = normalize(view_params.camera_pos - world_pos);
    output.color = particle.color;  // Pass particle color to fragment shader

    return output;
}
