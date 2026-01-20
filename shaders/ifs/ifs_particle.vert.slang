// IFS Particle Vertex Shader
// Renders particles as points from compute buffer

struct Particle {
    float2 position;
    float2 _padding;
};

[[vk::binding(0, 0)]]
StructuredBuffer<Particle> particles;

struct VertexOutput {
    float4 position : SV_Position;
    float pointSize : SV_PointSize;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

struct ViewParams {
    float2 screenSize;    // For aspect ratio correction
    float pointSize;      // Particle size
    float _padding;
};

[[vk::binding(1, 0)]]
ConstantBuffer<ViewParams> viewParams;

[shader("vertex")]
VertexOutput main(uint vertexID : SV_VertexID) {
    VertexOutput output;

    // Read particle position
    Particle particle = particles[vertexID];

    // Map [0,1] to [-1,1] NDC space
    float2 ndc = particle.position * 2.0 - 1.0;

    // Flip Y for Vulkan coordinate system
    ndc.y = -ndc.y;

    output.position = float4(ndc, 0.0, 1.0);
    output.pointSize = viewParams.pointSize;
    output.uv = particle.position; // For color variation

    return output;
}
