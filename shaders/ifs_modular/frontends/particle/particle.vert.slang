// Particle Renderer - Vertex Shader
// Renders particles as GL_POINTS with 3D camera support

// Unified particle structure (matches include/ifs/ParticleData.hpp)
import ifs_modular.common;

// Camera/view parameters
public struct ViewParams {
    public column_major float4x4 viewProjection;  // Combined view-projection matrix
    public float2 screenSize;        // For aspect ratio / point size scaling
    public float pointSize;          // Base point size
    public float _padding;
};

[[vk::binding(0, 0)]]
StructuredBuffer<Particle> particles;



[[vk::binding(1, 0)]]
ConstantBuffer<ViewParams> viewParams;

struct VertexOutput {
    float4 position : SV_Position;
    float pointSize : SV_PointSize;
    [[vk::location(0)]] float4 color : COLOR0;
};

[shader("vertex")]
VertexOutput main(uint vertexID : SV_VertexID) {
    VertexOutput output;

    // Read particle data
    Particle particle = particles[vertexID];

    // Transform position to clip space
    output.position = mul(viewParams.viewProjection, float4(particle.position, 1.0));

    // Set point size (can be made dynamic based on distance later)
    output.pointSize = viewParams.pointSize;

    // Pass color to fragment shader
    output.color = particle.color;

    return output;
}
