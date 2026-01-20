// Basic vertex shader with MVP transformation
struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 color : COLOR;
};

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 color : COLOR;
};

struct UniformBufferObject
{
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBufferObject> ubo;

[shader("vertex")]
VertexOutput main(VertexInput input)
{
    VertexOutput output;
    output.position = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.position, 1.0))));
    output.color = input.color;
    return output;
}
