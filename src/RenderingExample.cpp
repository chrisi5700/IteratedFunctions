#include <vulkan/vulkan.hpp>

#include "RenderTarget.hpp"

// ============================================================================
// RENDER PASS
// ============================================================================

auto colorAttachment0 = vk::AttachmentDescription()
    .setFormat(swapchainFormat)                              // From RenderTarget
    .setSamples(vk::SampleCountFlagBits::e1)                 // From RenderTarget
    .setLoadOp(vk::AttachmentLoadOp::eClear)                 // Non-obvious choice (clear vs load vs dont_care)
    .setStoreOp(vk::AttachmentStoreOp::eStore)               // Default (almost always store)
    .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)       // Default (no stencil on color)
    .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)     // Default (no stencil on color)
    .setInitialLayout(vk::ImageLayout::eUndefined)           // Non-obvious choice (depends on prior usage)
    .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);        // From RenderTarget (present vs shader read vs transfer)

// Additional color attachments for MRT (Multiple Render Targets)
// Use when: deferred rendering, G-buffer, multiple output textures
auto colorAttachment1 = vk::AttachmentDescription()
    .setFormat(vk::Format::eR16G16B16A16Sfloat)              // From RenderTarget (G-buffer normal)
    .setSamples(vk::SampleCountFlagBits::e1)                 // From RenderTarget (must match)
    .setLoadOp(vk::AttachmentLoadOp::eClear)                 // Non-obvious choice
    .setStoreOp(vk::AttachmentStoreOp::eStore)               // Default
    .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)       // Default
    .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)     // Default
    .setInitialLayout(vk::ImageLayout::eUndefined)           // Non-obvious choice
    .setFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal); // From RenderTarget (will sample later)

auto colorAttachment2 = vk::AttachmentDescription()
    .setFormat(vk::Format::eR8G8B8A8Unorm)                   // From RenderTarget (G-buffer albedo)
    .setSamples(vk::SampleCountFlagBits::e1)                 // From RenderTarget
    .setLoadOp(vk::AttachmentLoadOp::eClear)                 // Non-obvious choice
    .setStoreOp(vk::AttachmentStoreOp::eStore)               // Default
    .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)       // Default
    .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)     // Default
    .setInitialLayout(vk::ImageLayout::eUndefined)           // Non-obvious choice
    .setFinalLayout(vk::ImageLayout::eShaderReadOnlyOptimal); // From RenderTarget

auto depthAttachment = vk::AttachmentDescription()
    .setFormat(vk::Format::eD32SfloatS8Uint)                 // From RenderTarget (D32S8 for stencil support)
    .setSamples(vk::SampleCountFlagBits::e1)                 // From RenderTarget (must match color)
    .setLoadOp(vk::AttachmentLoadOp::eClear)                 // Non-obvious choice
    .setStoreOp(vk::AttachmentStoreOp::eDontCare)            // Non-obvious choice (store if reusing depth)
    .setStencilLoadOp(vk::AttachmentLoadOp::eClear)          // Non-obvious choice (if using stencil)
    .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)     // Non-obvious choice
    .setInitialLayout(vk::ImageLayout::eUndefined)           // Non-obvious choice
    .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal); // Default for depth

// Attachment references - indices must match attachment array order
auto colorAttachmentRef0 = vk::AttachmentReference()
    .setAttachment(0)                                        // From RenderTarget (attachment index)
    .setLayout(vk::ImageLayout::eColorAttachmentOptimal);    // Default for color output

auto colorAttachmentRef1 = vk::AttachmentReference()
    .setAttachment(1)                                        // From RenderTarget
    .setLayout(vk::ImageLayout::eColorAttachmentOptimal);    // Default

auto colorAttachmentRef2 = vk::AttachmentReference()
    .setAttachment(2)                                        // From RenderTarget
    .setLayout(vk::ImageLayout::eColorAttachmentOptimal);    // Default

std::array colorAttachmentRefs = {colorAttachmentRef0, colorAttachmentRef1, colorAttachmentRef2};

auto depthAttachmentRef = vk::AttachmentReference()
    .setAttachment(3)                                        // From RenderTarget (after color attachments)
    .setLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal); // Default for depth

auto subpass = vk::SubpassDescription()
    .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)  // Default (graphics pipeline)
    .setColorAttachments(colorAttachmentRefs)                // From fragment shader (output count) + RenderTarget
    .setPDepthStencilAttachment(&depthAttachmentRef);        // From RenderTarget (nullptr if no depth)

auto dependency = vk::SubpassDependency()
    .setSrcSubpass(VK_SUBPASS_EXTERNAL)                      // Default for single-subpass
    .setDstSubpass(0)                                        // Default for single-subpass
    .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput |
                     vk::PipelineStageFlagBits::eEarlyFragmentTests)  // Default
    .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput |
                     vk::PipelineStageFlagBits::eEarlyFragmentTests)  // Default
    .setSrcAccessMask(vk::AccessFlags{})                     // Default
    .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite |
                      vk::AccessFlagBits::eDepthStencilAttachmentWrite); // Default

std::array attachments = {colorAttachment0, colorAttachment1, colorAttachment2, depthAttachment};

auto renderPassInfo = vk::RenderPassCreateInfo()
    .setAttachments(attachments)                             // From RenderTarget + fragment shader output count
    .setSubpassCount(1)                                      // Default (single subpass)
    .setPSubpasses(&subpass)
    .setDependencyCount(1)                                   // Default
    .setPDependencies(&dependency);

renderPass = device.createRenderPass(renderPassInfo);

// ============================================================================
// DESCRIPTOR SET LAYOUT
// ============================================================================

// Set 0: Per-frame data (uniform buffers, etc.)
auto uboLayoutBinding = vk::DescriptorSetLayoutBinding()
    .setBinding(0)                                           // From shader reflection
    .setDescriptorType(vk::DescriptorType::eUniformBuffer)   // From shader reflection
    .setDescriptorCount(1)                                   // From shader reflection
    .setStageFlags(vk::ShaderStageFlagBits::eVertex |
                   vk::ShaderStageFlagBits::eTessellationControl |
                   vk::ShaderStageFlagBits::eTessellationEvaluation |
                   vk::ShaderStageFlagBits::eGeometry);      // From shader reflection (merged stages)

auto samplerLayoutBinding = vk::DescriptorSetLayoutBinding()
    .setBinding(1)                                           // From shader reflection
    .setDescriptorType(vk::DescriptorType::eCombinedImageSampler) // From shader reflection
    .setDescriptorCount(1)                                   // From shader reflection
    .setStageFlags(vk::ShaderStageFlagBits::eFragment);      // From shader reflection

// Storage buffer for compute/geometry shader data
// Use when: GPU-driven rendering, particle systems, skinning
auto storageBufferBinding = vk::DescriptorSetLayoutBinding()
    .setBinding(2)                                           // From shader reflection
    .setDescriptorType(vk::DescriptorType::eStorageBuffer)   // From shader reflection
    .setDescriptorCount(1)                                   // From shader reflection
    .setStageFlags(vk::ShaderStageFlagBits::eVertex |
                   vk::ShaderStageFlagBits::eGeometry);      // From shader reflection

std::array set0Bindings = {uboLayoutBinding, samplerLayoutBinding, storageBufferBinding};

auto set0LayoutInfo = vk::DescriptorSetLayoutCreateInfo()
    .setBindings(set0Bindings);                              // From shader reflection

descriptorSetLayout0 = device.createDescriptorSetLayout(set0LayoutInfo);

// Set 1: Per-material data (textures, material properties)
// Use when: you want to swap materials without rebinding everything
auto albedoBinding = vk::DescriptorSetLayoutBinding()
    .setBinding(0)                                           // From shader reflection
    .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
    .setDescriptorCount(1)
    .setStageFlags(vk::ShaderStageFlagBits::eFragment);

auto normalMapBinding = vk::DescriptorSetLayoutBinding()
    .setBinding(1)                                           // From shader reflection
    .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
    .setDescriptorCount(1)
    .setStageFlags(vk::ShaderStageFlagBits::eFragment);

std::array set1Bindings = {albedoBinding, normalMapBinding};

auto set1LayoutInfo = vk::DescriptorSetLayoutCreateInfo()
    .setBindings(set1Bindings);

descriptorSetLayout1 = device.createDescriptorSetLayout(set1LayoutInfo);

std::array descriptorSetLayouts = {descriptorSetLayout0, descriptorSetLayout1};

// ============================================================================
// PIPELINE LAYOUT
// ============================================================================

// Push constants - fast path for small, frequently changing data
// Use when: model matrix, object ID, material index
auto pushConstantRange = vk::PushConstantRange()
    .setStageFlags(vk::ShaderStageFlagBits::eVertex |
                   vk::ShaderStageFlagBits::eFragment)       // From shader reflection (merged stages)
    .setOffset(0)                                            // From shader reflection
    .setSize(sizeof(PushConstants));                         // From shader reflection

auto pipelineLayoutInfo = vk::PipelineLayoutCreateInfo()
    .setSetLayouts(descriptorSetLayouts)                     // From shader reflection
    .setPushConstantRanges(pushConstantRange);               // From shader reflection

pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

// ============================================================================
// SHADER MODULES
// ============================================================================

auto vertShaderCode = readFile("shader.vert.spv");           // From shader
auto hullShaderCode = readFile("shader.tesc.spv");           // From shader (tessellation control)
auto domainShaderCode = readFile("shader.tese.spv");         // From shader (tessellation evaluation)
auto geomShaderCode = readFile("shader.geom.spv");           // From shader
auto fragShaderCode = readFile("shader.frag.spv");           // From shader

vk::ShaderModule vertShaderModule = createShaderModule(vertShaderCode);
vk::ShaderModule hullShaderModule = createShaderModule(hullShaderCode);
vk::ShaderModule domainShaderModule = createShaderModule(domainShaderCode);
vk::ShaderModule geomShaderModule = createShaderModule(geomShaderCode);
vk::ShaderModule fragShaderModule = createShaderModule(fragShaderCode);

// ============================================================================
// SPECIALIZATION CONSTANTS
// ============================================================================

// Use when: compile-time shader variants without separate SPIR-V files
// Examples: MAX_LIGHTS, ENABLE_SHADOWS, SAMPLE_COUNT, KERNEL_SIZE

struct VertexSpecConstants {
    uint32_t maxBones = 64;                                  // User-provided value
    uint32_t enableSkinning = 1;                             // User-provided value
};
VertexSpecConstants vertSpecData;

std::array vertSpecEntries = {
    vk::SpecializationMapEntry()
        .setConstantID(0)                                    // From shader reflection
        .setOffset(offsetof(VertexSpecConstants, maxBones))  // Computed
        .setSize(sizeof(uint32_t)),                          // From shader reflection

    vk::SpecializationMapEntry()
        .setConstantID(1)                                    // From shader reflection
        .setOffset(offsetof(VertexSpecConstants, enableSkinning))
        .setSize(sizeof(uint32_t))
};

auto vertSpecInfo = vk::SpecializationInfo()
    .setMapEntries(vertSpecEntries)                          // From shader reflection
    .setDataSize(sizeof(vertSpecData))
    .setPData(&vertSpecData);                                // User-provided values

struct FragmentSpecConstants {
    uint32_t maxLights = 16;                                 // User-provided value
    float gamma = 2.2f;                                      // User-provided value
    uint32_t enablePCF = 1;                                  // User-provided value (soft shadows)
    uint32_t pcfKernelSize = 3;                              // User-provided value
};
FragmentSpecConstants fragSpecData;

std::array fragSpecEntries = {
    vk::SpecializationMapEntry()
        .setConstantID(0)                                    // From shader reflection
        .setOffset(offsetof(FragmentSpecConstants, maxLights))
        .setSize(sizeof(uint32_t)),

    vk::SpecializationMapEntry()
        .setConstantID(1)                                    // From shader reflection
        .setOffset(offsetof(FragmentSpecConstants, gamma))
        .setSize(sizeof(float)),

    vk::SpecializationMapEntry()
        .setConstantID(2)                                    // From shader reflection
        .setOffset(offsetof(FragmentSpecConstants, enablePCF))
        .setSize(sizeof(uint32_t)),

    vk::SpecializationMapEntry()
        .setConstantID(3)                                    // From shader reflection
        .setOffset(offsetof(FragmentSpecConstants, pcfKernelSize))
        .setSize(sizeof(uint32_t))
};

auto fragSpecInfo = vk::SpecializationInfo()
    .setMapEntries(fragSpecEntries)
    .setDataSize(sizeof(fragSpecData))
    .setPData(&fragSpecData);

// ============================================================================
// SHADER STAGES
// ============================================================================

auto vertShaderStageInfo = vk::PipelineShaderStageCreateInfo()
    .setStage(vk::ShaderStageFlagBits::eVertex)              // From shader reflection
    .setModule(vertShaderModule)                             // From shader
    .setPName("main")                                        // From shader reflection
    .setPSpecializationInfo(&vertSpecInfo);                  // User-provided (optional)

// Tessellation Control (Hull) shader
// Use when: dynamic LOD, terrain, curved surfaces, displacement mapping
auto hullShaderStageInfo = vk::PipelineShaderStageCreateInfo()
    .setStage(vk::ShaderStageFlagBits::eTessellationControl) // From shader reflection
    .setModule(hullShaderModule)                             // From shader
    .setPName("main");                                       // From shader reflection

// Tessellation Evaluation (Domain) shader
auto domainShaderStageInfo = vk::PipelineShaderStageCreateInfo()
    .setStage(vk::ShaderStageFlagBits::eTessellationEvaluation) // From shader reflection
    .setModule(domainShaderModule)                           // From shader
    .setPName("main");                                       // From shader reflection

// Geometry shader
// Use when: point sprites, wireframe overlay, silhouette detection, layered rendering, shadow volume
auto geomShaderStageInfo = vk::PipelineShaderStageCreateInfo()
    .setStage(vk::ShaderStageFlagBits::eGeometry)            // From shader reflection
    .setModule(geomShaderModule)                             // From shader
    .setPName("main");                                       // From shader reflection

auto fragShaderStageInfo = vk::PipelineShaderStageCreateInfo()
    .setStage(vk::ShaderStageFlagBits::eFragment)            // From shader reflection
    .setModule(fragShaderModule)                             // From shader
    .setPName("main")                                        // From shader reflection
    .setPSpecializationInfo(&fragSpecInfo);                  // User-provided (optional)

// Full pipeline: vert -> tess control -> tess eval -> geometry -> fragment
std::array shaderStages = {
    vertShaderStageInfo,
    hullShaderStageInfo,
    domainShaderStageInfo,
    geomShaderStageInfo,
    fragShaderStageInfo
};

// Minimal pipeline (no tessellation, no geometry): vert -> fragment
// std::array shaderStages = {vertShaderStageInfo, fragShaderStageInfo};

// ============================================================================
// VERTEX INPUT
// ============================================================================

// Binding 0: Per-vertex data
auto vertexBindingDescription = vk::VertexInputBindingDescription()
    .setBinding(0)                                           // From vertex shader reflection
    .setStride(sizeof(Vertex))                               // From vertex shader reflection (stride)
    .setInputRate(vk::VertexInputRate::eVertex);             // Non-obvious choice

// Binding 1: Per-instance data
// Use when: instanced rendering (grass, trees, particles, crowds)
auto instanceBindingDescription = vk::VertexInputBindingDescription()
    .setBinding(1)                                           // From vertex shader reflection
    .setStride(sizeof(InstanceData))                         // From vertex shader reflection
    .setInputRate(vk::VertexInputRate::eInstance);           // Non-obvious choice

std::array bindingDescriptions = {vertexBindingDescription, instanceBindingDescription};

// Per-vertex attributes (binding 0)
auto positionAttribute = vk::VertexInputAttributeDescription()
    .setBinding(0)                                           // From vertex shader reflection
    .setLocation(0)                                          // From vertex shader reflection
    .setFormat(vk::Format::eR32G32B32Sfloat)                 // From vertex shader reflection
    .setOffset(offsetof(Vertex, pos));                       // Computed from shader reflection

auto normalAttribute = vk::VertexInputAttributeDescription()
    .setBinding(0)                                           // From vertex shader reflection
    .setLocation(1)                                          // From vertex shader reflection
    .setFormat(vk::Format::eR32G32B32Sfloat)                 // From vertex shader reflection
    .setOffset(offsetof(Vertex, normal));                    // Computed

auto texCoordAttribute = vk::VertexInputAttributeDescription()
    .setBinding(0)                                           // From vertex shader reflection
    .setLocation(2)                                          // From vertex shader reflection
    .setFormat(vk::Format::eR32G32Sfloat)                    // From vertex shader reflection
    .setOffset(offsetof(Vertex, texCoord));                  // Computed

auto tangentAttribute = vk::VertexInputAttributeDescription()
    .setBinding(0)                                           // From vertex shader reflection
    .setLocation(3)                                          // From vertex shader reflection
    .setFormat(vk::Format::eR32G32B32A32Sfloat)              // From vertex shader reflection (w = handedness)
    .setOffset(offsetof(Vertex, tangent));                   // Computed

// Skinning data (bone indices + weights)
// Use when: skeletal animation
auto boneIndicesAttribute = vk::VertexInputAttributeDescription()
    .setBinding(0)                                           // From vertex shader reflection
    .setLocation(4)                                          // From vertex shader reflection
    .setFormat(vk::Format::eR32G32B32A32Uint)                // From vertex shader reflection (4 bone indices)
    .setOffset(offsetof(Vertex, boneIndices));               // Computed

auto boneWeightsAttribute = vk::VertexInputAttributeDescription()
    .setBinding(0)                                           // From vertex shader reflection
    .setLocation(5)                                          // From vertex shader reflection
    .setFormat(vk::Format::eR32G32B32A32Sfloat)              // From vertex shader reflection (4 weights)
    .setOffset(offsetof(Vertex, boneWeights));               // Computed

// Per-instance attributes (binding 1)
// Instance transform matrix (takes 4 locations for mat4)
auto instanceMatrixCol0 = vk::VertexInputAttributeDescription()
    .setBinding(1)                                           // From vertex shader reflection
    .setLocation(6)                                          // From vertex shader reflection
    .setFormat(vk::Format::eR32G32B32A32Sfloat)              // From vertex shader reflection
    .setOffset(offsetof(InstanceData, transform) + 0);       // Computed

auto instanceMatrixCol1 = vk::VertexInputAttributeDescription()
    .setBinding(1)
    .setLocation(7)
    .setFormat(vk::Format::eR32G32B32A32Sfloat)
    .setOffset(offsetof(InstanceData, transform) + 16);

auto instanceMatrixCol2 = vk::VertexInputAttributeDescription()
    .setBinding(1)
    .setLocation(8)
    .setFormat(vk::Format::eR32G32B32A32Sfloat)
    .setOffset(offsetof(InstanceData, transform) + 32);

auto instanceMatrixCol3 = vk::VertexInputAttributeDescription()
    .setBinding(1)
    .setLocation(9)
    .setFormat(vk::Format::eR32G32B32A32Sfloat)
    .setOffset(offsetof(InstanceData, transform) + 48);

auto instanceColorAttribute = vk::VertexInputAttributeDescription()
    .setBinding(1)                                           // From vertex shader reflection
    .setLocation(10)                                         // From vertex shader reflection
    .setFormat(vk::Format::eR32G32B32A32Sfloat)              // From vertex shader reflection
    .setOffset(offsetof(InstanceData, color));               // Computed

std::array attributeDescriptions = {
    positionAttribute, normalAttribute, texCoordAttribute, tangentAttribute,
    boneIndicesAttribute, boneWeightsAttribute,
    instanceMatrixCol0, instanceMatrixCol1, instanceMatrixCol2, instanceMatrixCol3,
    instanceColorAttribute
};

auto vertexInputInfo = vk::PipelineVertexInputStateCreateInfo()
    .setVertexBindingDescriptions(bindingDescriptions)       // From vertex shader reflection
    .setVertexAttributeDescriptions(attributeDescriptions);  // From vertex shader reflection

// ============================================================================
// INPUT ASSEMBLY
// ============================================================================

auto inputAssembly = vk::PipelineInputAssemblyStateCreateInfo()
    // For tessellation: must be ePatchList
    // For geometry shader with triangle input: eTriangleList, eTriangleStrip, eTriangleFan
    // For geometry shader with line input: eLineList, eLineStrip
    // For geometry shader with point input: ePointList
    // For adjacency (silhouette detection): eTriangleListWithAdjacency, eLineListWithAdjacency
    .setTopology(vk::PrimitiveTopology::ePatchList)          // Non-obvious choice (ePatchList for tessellation)
    .setPrimitiveRestartEnable(VK_FALSE);                    // Default (VK_TRUE for strip topologies with 0xFFFF index)

// ============================================================================
// TESSELLATION STATE
// ============================================================================

// Use when: tessellation shaders are present
// Patch control points: number of vertices per patch
// Common values: 3 (triangles), 4 (quads), 16 (bicubic patches)
auto tessellationState = vk::PipelineTessellationStateCreateInfo()
    .setPatchControlPoints(3);                               // From tess control shader (outputcontrolpoints)
                                                             // Or non-obvious choice if not extractable

// ============================================================================
// VIEWPORT / SCISSOR
// ============================================================================

// Single viewport (normal rendering)
auto viewport = vk::Viewport()
    .setX(0.0f)                                              // Default
    .setY(0.0f)                                              // Default
    .setWidth(static_cast<float>(swapchainExtent.width))     // From RenderTarget
    .setHeight(static_cast<float>(swapchainExtent.height))   // From RenderTarget
    .setMinDepth(0.0f)                                       // Default
    .setMaxDepth(1.0f);                                      // Default

auto scissor = vk::Rect2D()
    .setOffset({0, 0})                                       // Default
    .setExtent(swapchainExtent);                             // From RenderTarget

// Multiple viewports
// Use when: VR (left/right eye), split-screen, cubemap rendering, cascaded shadow maps
// Geometry shader selects viewport via SV_ViewportArrayIndex
std::array<vk::Viewport, 2> multiViewports = {
    vk::Viewport(0.0f, 0.0f,
                 swapchainExtent.width / 2.0f, swapchainExtent.height,
                 0.0f, 1.0f),                                // Left half
    vk::Viewport(swapchainExtent.width / 2.0f, 0.0f,
                 swapchainExtent.width / 2.0f, swapchainExtent.height,
                 0.0f, 1.0f)                                 // Right half
};

std::array<vk::Rect2D, 2> multiScissors = {
    vk::Rect2D({0, 0}, {swapchainExtent.width / 2, swapchainExtent.height}),
    vk::Rect2D({static_cast<int32_t>(swapchainExtent.width / 2), 0},
               {swapchainExtent.width / 2, swapchainExtent.height})
};

// Single viewport state (most common)
auto viewportState = vk::PipelineViewportStateCreateInfo()
    .setViewportCount(1)                                     // Default (or multi-viewport count)
    .setPViewports(&viewport)                                // From RenderTarget (ignored if dynamic)
    .setScissorCount(1)                                      // Default (must match viewport count)
    .setPScissors(&scissor);                                 // From RenderTarget (ignored if dynamic)

// Multi-viewport state (requires multiViewport device feature)
// auto viewportState = vk::PipelineViewportStateCreateInfo()
//     .setViewports(multiViewports)                         // Non-obvious choice
//     .setScissors(multiScissors);                          // Non-obvious choice

// ============================================================================
// RASTERIZATION
// ============================================================================

auto rasterizer = vk::PipelineRasterizationStateCreateInfo()
    // Depth clamp: clamp fragments outside [0,1] instead of discarding
    // Use when: shadow mapping (avoid near plane clipping)
    // Requires: depthClamp device feature
    .setDepthClampEnable(VK_FALSE)                           // Non-obvious choice (VK_TRUE for shadow maps)

    // Rasterizer discard: skip rasterization entirely
    // Use when: transform feedback, compute-only passes
    .setRasterizerDiscardEnable(VK_FALSE)                    // Default

    // Polygon mode: how to draw polygons
    // eFill (solid), eLine (wireframe), ePoint (vertices only)
    // Requires: fillModeNonSolid device feature for eLine/ePoint
    .setPolygonMode(vk::PolygonMode::eFill)                  // Default (eLine for debug wireframe)

    // Face culling
    // eNone: draw all faces (two-sided materials, foliage)
    // eBack: cull back faces (most solid objects)
    // eFront: cull front faces (inside-out rendering, shadow volumes)
    // eFrontAndBack: cull all faces (occlusion queries)
    .setCullMode(vk::CullModeFlagBits::eBack)                // Non-obvious choice

    // Front face winding order
    // eCounterClockwise: OpenGL/glTF convention
    // eClockwise: DirectX convention, some modeling tools
    .setFrontFace(vk::FrontFace::eCounterClockwise)          // Non-obvious choice

    // Depth bias: offset depth values
    // Use when: shadow mapping (avoid shadow acne), decals (avoid z-fighting)
    .setDepthBiasEnable(VK_FALSE)                            // Non-obvious choice (VK_TRUE for shadows/decals)
    .setDepthBiasConstantFactor(0.0f)                        // Non-obvious choice (typical: 1.25 for shadows)
    .setDepthBiasClamp(0.0f)                                 // Non-obvious choice (0 = no clamp)
    .setDepthBiasSlopeFactor(0.0f)                           // Non-obvious choice (typical: 1.75 for shadows)

    // Line width for eLine polygon mode or line primitives
    // Requires: wideLines device feature for values != 1.0
    .setLineWidth(1.0f);                                     // Default (or non-obvious choice)

// Conservative rasterization
// Use when: visibility buffer, voxelization, collision detection
// Requires: VK_EXT_conservative_rasterization extension
auto conservativeRasterInfo = vk::PipelineRasterizationConservativeStateCreateInfoEXT()
    // eOverestimate: rasterize if ANY part of primitive touches pixel (more fragments)
    // eUnderestimate: rasterize only if primitive fully covers pixel (fewer fragments)
    .setConservativeRasterizationMode(
        vk::ConservativeRasterizationModeEXT::eOverestimate) // Non-obvious choice
    .setExtraPrimitiveOverestimationSize(0.0f);              // Non-obvious choice

// Chain conservative rasterization if needed:
// rasterizer.setPNext(&conservativeRasterInfo);

// ============================================================================
// MULTISAMPLING
// ============================================================================

auto multisampling = vk::PipelineMultisampleStateCreateInfo()
    // Sample count: must match render target attachments
    // Higher = better quality, more expensive
    // Common values: e1 (no MSAA), e2, e4, e8
    .setRasterizationSamples(vk::SampleCountFlagBits::e1)    // From RenderTarget

    // Sample shading: run fragment shader per sample instead of per pixel
    // Use when: high-quality antialiasing for textures, alpha-tested geometry
    // Requires: sampleRateShading device feature
    .setSampleShadingEnable(VK_FALSE)                        // Non-obvious choice (VK_TRUE for quality)
    .setMinSampleShading(1.0f)                               // Non-obvious choice (0.0-1.0, fraction of samples)

    // Sample mask: bitmask of which samples to update
    // Use when: custom MSAA patterns, stencil-like effects
    .setPSampleMask(nullptr)                                 // Default (all samples)

    // Alpha to coverage: use alpha as sample coverage mask
    // Use when: order-independent transparency for foliage, fences
    .setAlphaToCoverageEnable(VK_FALSE)                      // Non-obvious choice

    // Alpha to one: force alpha to 1.0 after test
    // Use when: specific blending requirements
    .setAlphaToOneEnable(VK_FALSE);                          // Default

// ============================================================================
// DEPTH / STENCIL
// ============================================================================

// Stencil operation state (for stencil test)
// Use when: portals, mirrors, decals, outline effects, shadow volumes
auto stencilOpState = vk::StencilOpState()
    .setFailOp(vk::StencilOp::eKeep)                         // Non-obvious choice (on stencil fail)
    .setPassOp(vk::StencilOp::eReplace)                      // Non-obvious choice (on stencil+depth pass)
    .setDepthFailOp(vk::StencilOp::eKeep)                    // Non-obvious choice (on depth fail)
    .setCompareOp(vk::CompareOp::eAlways)                    // Non-obvious choice
    .setCompareMask(0xFF)                                    // Non-obvious choice
    .setWriteMask(0xFF)                                      // Non-obvious choice
    .setReference(1);                                        // Non-obvious choice (set at draw time if dynamic)

auto depthStencil = vk::PipelineDepthStencilStateCreateInfo()
    // Depth test: compare fragment depth against depth buffer
    // Disable for: UI, skybox (draw last), particles (sorted back-to-front)
    .setDepthTestEnable(VK_TRUE)                             // Non-obvious choice

    // Depth write: update depth buffer with fragment depth
    // Disable for: transparent objects (after opaque pass), decals
    .setDepthWriteEnable(VK_TRUE)                            // Non-obvious choice

    // Depth compare operation
    // eLess: standard (closer fragments win)
    // eLessOrEqual: for equal-depth overdraw (decals)
    // eGreater/eGreaterOrEqual: reverse-Z buffer (better precision)
    // eAlways: always pass (disabled depth test but still write)
    .setDepthCompareOp(vk::CompareOp::eLess)                 // Default (or eGreater for reverse-Z)

    // Depth bounds test: discard fragments outside [min, max] depth range
    // Use when: light volumes (deferred rendering), portal rendering
    // Requires: depthBounds device feature
    .setDepthBoundsTestEnable(VK_FALSE)                      // Non-obvious choice
    .setMinDepthBounds(0.0f)                                 // Non-obvious choice
    .setMaxDepthBounds(1.0f)                                 // Non-obvious choice

    // Stencil test
    // Use when: portals, mirrors, decals, outline effects, shadow volumes, masking
    .setStencilTestEnable(VK_FALSE)                          // Non-obvious choice
    .setFront(stencilOpState)                                // Non-obvious choice (front-facing stencil)
    .setBack(stencilOpState);                                // Non-obvious choice (back-facing stencil)

// ============================================================================
// COLOR BLENDING
// ============================================================================

// Per-attachment blend state
// Need one for each color attachment (MRT)

// Attachment 0: main color output
auto colorBlendAttachment0 = vk::PipelineColorBlendAttachmentState()
    .setBlendEnable(VK_FALSE)                                // Non-obvious choice
    // Standard alpha blending (when enabled):
    // result.rgb = src.rgb * src.a + dst.rgb * (1 - src.a)
    // result.a = src.a + dst.a * (1 - src.a)
    .setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)      // Default (if blend enabled)
    .setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha) // Default (if blend enabled)
    .setColorBlendOp(vk::BlendOp::eAdd)                      // Default
    .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)           // Default
    .setDstAlphaBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha) // Default
    .setAlphaBlendOp(vk::BlendOp::eAdd)                      // Default
    .setColorWriteMask(vk::ColorComponentFlagBits::eR |
                       vk::ColorComponentFlagBits::eG |
                       vk::ColorComponentFlagBits::eB |
                       vk::ColorComponentFlagBits::eA);      // Default (disable channels for MRT optimization)

// Additive blending (particles, glow, light accumulation)
// result.rgb = src.rgb * src.a + dst.rgb
auto additiveBlendAttachment = vk::PipelineColorBlendAttachmentState()
    .setBlendEnable(VK_TRUE)
    .setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)      // Non-obvious choice
    .setDstColorBlendFactor(vk::BlendFactor::eOne)           // Non-obvious choice (additive)
    .setColorBlendOp(vk::BlendOp::eAdd)
    .setSrcAlphaBlendFactor(vk::BlendFactor::eZero)
    .setDstAlphaBlendFactor(vk::BlendFactor::eOne)
    .setAlphaBlendOp(vk::BlendOp::eAdd)
    .setColorWriteMask(vk::ColorComponentFlagBits::eR |
                       vk::ColorComponentFlagBits::eG |
                       vk::ColorComponentFlagBits::eB |
                       vk::ColorComponentFlagBits::eA);

// Premultiplied alpha (UI, sprites with premultiplied textures)
// result.rgb = src.rgb + dst.rgb * (1 - src.a)
auto premultipliedBlendAttachment = vk::PipelineColorBlendAttachmentState()
    .setBlendEnable(VK_TRUE)
    .setSrcColorBlendFactor(vk::BlendFactor::eOne)           // Non-obvious choice (premultiplied)
    .setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
    .setColorBlendOp(vk::BlendOp::eAdd)
    .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
    .setDstAlphaBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
    .setAlphaBlendOp(vk::BlendOp::eAdd)
    .setColorWriteMask(vk::ColorComponentFlagBits::eR |
                       vk::ColorComponentFlagBits::eG |
                       vk::ColorComponentFlagBits::eB |
                       vk::ColorComponentFlagBits::eA);

// Multiplicative blending (shadows, tinting, color grading)
// result.rgb = src.rgb * dst.rgb
auto multiplicativeBlendAttachment = vk::PipelineColorBlendAttachmentState()
    .setBlendEnable(VK_TRUE)
    .setSrcColorBlendFactor(vk::BlendFactor::eDstColor)      // Non-obvious choice
    .setDstColorBlendFactor(vk::BlendFactor::eZero)          // Non-obvious choice
    .setColorBlendOp(vk::BlendOp::eAdd)
    .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
    .setDstAlphaBlendFactor(vk::BlendFactor::eZero)
    .setAlphaBlendOp(vk::BlendOp::eAdd)
    .setColorWriteMask(vk::ColorComponentFlagBits::eR |
                       vk::ColorComponentFlagBits::eG |
                       vk::ColorComponentFlagBits::eB |
                       vk::ColorComponentFlagBits::eA);

// Dual-source blending (advanced transparency, custom blend modes)
// Requires: dualSrcBlend device feature
// Fragment shader outputs to both index 0 and index 1
auto dualSourceBlendAttachment = vk::PipelineColorBlendAttachmentState()
    .setBlendEnable(VK_TRUE)
    .setSrcColorBlendFactor(vk::BlendFactor::eOne)           // Non-obvious choice
    .setDstColorBlendFactor(vk::BlendFactor::eSrc1Color)     // Non-obvious choice (second output)
    .setColorBlendOp(vk::BlendOp::eAdd)
    .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
    .setDstAlphaBlendFactor(vk::BlendFactor::eSrc1Alpha)
    .setAlphaBlendOp(vk::BlendOp::eAdd)
    .setColorWriteMask(vk::ColorComponentFlagBits::eR |
                       vk::ColorComponentFlagBits::eG |
                       vk::ColorComponentFlagBits::eB |
                       vk::ColorComponentFlagBits::eA);

// Attachment 1: G-buffer normal (MRT - no blending)
auto colorBlendAttachment1 = vk::PipelineColorBlendAttachmentState()
    .setBlendEnable(VK_FALSE)                                // Non-obvious choice (usually no blend for G-buffer)
    .setColorWriteMask(vk::ColorComponentFlagBits::eR |
                       vk::ColorComponentFlagBits::eG |
                       vk::ColorComponentFlagBits::eB |
                       vk::ColorComponentFlagBits::eA);

// Attachment 2: G-buffer albedo (MRT - no blending)
auto colorBlendAttachment2 = vk::PipelineColorBlendAttachmentState()
    .setBlendEnable(VK_FALSE)                                // Non-obvious choice
    .setColorWriteMask(vk::ColorComponentFlagBits::eR |
                       vk::ColorComponentFlagBits::eG |
                       vk::ColorComponentFlagBits::eB |
                       vk::ColorComponentFlagBits::eA);

// Array of attachments (must match fragment shader output count and render pass)
std::array colorBlendAttachments = {
    colorBlendAttachment0,
    colorBlendAttachment1,
    colorBlendAttachment2
};                                                           // From fragment shader (output count)

auto colorBlending = vk::PipelineColorBlendStateCreateInfo()
    // Logic op: bitwise operation instead of blending
    // Use when: specific integer framebuffer effects
    // Mutually exclusive with blendEnable (per attachment)
    .setLogicOpEnable(VK_FALSE)                              // Default
    .setLogicOp(vk::LogicOp::eCopy)                          // Default (ignored if disabled)
    .setAttachments(colorBlendAttachments)                   // From fragment shader (output count)
    .setBlendConstants({0.0f, 0.0f, 0.0f, 0.0f});            // Non-obvious choice (for eConstantColor blend factor)

// ============================================================================
// DYNAMIC STATE
// ============================================================================

// Dynamic state: values set at command buffer record time instead of pipeline creation
// Use when: viewport changes (resize), scissor changes, per-draw variations
// Trade-off: slightly more CPU overhead, but fewer pipeline objects

std::vector<vk::DynamicState> dynamicStates = {
    vk::DynamicState::eViewport,                             // Default (almost always dynamic)
    vk::DynamicState::eScissor,                              // Default (almost always dynamic)
};

// Optional dynamic states:

// Dynamic line width (debug visualization, CAD)
// dynamicStates.push_back(vk::DynamicState::eLineWidth);

// Dynamic depth bias (different bias per shadow cascade)
// dynamicStates.push_back(vk::DynamicState::eDepthBias);

// Dynamic blend constants (fade effects, UI)
// dynamicStates.push_back(vk::DynamicState::eBlendConstants);

// Dynamic stencil (portals, masks)
// dynamicStates.push_back(vk::DynamicState::eStencilCompareMask);
// dynamicStates.push_back(vk::DynamicState::eStencilWriteMask);
// dynamicStates.push_back(vk::DynamicState::eStencilReference);

// Dynamic depth bounds (light volumes)
// dynamicStates.push_back(vk::DynamicState::eDepthBounds);

// VK_EXT_extended_dynamic_state (Vulkan 1.3 core):
// Allows changing more state without pipeline recreation
// dynamicStates.push_back(vk::DynamicState::eCullMode);
// dynamicStates.push_back(vk::DynamicState::eFrontFace);
// dynamicStates.push_back(vk::DynamicState::ePrimitiveTopology);
// dynamicStates.push_back(vk::DynamicState::eDepthTestEnable);
// dynamicStates.push_back(vk::DynamicState::eDepthWriteEnable);
// dynamicStates.push_back(vk::DynamicState::eDepthCompareOp);
// dynamicStates.push_back(vk::DynamicState::eStencilTestEnable);

auto dynamicState = vk::PipelineDynamicStateCreateInfo()
    .setDynamicStates(dynamicStates);                        // Non-obvious choice

// ============================================================================
// PIPELINE CACHE
// ============================================================================

// Use when: faster pipeline creation on subsequent runs, shipping games
// Pipeline cache stores compiled shaders/state
// Load from disk on startup, save on shutdown

std::vector<uint8_t> cacheData;
// cacheData = loadFromDisk("pipeline_cache.bin");           // User responsibility

auto cacheInfo = vk::PipelineCacheCreateInfo()
    .setInitialDataSize(cacheData.size())                    // User-provided (0 if no cache)
    .setPInitialData(cacheData.empty() ? nullptr : cacheData.data()); // User-provided

pipelineCache = device.createPipelineCache(cacheInfo);

// After app shutdown:
// auto newCacheData = device.getPipelineCacheData(pipelineCache);
// saveToDisk("pipeline_cache.bin", newCacheData);           // User responsibility

// ============================================================================
// GRAPHICS PIPELINE
// ============================================================================

auto pipelineInfo = vk::GraphicsPipelineCreateInfo()
    // Allow deriving child pipelines from this one
    // Use when: many similar pipelines (different blend modes, etc.)
    .setFlags(vk::PipelineCreateFlagBits::eAllowDerivatives) // Non-obvious choice (optional)

    .setStages(shaderStages)                                 // From shaders
    .setPVertexInputState(&vertexInputInfo)                  // From vertex shader reflection
    .setPInputAssemblyState(&inputAssembly)                  // Non-obvious choice (topology)
    .setPTessellationState(&tessellationState)               // From tess control shader (nullptr if no tessellation)
    .setPViewportState(&viewportState)                       // From RenderTarget / dynamic
    .setPRasterizationState(&rasterizer)                     // Mostly defaults + non-obvious choices
    .setPMultisampleState(&multisampling)                    // From RenderTarget + non-obvious choices
    .setPDepthStencilState(&depthStencil)                    // Non-obvious choices
    .setPColorBlendState(&colorBlending)                     // Non-obvious choices + fragment output count
    .setPDynamicState(&dynamicState)                         // Non-obvious choice
    .setLayout(pipelineLayout)                               // From shader reflection
    .setRenderPass(renderPass)                               // From RenderTarget
    .setSubpass(0)                                           // Default (or index if multi-subpass)
    .setBasePipelineHandle(nullptr)                          // Default (or parent for derivative)
    .setBasePipelineIndex(-1);                               // Default

auto result = device.createGraphicsPipeline(pipelineCache, pipelineInfo);
if (result.result != vk::Result::eSuccess) {
    throw std::runtime_error("Failed to create graphics pipeline");
}
graphicsPipeline = result.value;

// ============================================================================
// PIPELINE DERIVATIVES (optional)
// ============================================================================

// Use when: many similar pipelines, minor variations
// Faster creation than from scratch (driver can share compiled code)

// Example: wireframe variant
auto wireframeRasterizer = rasterizer;
wireframeRasterizer.setPolygonMode(vk::PolygonMode::eLine); // Non-obvious choice

auto wireframePipelineInfo = pipelineInfo;
wireframePipelineInfo
    .setFlags(vk::PipelineCreateFlagBits::eDerivative)       // Non-obvious choice
    .setBasePipelineHandle(graphicsPipeline)                 // From parent pipeline
    .setBasePipelineIndex(-1)
    .setPRasterizationState(&wireframeRasterizer);

auto wireframeResult = device.createGraphicsPipeline(pipelineCache, wireframePipelineInfo);
wireframePipeline = wireframeResult.value;

// Example: transparent variant (alpha blending enabled)
auto transparentBlendAttachment = colorBlendAttachment0;
transparentBlendAttachment.setBlendEnable(VK_TRUE);          // Non-obvious choice

std::array transparentBlendAttachments = {transparentBlendAttachment};
auto transparentColorBlending = colorBlending;
transparentColorBlending.setAttachments(transparentBlendAttachments);

auto transparentDepthStencil = depthStencil;
transparentDepthStencil.setDepthWriteEnable(VK_FALSE);       // Non-obvious choice (don't write depth for transparent)

auto transparentPipelineInfo = pipelineInfo;
transparentPipelineInfo
    .setFlags(vk::PipelineCreateFlagBits::eDerivative)
    .setBasePipelineHandle(graphicsPipeline)
    .setBasePipelineIndex(-1)
    .setPDepthStencilState(&transparentDepthStencil)
    .setPColorBlendState(&transparentColorBlending);

auto transparentResult = device.createGraphicsPipeline(pipelineCache, transparentPipelineInfo);
transparentPipeline = transparentResult.value;

// Cleanup shader modules (no longer needed after pipeline creation)
device.destroyShaderModule(vertShaderModule);
device.destroyShaderModule(hullShaderModule);
device.destroyShaderModule(domainShaderModule);
device.destroyShaderModule(geomShaderModule);
device.destroyShaderModule(fragShaderModule);

// ============================================================================
// FRAMEBUFFERS
// ============================================================================

swapchainFramebuffers.resize(swapchainImageViews.size());

for (size_t i = 0; i < swapchainImageViews.size(); i++) {
    // Attachment order must match render pass attachment order
    std::array framebufferAttachments = {
        swapchainImageViews[i],                              // From RenderTarget (color 0)
        gbufferNormalImageView,                              // From RenderTarget (color 1 - MRT)
        gbufferAlbedoImageView,                              // From RenderTarget (color 2 - MRT)
        depthImageView                                       // From RenderTarget (depth)
    };

    auto framebufferInfo = vk::FramebufferCreateInfo()
        .setRenderPass(renderPass)                           // From RenderTarget
        .setAttachments(framebufferAttachments)              // From RenderTarget
        .setWidth(swapchainExtent.width)                     // From RenderTarget
        .setHeight(swapchainExtent.height)                   // From RenderTarget
        .setLayers(1);                                       // Default (>1 for layered rendering / VR)

    swapchainFramebuffers[i] = device.createFramebuffer(framebufferInfo);
}

// ============================================================================
// COMMAND POOL
// ============================================================================

auto poolInfo = vk::CommandPoolCreateInfo()
    // eResetCommandBuffer: allow individual buffer reset (most common)
    // eTransient: short-lived buffers (per-frame recording)
    .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer) // Default
    .setQueueFamilyIndex(graphicsQueueFamily);               // From VulkanContext

commandPool = device.createCommandPool(poolInfo);

// ============================================================================
// COMMAND BUFFERS
// ============================================================================

auto allocInfo = vk::CommandBufferAllocateInfo()
    .setCommandPool(commandPool)
    .setLevel(vk::CommandBufferLevel::ePrimary)              // Default (eSecondary for multi-threaded recording)
    .setCommandBufferCount(MAX_FRAMES_IN_FLIGHT);            // Non-obvious choice (frame count)

commandBuffers = device.allocateCommandBuffers(allocInfo);

// ============================================================================
// SYNC OBJECTS
// ============================================================================

imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);       // Non-obvious choice (frame count)
renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

auto semaphoreInfo = vk::SemaphoreCreateInfo();              // Default (no flags)

auto fenceInfo = vk::FenceCreateInfo()
    .setFlags(vk::FenceCreateFlagBits::eSignaled);           // Default (start signaled so first wait succeeds)

for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    imageAvailableSemaphores[i] = device.createSemaphore(semaphoreInfo);
    renderFinishedSemaphores[i] = device.createSemaphore(semaphoreInfo);
    inFlightFences[i] = device.createFence(fenceInfo);
}

// ============================================================================
// DESCRIPTOR POOL
// ============================================================================

// Pool sizes: count how many of each descriptor type across all sets * frames
std::array poolSizes = {
    vk::DescriptorPoolSize()
        .setType(vk::DescriptorType::eUniformBuffer)         // From shader reflection
        .setDescriptorCount(MAX_FRAMES_IN_FLIGHT * 2),       // From shader reflection * frames * sets

    vk::DescriptorPoolSize()
        .setType(vk::DescriptorType::eCombinedImageSampler)  // From shader reflection
        .setDescriptorCount(MAX_FRAMES_IN_FLIGHT * 4),       // From shader reflection * frames * sets

    vk::DescriptorPoolSize()
        .setType(vk::DescriptorType::eStorageBuffer)         // From shader reflection
        .setDescriptorCount(MAX_FRAMES_IN_FLIGHT * 2)        // From shader reflection * frames * sets
};

auto descriptorPoolInfo = vk::DescriptorPoolCreateInfo()
    .setPoolSizes(poolSizes)                                 // From shader reflection
    .setMaxSets(MAX_FRAMES_IN_FLIGHT * 2);                   // From shader reflection (sets per frame * frames)

descriptorPool = device.createDescriptorPool(descriptorPoolInfo);

// ============================================================================
// DESCRIPTOR SETS
// ============================================================================

// Allocate set 0 for each frame
std::vector<vk::DescriptorSetLayout> set0Layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout0);
auto set0AllocInfo = vk::DescriptorSetAllocateInfo()
    .setDescriptorPool(descriptorPool)
    .setSetLayouts(set0Layouts);                             // From shader reflection

descriptorSets0 = device.allocateDescriptorSets(set0AllocInfo);

// Allocate set 1 for each frame (or per-material)
std::vector<vk::DescriptorSetLayout> set1Layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout1);
auto set1AllocInfo = vk::DescriptorSetAllocateInfo()
    .setDescriptorPool(descriptorPool)
    .setSetLayouts(set1Layouts);

descriptorSets1 = device.allocateDescriptorSets(set1AllocInfo);

// Update descriptor sets
for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    // Set 0: per-frame data
    auto uboInfo = vk::DescriptorBufferInfo()
        .setBuffer(uniformBuffers[i])                        // User-provided resource
        .setOffset(0)                                        // User-provided
        .setRange(sizeof(UniformBufferObject));              // From shader reflection (size)


    auto storageInfo = vk::DescriptorBufferInfo()
        .setBuffer(storageBuffer)                            // User-provided resource
        .setOffset(0)
        .setRange(VK_WHOLE_SIZE);                            // From shader reflection (or whole buffer)

    // Set 1: per-material data
    auto albedoImageInfo = vk::DescriptorImageInfo()
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal) // Default for sampled image
        .setImageView(albedoImageView)                       // User-provided resource
        .setSampler(textureSampler);                         // User-provided resource

    auto normalImageInfo = vk::DescriptorImageInfo()
        .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
        .setImageView(normalImageView)                       // User-provided resource
        .setSampler(textureSampler);                         // User-provided resource

    std::array descriptorWrites = {
        // Set 0, binding 0: UBO
        vk::WriteDescriptorSet()
            .setDstSet(descriptorSets0[i])
            .setDstBinding(0)                                // From shader reflection
            .setDstArrayElement(0)                           // Default
            .setDescriptorType(vk::DescriptorType::eUniformBuffer) // From shader reflection
            .setDescriptorCount(1)                           // From shader reflection
            .setPBufferInfo(&uboInfo),

        // Set 0, binding 2: storage buffer
        vk::WriteDescriptorSet()
            .setDstSet(descriptorSets0[i])
            .setDstBinding(2)                                // From shader reflection
            .setDstArrayElement(0)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer) // From shader reflection
            .setDescriptorCount(1)
            .setPBufferInfo(&storageInfo),

        // Set 1, binding 0: albedo texture
        vk::WriteDescriptorSet()
            .setDstSet(descriptorSets1[i])
            .setDstBinding(0)                                // From shader reflection
            .setDstArrayElement(0)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler) // From shader reflection
            .setDescriptorCount(1)
            .setPImageInfo(&albedoImageInfo),

        // Set 1, binding 1: normal map
        vk::WriteDescriptorSet()
            .setDstSet(descriptorSets1[i])
            .setDstBinding(1)                                // From shader reflection
            .setDstArrayElement(0)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setDescriptorCount(1)
            .setPImageInfo(&normalImageInfo)
    };

    device.updateDescriptorSets(descriptorWrites, {});
}

// ============================================================================
// RENDER LOOP
// ============================================================================

while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // Wait for previous frame
    device.waitForFences(inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    // Acquire next image
    auto [acquireResult, imageIndex] = device.acquireNextImageKHR(
        swapchain, UINT64_MAX, imageAvailableSemaphores[currentFrame], nullptr);

    if (acquireResult == vk::Result::eErrorOutOfDateKHR) {
        recreateSwapchain();                                 // From RenderTarget (resize handling)
        continue;
    }

    device.resetFences(inFlightFences[currentFrame]);

    // Update uniforms
    updateUniformBuffer(currentFrame);                       // User logic

    // Record command buffer
    commandBuffers[currentFrame].reset();

    auto beginInfo = vk::CommandBufferBeginInfo()
        // eOneTimeSubmit: buffer will be rerecorded before next submit
        // eRenderPassContinue: secondary buffer entirely within render pass
        // eSimultaneousUse: can be submitted while already pending
        .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit); // Non-obvious choice (optional)

    commandBuffers[currentFrame].begin(beginInfo);

    // Clear values (one per attachment, order matches render pass)
    std::array<vk::ClearValue, 4> clearValues = {
        vk::ClearValue().setColor({0.0f, 0.0f, 0.0f, 1.0f}), // Non-obvious choice (attachment 0 clear color)
        vk::ClearValue().setColor({0.0f, 0.0f, 0.0f, 0.0f}), // Non-obvious choice (attachment 1 - normals)
        vk::ClearValue().setColor({0.0f, 0.0f, 0.0f, 1.0f}), // Non-obvious choice (attachment 2 - albedo)
        vk::ClearValue().setDepthStencil({1.0f, 0})          // Default (max depth, 0 stencil; 0.0f for reverse-Z)
    };

    auto renderPassBeginInfo = vk::RenderPassBeginInfo()
        .setRenderPass(renderPass)                           // From RenderTarget
        .setFramebuffer(swapchainFramebuffers[imageIndex])   // From RenderTarget
        .setRenderArea({{0, 0}, swapchainExtent})            // From RenderTarget
        .setClearValues(clearValues);                        // Non-obvious choice

    // eInline: commands recorded directly in primary buffer
    // eSecondaryCommandBuffers: execute secondary buffers (multi-threaded recording)
    commandBuffers[currentFrame].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

    // Bind pipeline
    commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

    // Set dynamic state
    auto dynamicViewport = vk::Viewport()
        .setX(0.0f)
        .setY(0.0f)
        .setWidth(static_cast<float>(swapchainExtent.width))  // From RenderTarget
        .setHeight(static_cast<float>(swapchainExtent.height))
        .setMinDepth(0.0f)
        .setMaxDepth(1.0f);

    commandBuffers[currentFrame].setViewport(0, dynamicViewport);
    commandBuffers[currentFrame].setScissor(0, vk::Rect2D{{0, 0}, swapchainExtent});

    // Optional: set other dynamic state if enabled
    // commandBuffers[currentFrame].setLineWidth(2.0f);
    // commandBuffers[currentFrame].setDepthBias(1.25f, 0.0f, 1.75f);
    // commandBuffers[currentFrame].setBlendConstants({1.0f, 1.0f, 1.0f, 1.0f});
    // commandBuffers[currentFrame].setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, 1);

    // Bind vertex buffers
    std::array<vk::Buffer, 2> vertexBuffers = {vertexBuffer, instanceBuffer};
    std::array<vk::DeviceSize, 2> offsets = {0, 0};
    commandBuffers[currentFrame].bindVertexBuffers(0, vertexBuffers, offsets); // User-provided

    // Bind index buffer
    commandBuffers[currentFrame].bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32); // User-provided

    // Bind descriptor sets
    std::array boundDescriptorSets = {descriptorSets0[currentFrame], descriptorSets1[currentFrame]};
    commandBuffers[currentFrame].bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        pipelineLayout,                                      // From shader reflection
        0,                                                   // First set index
        boundDescriptorSets,                                 // User-provided
        {});                                                 // Dynamic offsets (user-provided if any)

    // Push constants
    PushConstants pushConstants = {
        .model = modelMatrix,                                // User-provided
        .objectId = objectId                                 // User-provided
    };
    commandBuffers[currentFrame].pushConstants(
        pipelineLayout,                                      // From shader reflection
        vk::ShaderStageFlagBits::eVertex |
        vk::ShaderStageFlagBits::eFragment,                  // From shader reflection
        0,                                                   // From shader reflection (offset)
        sizeof(PushConstants),                               // From shader reflection (size)
        &pushConstants);                                     // User-provided

    // Draw calls
    // Indexed draw (most common)
    commandBuffers[currentFrame].drawIndexed(
        indexCount,                                          // User-provided
        instanceCount,                                       // User-provided (1 for non-instanced)
        0,                                                   // First index
        0,                                                   // Vertex offset
        0);                                                  // First instance

    // Non-indexed draw
    // commandBuffers[currentFrame].draw(vertexCount, instanceCount, firstVertex, firstInstance);

    // Indirect draw (GPU-driven rendering)
    // commandBuffers[currentFrame].drawIndexedIndirect(indirectBuffer, offset, drawCount, stride);

    // Multi-draw indirect (batch multiple draw calls)
    // commandBuffers[currentFrame].drawIndexedIndirectCount(
    //     indirectBuffer, offset, countBuffer, countOffset, maxDrawCount, stride);

    commandBuffers[currentFrame].endRenderPass();
    commandBuffers[currentFrame].end();

    // Submit
    std::array waitSemaphores = {imageAvailableSemaphores[currentFrame]};
    std::array<vk::PipelineStageFlags, 1> waitStages = {
        vk::PipelineStageFlagBits::eColorAttachmentOutput    // Default
    };
    std::array signalSemaphores = {renderFinishedSemaphores[currentFrame]};

    auto submitInfo = vk::SubmitInfo()
        .setWaitSemaphores(waitSemaphores)
        .setWaitDstStageMask(waitStages)
        .setCommandBuffers(commandBuffers[currentFrame])
        .setSignalSemaphores(signalSemaphores);

    graphicsQueue.submit(submitInfo, inFlightFences[currentFrame]);

    // Present
    auto presentInfo = vk::PresentInfoKHR()
        .setWaitSemaphores(signalSemaphores)
        .setSwapchains(swapchain)                            // From RenderTarget
        .setImageIndices(imageIndex);

    auto presentResult = presentQueue.presentKHR(presentInfo);

    if (presentResult == vk::Result::eErrorOutOfDateKHR ||
        presentResult == vk::Result::eSuboptimalKHR ||
        framebufferResized) {
        framebufferResized = false;
        recreateSwapchain();                                 // From RenderTarget
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

device.waitIdle();

// ============================================================================
// CLEANUP (in reverse creation order)
// ============================================================================

// Save pipeline cache before destroying
auto cacheDataToSave = device.getPipelineCacheData(pipelineCache);
// saveToDisk("pipeline_cache.bin", cacheDataToSave);        // User responsibility

for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    device.destroySemaphore(renderFinishedSemaphores[i]);
    device.destroySemaphore(imageAvailableSemaphores[i]);
    device.destroyFence(inFlightFences[i]);
}

device.destroyCommandPool(commandPool);

for (auto framebuffer : swapchainFramebuffers) {
    device.destroyFramebuffer(framebuffer);
}

device.destroyPipeline(transparentPipeline);
device.destroyPipeline(wireframePipeline);
device.destroyPipeline(graphicsPipeline);
device.destroyPipelineCache(pipelineCache);
device.destroyPipelineLayout(pipelineLayout);
device.destroyDescriptorPool(descriptorPool);
device.destroyDescriptorSetLayout(descriptorSetLayout1);
device.destroyDescriptorSetLayout(descriptorSetLayout0);
device.destroyRenderPass(renderPass);


int main()
{
	VulkanContext context;
	RenderTarget target{context, surface, ...};
	auto render_pipeline = RenderPipeline::create(context, target, shaders, cfg);
	// Somehow set up buffers for the various descriptors
	// Write some initial stuff to buffers

	while (should_render)
	{
		update_ubos();
		render_pipeline.draw(my_verts, ...);
	}
}