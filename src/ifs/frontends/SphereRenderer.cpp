#include <ifs/frontends/SphereRenderer.hpp>
#include <ifs/Logger.hpp>
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <format>
#include <glm/gtc/matrix_transform.hpp>
#include <unordered_map>

namespace ifs {

SphereRenderer::SphereRenderer(const VulkanContext& context, vk::Device device)
    : m_context(&context)
    , m_device(device)
    , m_render_pass(nullptr)
    , m_extent{}
    , m_pipeline_layout(nullptr)
    , m_graphics_pipeline(nullptr)
    , m_descriptor_layout(nullptr)
    , m_descriptor_pool(nullptr)
    , m_descriptor_set(nullptr)
    , m_vertex_buffer(nullptr)
    , m_vertex_memory(nullptr)
    , m_index_buffer(nullptr)
    , m_index_memory(nullptr)
    , m_view_buffer(nullptr)
    , m_view_memory(nullptr)
    , m_graphics_command_pool(nullptr)
{}

SphereRenderer::~SphereRenderer() {
    // Cleanup graphics infrastructure (Phase 3: frontend owns these)
    for (auto fence : m_in_flight_fences) {
        m_device.destroyFence(fence);
    }
    for (auto sem : m_render_finished_semaphores) {
        m_device.destroySemaphore(sem);
    }
    if (m_graphics_command_pool) {
        m_device.destroyCommandPool(m_graphics_command_pool);
    }

    // Cleanup view buffer
    if (m_view_memory) {
        if (m_view_mapped) {
            m_device.unmapMemory(m_view_memory);
        }
        m_device.freeMemory(m_view_memory);
    }
    if (m_view_buffer) {
        m_device.destroyBuffer(m_view_buffer);
    }

    // Cleanup sphere buffers
    if (m_index_memory) m_device.freeMemory(m_index_memory);
    if (m_index_buffer) m_device.destroyBuffer(m_index_buffer);
    if (m_vertex_memory) m_device.freeMemory(m_vertex_memory);
    if (m_vertex_buffer) m_device.destroyBuffer(m_vertex_buffer);

    // Cleanup descriptor sets
    if (m_descriptor_pool) m_device.destroyDescriptorPool(m_descriptor_pool);
    if (m_descriptor_layout) m_device.destroyDescriptorSetLayout(m_descriptor_layout);

    // Cleanup pipeline
    if (m_graphics_pipeline) m_device.destroyPipeline(m_graphics_pipeline);
    if (m_pipeline_layout) m_device.destroyPipelineLayout(m_pipeline_layout);
}

void SphereRenderer::generate_sphere_mesh(uint32_t subdivisions) {
    // Generate an icosphere using subdivision
    // Start with an icosahedron and subdivide

    const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;

    // Initial icosahedron vertices
    std::vector<glm::vec3> positions = {
        glm::normalize(glm::vec3(-1,  t,  0)),
        glm::normalize(glm::vec3( 1,  t,  0)),
        glm::normalize(glm::vec3(-1, -t,  0)),
        glm::normalize(glm::vec3( 1, -t,  0)),
        glm::normalize(glm::vec3( 0, -1,  t)),
        glm::normalize(glm::vec3( 0,  1,  t)),
        glm::normalize(glm::vec3( 0, -1, -t)),
        glm::normalize(glm::vec3( 0,  1, -t)),
        glm::normalize(glm::vec3( t,  0, -1)),
        glm::normalize(glm::vec3( t,  0,  1)),
        glm::normalize(glm::vec3(-t,  0, -1)),
        glm::normalize(glm::vec3(-t,  0,  1))
    };

    // Initial icosahedron faces
    std::vector<uint32_t> indices = {
        0, 11, 5,   0, 5, 1,    0, 1, 7,    0, 7, 10,   0, 10, 11,
        1, 5, 9,    5, 11, 4,   11, 10, 2,  10, 7, 6,   7, 1, 8,
        3, 9, 4,    3, 4, 2,    3, 2, 6,    3, 6, 8,    3, 8, 9,
        4, 9, 5,    2, 4, 11,   6, 2, 10,   8, 6, 7,    9, 8, 1
    };

    // Subdivide
    for (uint32_t i = 0; i < subdivisions; i++) {
        std::vector<uint32_t> new_indices;
        std::unordered_map<uint64_t, uint32_t> midpoint_cache;

        auto get_midpoint = [&](uint32_t i1, uint32_t i2) -> uint32_t {
            // Order indices to ensure consistent cache key
            if (i1 > i2) std::swap(i1, i2);
            uint64_t key = (static_cast<uint64_t>(i1) << 32) | i2;

            auto it = midpoint_cache.find(key);
            if (it != midpoint_cache.end()) {
                return it->second;
            }

            // Create new midpoint
            glm::vec3 mid = glm::normalize(positions[i1] + positions[i2]);
            uint32_t idx = static_cast<uint32_t>(positions.size());
            positions.push_back(mid);
            midpoint_cache[key] = idx;
            return idx;
        };

        // Subdivide each triangle into 4 triangles
        for (size_t j = 0; j < indices.size(); j += 3) {
            uint32_t v1 = indices[j];
            uint32_t v2 = indices[j + 1];
            uint32_t v3 = indices[j + 2];

            uint32_t a = get_midpoint(v1, v2);
            uint32_t b = get_midpoint(v2, v3);
            uint32_t c = get_midpoint(v3, v1);

            new_indices.insert(new_indices.end(), {
                v1, a, c,
                v2, b, a,
                v3, c, b,
                a, b, c
            });
        }

        indices = std::move(new_indices);
    }

    // Build final vertex buffer with positions and normals
    m_sphere_vertices.clear();
    m_sphere_vertices.reserve(positions.size());
    for (const auto& pos : positions) {
        m_sphere_vertices.push_back({pos, pos}); // normal = position for unit sphere
    }

    m_sphere_indices = std::move(indices);

    Logger::instance().info("Generated sphere mesh: {} vertices, {} indices",
        m_sphere_vertices.size(), m_sphere_indices.size());
}

std::expected<void, std::string> SphereRenderer::create_sphere_buffers() {
    auto physical_device = m_context->physical_device();

    // Create vertex buffer
    vk::DeviceSize vertex_size = sizeof(Vertex) * m_sphere_vertices.size();

    auto vertex_buffer_info = vk::BufferCreateInfo()
        .setSize(vertex_size)
        .setUsage(vk::BufferUsageFlagBits::eVertexBuffer)
        .setSharingMode(vk::SharingMode::eExclusive);

	auto vertex_buffer_res = m_device.createBuffer(vertex_buffer_info);
	if (vertex_buffer_res.result != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to create vertex buffer: {}", to_string(vertex_buffer_res.result)));
	}
	m_vertex_buffer = vertex_buffer_res.value;

    auto vertex_mem_reqs = m_device.getBufferMemoryRequirements(m_vertex_buffer);
    auto vertex_mem_props = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

    auto find_memory_type = [&](uint32_t type_filter, vk::MemoryPropertyFlags properties) -> std::optional<uint32_t> {
        auto mem_props = physical_device.getMemoryProperties();
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
            if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        return std::nullopt;
    };

    auto vertex_mem_type = find_memory_type(vertex_mem_reqs.memoryTypeBits, vertex_mem_props);
    if (!vertex_mem_type) {
        return std::unexpected("Failed to find suitable memory type for vertex buffer");
    }

    auto vertex_alloc_info = vk::MemoryAllocateInfo()
        .setAllocationSize(vertex_mem_reqs.size)
        .setMemoryTypeIndex(*vertex_mem_type);

	auto vertex_mem_res = m_device.allocateMemory(vertex_alloc_info);
	if (vertex_mem_res.result != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to allocate vertex memory: {}", to_string(vertex_mem_res.result)));
	}
	m_vertex_memory = vertex_mem_res.value;
	auto vertex_bind_res = m_device.bindBufferMemory(m_vertex_buffer, m_vertex_memory, 0);
	if (vertex_bind_res != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to bind vertex memory: {}", to_string(vertex_bind_res)));
	}

	// Upload vertex data
	auto data_vert_res = m_device.mapMemory(m_vertex_memory, 0, vertex_size);
	CHECK_VK_RESULT(data_vert_res, "Failed to map vert memory {}");
	auto data_vert = data_vert_res.value;
	std::memcpy(data_vert, m_sphere_vertices.data(), vertex_size);
	m_device.unmapMemory(m_vertex_memory);

    // Create index buffer
    vk::DeviceSize index_size = sizeof(uint32_t) * m_sphere_indices.size();

    auto index_buffer_info = vk::BufferCreateInfo()
        .setSize(index_size)
        .setUsage(vk::BufferUsageFlagBits::eIndexBuffer)
        .setSharingMode(vk::SharingMode::eExclusive);

	auto index_buffer_res = m_device.createBuffer(index_buffer_info);
	if (index_buffer_res.result != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to create index buffer: {}", to_string(index_buffer_res.result)));
	}
	m_index_buffer = index_buffer_res.value;

    auto index_mem_reqs = m_device.getBufferMemoryRequirements(m_index_buffer);
    auto index_mem_type = find_memory_type(index_mem_reqs.memoryTypeBits, vertex_mem_props);
    if (!index_mem_type) {
        return std::unexpected("Failed to find suitable memory type for index buffer");
    }

    auto index_alloc_info = vk::MemoryAllocateInfo()
        .setAllocationSize(index_mem_reqs.size)
        .setMemoryTypeIndex(*index_mem_type);

	auto index_mem_res = m_device.allocateMemory(index_alloc_info);
	if (index_mem_res.result != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to allocate index memory: {}", to_string(index_mem_res.result)));
	}
	m_index_memory = index_mem_res.value;
	auto index_bind_res = m_device.bindBufferMemory(m_index_buffer, m_index_memory, 0);
	if (index_bind_res != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to bind index memory: {}", to_string(index_bind_res)));
	}

	// Upload index data
	auto data_index_res = m_device.mapMemory(m_index_memory, 0, index_size);
	CHECK_VK_RESULT(data_index_res, "Failed to map index memory {}");
	auto data_index = data_index_res.value;
	std::memcpy(data_index, m_sphere_indices.data(), index_size);
	m_device.unmapMemory(m_index_memory);

    return {};
}

std::expected<void, std::string> SphereRenderer::create_descriptor_layout() {
    // Binding 0: View parameters (uniform buffer)
    auto view_binding = vk::DescriptorSetLayoutBinding()
        .setBinding(0)
        .setDescriptorType(vk::DescriptorType::eUniformBuffer)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);

    // Binding 1: Particle positions (storage buffer)
    auto particle_binding = vk::DescriptorSetLayoutBinding()
        .setBinding(1)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setDescriptorCount(1)
        .setStageFlags(vk::ShaderStageFlagBits::eVertex);

    std::array bindings = {view_binding, particle_binding};

    auto layout_info = vk::DescriptorSetLayoutCreateInfo()
        .setBindings(bindings);

	auto layout_res = m_device.createDescriptorSetLayout(layout_info);
	if (layout_res.result != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to create descriptor layout: {}", to_string(layout_res.result)));
	}
	m_descriptor_layout = layout_res.value;

    return {};
}

std::expected<void, std::string> SphereRenderer::create_pipeline() {
    // Pipeline layout
    auto pipeline_layout_info = vk::PipelineLayoutCreateInfo()
        .setSetLayouts(m_descriptor_layout);

	auto pipeline_layout_res = m_device.createPipelineLayout(pipeline_layout_info);
	if (pipeline_layout_res.result != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to create pipeline layout: {}", to_string(pipeline_layout_res.result)));
	}
	m_pipeline_layout = pipeline_layout_res.value;

    // Shader stages
    auto vert_stage = vk::PipelineShaderStageCreateInfo()
        .setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(m_vertex_shader->get_shader_module())
        .setPName("main");

    auto frag_stage = vk::PipelineShaderStageCreateInfo()
        .setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(m_fragment_shader->get_shader_module())
        .setPName("main");

    std::array shader_stages = {vert_stage, frag_stage};

    // Vertex input: per-vertex (position, normal) and per-instance (nothing, we read from SSBO)
    auto position_attrib = vk::VertexInputAttributeDescription()
        .setLocation(0)
        .setBinding(0)
        .setFormat(vk::Format::eR32G32B32Sfloat)
        .setOffset(offsetof(Vertex, position));

    auto normal_attrib = vk::VertexInputAttributeDescription()
        .setLocation(1)
        .setBinding(0)
        .setFormat(vk::Format::eR32G32B32Sfloat)
        .setOffset(offsetof(Vertex, normal));

    std::array attributes = {position_attrib, normal_attrib};

    auto vertex_binding = vk::VertexInputBindingDescription()
        .setBinding(0)
        .setStride(sizeof(Vertex))
        .setInputRate(vk::VertexInputRate::eVertex);

    auto vertex_input = vk::PipelineVertexInputStateCreateInfo()
        .setVertexBindingDescriptions(vertex_binding)
        .setVertexAttributeDescriptions(attributes);

    // Input assembly
    auto input_assembly = vk::PipelineInputAssemblyStateCreateInfo()
        .setTopology(vk::PrimitiveTopology::eTriangleList)
        .setPrimitiveRestartEnable(false);

    // Viewport and scissor (dynamic)
    auto viewport_state = vk::PipelineViewportStateCreateInfo()
        .setViewportCount(1)
        .setScissorCount(1);

    // Rasterization
    auto rasterizer = vk::PipelineRasterizationStateCreateInfo()
        .setDepthClampEnable(false)
        .setRasterizerDiscardEnable(false)
        .setPolygonMode(vk::PolygonMode::eFill)
        .setLineWidth(1.0f)
        .setCullMode(vk::CullModeFlagBits::eNone)
        .setFrontFace(vk::FrontFace::eCounterClockwise)  // Standard winding order
        .setDepthBiasEnable(false);

    // Multisampling
    auto multisampling = vk::PipelineMultisampleStateCreateInfo()
        .setSampleShadingEnable(false)
        .setRasterizationSamples(vk::SampleCountFlagBits::e1);

    // Depth and stencil
    auto depth_stencil = vk::PipelineDepthStencilStateCreateInfo()
        .setDepthTestEnable(true)
        .setDepthWriteEnable(true)
        .setDepthCompareOp(vk::CompareOp::eLess)
        .setDepthBoundsTestEnable(false)
        .setStencilTestEnable(false);

    // Color blending
    auto color_blend_attachment = vk::PipelineColorBlendAttachmentState()
        .setColorWriteMask(
            vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA
        )
        .setBlendEnable(false);

    auto color_blending = vk::PipelineColorBlendStateCreateInfo()
        .setLogicOpEnable(false)
        .setAttachments(color_blend_attachment);

    // Dynamic state
    std::array dynamic_states = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    auto dynamic_state = vk::PipelineDynamicStateCreateInfo()
        .setDynamicStates(dynamic_states);

    // Create pipeline
    auto pipeline_info = vk::GraphicsPipelineCreateInfo()
        .setStages(shader_stages)
        .setPVertexInputState(&vertex_input)
        .setPInputAssemblyState(&input_assembly)
        .setPViewportState(&viewport_state)
        .setPRasterizationState(&rasterizer)
        .setPMultisampleState(&multisampling)
        .setPDepthStencilState(&depth_stencil)
        .setPColorBlendState(&color_blending)
        .setPDynamicState(&dynamic_state)
        .setLayout(m_pipeline_layout)
        .setRenderPass(m_render_pass)
        .setSubpass(0);

	auto pipeline_res = m_device.createGraphicsPipeline(nullptr, pipeline_info);
	if (pipeline_res.result != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to create graphics pipeline: {}", to_string(pipeline_res.result)));
	}
	m_graphics_pipeline = pipeline_res.value;

    return {};
}

std::expected<void, std::string> SphereRenderer::create_descriptor_set() {
    // Create descriptor pool
    std::array pool_sizes = {
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1)
    };

    auto pool_info = vk::DescriptorPoolCreateInfo()
        .setMaxSets(1)
        .setPoolSizes(pool_sizes);

	auto descriptor_pool_res = m_device.createDescriptorPool(pool_info);
	if (descriptor_pool_res.result != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to create descriptor pool: {}", to_string(descriptor_pool_res.result)));
	}
	m_descriptor_pool = descriptor_pool_res.value;

    // Allocate descriptor set
    auto alloc_info = vk::DescriptorSetAllocateInfo()
        .setDescriptorPool(m_descriptor_pool)
        .setSetLayouts(m_descriptor_layout);

	auto descriptor_set_res = m_device.allocateDescriptorSets(alloc_info);
	if (descriptor_set_res.result != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to allocate descriptor set: {}", to_string(descriptor_set_res.result)));
	}
	m_descriptor_set = descriptor_set_res.value[0];

    // Update view buffer binding (particle buffer will be updated per frame)
    auto view_buffer_info = vk::DescriptorBufferInfo()
        .setBuffer(m_view_buffer)
        .setOffset(0)
        .setRange(sizeof(ViewParams));

    auto view_write = vk::WriteDescriptorSet()
        .setDstSet(m_descriptor_set)
        .setDstBinding(0)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eUniformBuffer)
        .setBufferInfo(view_buffer_info);

    m_device.updateDescriptorSets(view_write, nullptr);

    return {};
}

std::expected<std::unique_ptr<SphereRenderer>, std::string> SphereRenderer::create(
    const VulkanContext& context,
    vk::Device device,
    vk::RenderPass render_pass,
    const vk::Extent2D& extent,
    uint32_t sphere_subdivisions
) {
    auto renderer = std::unique_ptr<SphereRenderer>(new SphereRenderer(context, device));
    renderer->m_render_pass = render_pass;
    renderer->m_extent = extent;

    // Generate sphere mesh
    renderer->generate_sphere_mesh(sphere_subdivisions);

    // Create sphere buffers
    if (auto result = renderer->create_sphere_buffers(); !result) {
        return std::unexpected(result.error());
    }

    // Load shaders
    auto vert_result = Shader::create_shader(device, "ifs_modular/frontends/sphere/sphere.vert.slang", "main");
    if (!vert_result) {
        return std::unexpected(std::format("Failed to load vertex shader: {}", vert_result.error()));
    }
    renderer->m_vertex_shader = std::make_unique<Shader>(std::move(*vert_result));

    auto frag_result = Shader::create_shader(device, "ifs_modular/frontends/sphere/sphere.frag.slang", "main");
    if (!frag_result) {
        return std::unexpected(std::format("Failed to load fragment shader: {}", frag_result.error()));
    }
    renderer->m_fragment_shader = std::make_unique<Shader>(std::move(*frag_result));

    // Create view parameter buffer
    auto buffer_info = vk::BufferCreateInfo()
        .setSize(sizeof(ViewParams))
        .setUsage(vk::BufferUsageFlagBits::eUniformBuffer)
        .setSharingMode(vk::SharingMode::eExclusive);

	auto view_buffer_res = device.createBuffer(buffer_info);
	if (view_buffer_res.result != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to create view buffer: {}", to_string(view_buffer_res.result)));
	}
	renderer->m_view_buffer = view_buffer_res.value;

    // Allocate and map view buffer memory
    auto mem_reqs = device.getBufferMemoryRequirements(renderer->m_view_buffer);
    auto mem_props = context.physical_device().getMemoryProperties();

    std::optional<uint32_t> memory_type;
    auto required_props = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((mem_reqs.memoryTypeBits & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & required_props) == required_props) {
            memory_type = i;
            break;
        }
    }

    if (!memory_type) {
        return std::unexpected("Failed to find suitable memory type for view buffer");
    }

    auto alloc_info = vk::MemoryAllocateInfo()
        .setAllocationSize(mem_reqs.size)
        .setMemoryTypeIndex(*memory_type);

	auto view_mem_res = device.allocateMemory(alloc_info);
	if (view_mem_res.result != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to allocate view memory: {}", to_string(view_mem_res.result)));
	}
	renderer->m_view_memory = view_mem_res.value;
	auto view_bind_res = device.bindBufferMemory(renderer->m_view_buffer, renderer->m_view_memory, 0);
	if (view_bind_res != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to bind view memory: {}", to_string(view_bind_res)));
	}
	auto view_map_res = device.mapMemory(renderer->m_view_memory, 0, sizeof(ViewParams));
	CHECK_VK_RESULT(view_map_res, "Failed to map memory for view {}");
	renderer->m_view_mapped = view_map_res.value;

    // Create descriptor layout
    if (auto result = renderer->create_descriptor_layout(); !result) {
        return std::unexpected(result.error());
    }

    // Create pipeline
    if (auto result = renderer->create_pipeline(); !result) {
        return std::unexpected(result.error());
    }

    // Create descriptor set
    if (auto result = renderer->create_descriptor_set(); !result) {
        return std::unexpected(result.error());
    }

    // Phase 3: Create graphics infrastructure
    auto pool_info = vk::CommandPoolCreateInfo()
        .setQueueFamilyIndex(context.queue_indices().graphics)
        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

	auto cmd_pool_res = device.createCommandPool(pool_info);
	if (cmd_pool_res.result != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to create command pool: {}", to_string(cmd_pool_res.result)));
	}
	renderer->m_graphics_command_pool = cmd_pool_res.value;

    // Create command buffers (one per swapchain image)
    // Note: We'll create initial command buffers, but handle_swapchain_recreation will resize if needed
    auto alloc_info_cmd = vk::CommandBufferAllocateInfo()
        .setCommandPool(renderer->m_graphics_command_pool)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(1);  // Placeholder, will be resized

	auto cmd_buffers_res = device.allocateCommandBuffers(alloc_info_cmd);
	if (cmd_buffers_res.result != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to allocate command buffers: {}", to_string(cmd_buffers_res.result)));
	}
	renderer->m_command_buffers = cmd_buffers_res.value;

    // Create synchronization objects
    for (size_t i = 0; i < SphereRenderer::MAX_FRAMES_IN_FLIGHT; i++) {
		auto fence_res = device.createFence({vk::FenceCreateFlagBits::eSignaled});
		if (fence_res.result != vk::Result::eSuccess)
		{
			return std::unexpected(std::format("Failed to create fence: {}", to_string(fence_res.result)));
		}
		renderer->m_in_flight_fences.push_back(fence_res.value);
		auto semaphore_res = device.createSemaphore({});
		if (semaphore_res.result != vk::Result::eSuccess)
		{
			return std::unexpected(std::format("Failed to create semaphore: {}", to_string(semaphore_res.result)));
		}
		renderer->m_render_finished_semaphores.push_back(semaphore_res.value);
    }

    renderer->m_images_in_flight.resize(1, nullptr);  // Placeholder

    Logger::instance().info("SphereRenderer created successfully");

    return renderer;
}

void SphereRenderer::resize(const vk::Extent2D& new_extent) {
    m_extent = new_extent;
}

void SphereRenderer::update_particle_buffer(vk::Buffer particle_buffer) {
    // Update particle buffer descriptor (only call this once, not every frame!)
    auto particle_buffer_info = vk::DescriptorBufferInfo()
        .setBuffer(particle_buffer)
        .setOffset(0)
        .setRange(VK_WHOLE_SIZE);

    auto particle_write = vk::WriteDescriptorSet()
        .setDstSet(m_descriptor_set)
        .setDstBinding(1)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setBufferInfo(particle_buffer_info);

    m_device.updateDescriptorSets(particle_write, nullptr);
}

void SphereRenderer::render(
    vk::CommandBuffer cmd,
    [[maybe_unused]] vk::Buffer particle_buffer,
    uint32_t particle_count,
    Camera& camera,
    const vk::Extent2D* extent
) {
    // Use provided extent or fallback to stored extent
    vk::Extent2D render_extent = extent ? *extent : m_extent;
    // Update view parameters
    ViewParams params{};
    params.view_projection = camera.view_projection_matrix();
    params.camera_pos = camera.position();
    params.sphere_radius = m_sphere_radius;
    params.light_dir = glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f));

    std::memcpy(m_view_mapped, &params, sizeof(ViewParams));

    // Set viewport and scissor
    // Use negative height to flip Y-axis (Vulkan convention)
    auto viewport = vk::Viewport()
        .setX(0.0f)
        .setY(static_cast<float>(render_extent.height))  // Start at bottom
        .setWidth(static_cast<float>(render_extent.width))
        .setHeight(-static_cast<float>(render_extent.height))  // Negative = Y-flip
        .setMinDepth(0.0f)
        .setMaxDepth(1.0f);

    auto scissor = vk::Rect2D()
        .setOffset({0, 0})
        .setExtent(render_extent);

    cmd.setViewport(0, viewport);
    cmd.setScissor(0, scissor);

    // Bind pipeline and draw instanced
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_graphics_pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipeline_layout, 0, m_descriptor_set, {});
    cmd.bindVertexBuffers(0, m_vertex_buffer, vk::DeviceSize{0});
    cmd.bindIndexBuffer(m_index_buffer, 0, vk::IndexType::eUint32);

    // Draw instanced: one sphere instance per particle
    cmd.drawIndexed(static_cast<uint32_t>(m_sphere_indices.size()), particle_count, 0, 0, 0);
}

vk::Semaphore SphereRenderer::render_frame(
    const FrameRenderInfo& info,
    vk::Queue graphics_queue
) {
    // Wait for frame-in-flight fence
    [[maybe_unused]] auto wait_result = m_device.waitForFences(m_in_flight_fences[info.current_frame], true, UINT64_MAX);

    // Wait for image if still in use
    if (m_images_in_flight[info.image_index]) {
        [[maybe_unused]] auto fence_wait_result = m_device.waitForFences(m_images_in_flight[info.image_index], true, UINT64_MAX);
    }

    m_images_in_flight[info.image_index] = m_in_flight_fences[info.current_frame];
    m_device.resetFences(m_in_flight_fences[info.current_frame]);

    // Record command buffer
    auto& cmd = m_command_buffers[info.image_index];
    auto _ = cmd.reset();
    auto begin_info = vk::CommandBufferBeginInfo();
    auto _ = cmd.begin(begin_info);

    // Acquire buffer ownership if needed
    if (info.needs_ownership_acquire) {
        // Queue family ownership transfer barrier
        auto barrier = vk::BufferMemoryBarrier()
            .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
            .setDstAccessMask(vk::AccessFlagBits::eVertexAttributeRead)
            .setSrcQueueFamilyIndex(info.compute_queue_family)
            .setDstQueueFamilyIndex(info.graphics_queue_family)
            .setBuffer(info.particle_buffer)
            .setOffset(0)
            .setSize(VK_WHOLE_SIZE);

        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eVertexInput,
            {},
            nullptr,
            barrier,
            nullptr
        );
    }

    // Begin render pass
    auto render_pass_begin = vk::RenderPassBeginInfo()
        .setRenderPass(info.render_pass)
        .setFramebuffer(info.framebuffer)
        .setRenderArea(vk::Rect2D({0, 0}, info.extent))
        .setClearValues(info.clear_values);

    cmd.beginRenderPass(render_pass_begin, vk::SubpassContents::eInline);

    // Render spheres
    render(cmd, info.particle_buffer, info.particle_count, info.camera);

    // Render ImGui if provided
    if (info.imgui_draw_data) {
        ImGui_ImplVulkan_RenderDrawData(
            static_cast<ImDrawData*>(info.imgui_draw_data),
            static_cast<VkCommandBuffer>(cmd)
        );
    }

    cmd.endRenderPass();
    auto _ = cmd.end();

    // Submit
    vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    auto submit_info = vk::SubmitInfo()
        .setWaitSemaphores(info.image_available_semaphore)
        .setWaitDstStageMask(wait_stage)
        .setCommandBuffers(cmd)
        .setSignalSemaphores(m_render_finished_semaphores[info.image_index]);

    auto _ = graphics_queue.submit(submit_info, m_in_flight_fences[info.current_frame]);

    return m_render_finished_semaphores[info.image_index];
}

void SphereRenderer::handle_swapchain_recreation(uint32_t new_image_count) {
    // Resize command buffers and per-image semaphores
    if (m_command_buffers.size() != new_image_count) {
        // Free old command buffers
        if (!m_command_buffers.empty()) {
            m_device.freeCommandBuffers(m_graphics_command_pool, m_command_buffers);
        }

        // Allocate new command buffers
        auto alloc_info = vk::CommandBufferAllocateInfo()
            .setCommandPool(m_graphics_command_pool)
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(new_image_count);
		auto command_buffer_res = m_device.allocateCommandBuffers(alloc_info);
        m_command_buffers = command_buffer_res.value;
    }

    // Resize per-image semaphores
    if (m_render_finished_semaphores.size() != new_image_count) {
        for (auto sem : m_render_finished_semaphores) {
            m_device.destroySemaphore(sem);
        }
        m_render_finished_semaphores.clear();

        for (uint32_t i = 0; i < new_image_count; i++) {
        	auto semaphore_res = m_device.createSemaphore({});
        	// Should probably check this but I dont want to change the function signature right now
            m_render_finished_semaphores.push_back(std::move(semaphore_res.value));
        }
    }

    m_images_in_flight.resize(new_image_count, nullptr);
}

} // namespace ifs
