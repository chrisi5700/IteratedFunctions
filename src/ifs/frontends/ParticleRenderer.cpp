#include <ifs/frontends/ParticleRenderer.hpp>
#include <ifs/Logger.hpp>
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <format>

namespace ifs {

// View parameter structure matching shader layout
struct alignas(16) ViewShaderParams {
    glm::mat4 view_projection;
    glm::vec2 screen_size;
    float point_size;
    float padding;
};

ParticleRenderer::ParticleRenderer(
    const VulkanContext& context,
    vk::Device device,
    vk::RenderPass render_pass,
    const vk::Extent2D& initial_extent
)
    : m_context(&context)
    , m_device(device)
    , m_render_pass(render_pass)
    , m_extent(initial_extent)
    , m_vertex_shader(nullptr)
    , m_fragment_shader(nullptr)
    , m_descriptor_layout(nullptr)
    , m_pipeline_layout(nullptr)
    , m_graphics_pipeline(nullptr)
    , m_descriptor_pool(nullptr)
    , m_descriptor_set(nullptr)
    , m_view_buffer(nullptr)
    , m_view_memory(nullptr)
    , m_point_size(2.0f)
    , m_graphics_command_pool(nullptr)
    , m_graphics_queue(nullptr)
{}

std::expected<std::unique_ptr<ParticleRenderer>, std::string> ParticleRenderer::create(
    const VulkanContext& context,
    vk::Device device,
    vk::RenderPass render_pass,
    const vk::Extent2D& initial_extent
) {
    auto frontend = std::unique_ptr<ParticleRenderer>(
        new ParticleRenderer(context, device, render_pass, initial_extent)
    );

    if (auto result = frontend->initialize(); !result) {
        return std::unexpected(result.error());
    }

    Logger::instance().info("Created ParticleRenderer frontend");
    return frontend;
}

ParticleRenderer::~ParticleRenderer() {
    cleanup();
}

ParticleRenderer::ParticleRenderer(ParticleRenderer&& other) noexcept
    : m_context(other.m_context)
    , m_device(other.m_device)
    , m_render_pass(other.m_render_pass)
    , m_extent(other.m_extent)
    , m_vertex_shader(std::move(other.m_vertex_shader))
    , m_fragment_shader(std::move(other.m_fragment_shader))
    , m_descriptor_layout(other.m_descriptor_layout)
    , m_pipeline_layout(other.m_pipeline_layout)
    , m_graphics_pipeline(other.m_graphics_pipeline)
    , m_descriptor_pool(other.m_descriptor_pool)
    , m_descriptor_set(other.m_descriptor_set)
    , m_view_buffer(other.m_view_buffer)
    , m_view_memory(other.m_view_memory)
    , m_point_size(other.m_point_size)
    , m_graphics_command_pool(other.m_graphics_command_pool)
    , m_command_buffers(std::move(other.m_command_buffers))
    , m_render_finished_semaphores(std::move(other.m_render_finished_semaphores))
    , m_in_flight_fences(std::move(other.m_in_flight_fences))
    , m_images_in_flight(std::move(other.m_images_in_flight))
    , m_graphics_queue(other.m_graphics_queue)
{
    other.m_descriptor_layout = nullptr;
    other.m_pipeline_layout = nullptr;
    other.m_graphics_pipeline = nullptr;
    other.m_descriptor_pool = nullptr;
    other.m_descriptor_set = nullptr;
    other.m_view_buffer = nullptr;
    other.m_view_memory = nullptr;
    other.m_graphics_command_pool = nullptr;
    other.m_graphics_queue = nullptr;
}

ParticleRenderer& ParticleRenderer::operator=(ParticleRenderer&& other) noexcept {
    if (this != &other) {
        cleanup();

        m_context = other.m_context;
        m_device = other.m_device;
        m_render_pass = other.m_render_pass;
        m_extent = other.m_extent;
        m_vertex_shader = std::move(other.m_vertex_shader);
        m_fragment_shader = std::move(other.m_fragment_shader);
        m_descriptor_layout = other.m_descriptor_layout;
        m_pipeline_layout = other.m_pipeline_layout;
        m_graphics_pipeline = other.m_graphics_pipeline;
        m_descriptor_pool = other.m_descriptor_pool;
        m_descriptor_set = other.m_descriptor_set;
        m_view_buffer = other.m_view_buffer;
        m_view_memory = other.m_view_memory;
        m_point_size = other.m_point_size;
        m_graphics_command_pool = other.m_graphics_command_pool;
        m_command_buffers = std::move(other.m_command_buffers);
        m_render_finished_semaphores = std::move(other.m_render_finished_semaphores);
        m_in_flight_fences = std::move(other.m_in_flight_fences);
        m_images_in_flight = std::move(other.m_images_in_flight);
        m_graphics_queue = other.m_graphics_queue;

        other.m_descriptor_layout = nullptr;
        other.m_pipeline_layout = nullptr;
        other.m_graphics_pipeline = nullptr;
        other.m_descriptor_pool = nullptr;
        other.m_descriptor_set = nullptr;
        other.m_view_buffer = nullptr;
        other.m_view_memory = nullptr;
        other.m_graphics_command_pool = nullptr;
        other.m_graphics_queue = nullptr;
    }
    return *this;
}

std::expected<void, std::string> ParticleRenderer::initialize() {
    // Load vertex shader
    auto vert_result = Shader::create_shader(
        m_device,
        "ifs_modular/frontends/particle/particle.vert.slang",
        "main"
    );
    if (!vert_result) {
        return std::unexpected(std::format("Failed to load vertex shader: {}", vert_result.error()));
    }
    m_vertex_shader = std::make_unique<Shader>(std::move(*vert_result));

    // Load fragment shader
    auto frag_result = Shader::create_shader(
        m_device,
        "ifs_modular/frontends/particle/particle.frag.slang",
        "main"
    );
    if (!frag_result) {
        return std::unexpected(std::format("Failed to load fragment shader: {}", frag_result.error()));
    }
    m_fragment_shader = std::make_unique<Shader>(std::move(*frag_result));

    // Create descriptor layout
    if (auto result = create_descriptor_layout(); !result) {
        return std::unexpected(result.error());
    }

    // Create pipeline
    if (auto result = create_pipeline(); !result) {
        return std::unexpected(result.error());
    }

    // Create view parameter buffer
    auto buffer_info = vk::BufferCreateInfo()
        .setSize(sizeof(ViewShaderParams))
        .setUsage(vk::BufferUsageFlagBits::eUniformBuffer)
        .setSharingMode(vk::SharingMode::eExclusive);

	auto buffer_res = m_device.createBuffer(buffer_info);
	if (buffer_res.result != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to create view buffer: {}", to_string(buffer_res.result)));
	}
	m_view_buffer = buffer_res.value;

    // Allocate host-visible memory
    auto mem_reqs = m_device.getBufferMemoryRequirements(m_view_buffer);
    auto mem_props = m_context->physical_device().getMemoryProperties();

    uint32_t memory_type = UINT32_MAX;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((mem_reqs.memoryTypeBits & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags &
             (vk::MemoryPropertyFlagBits::eHostVisible |
              vk::MemoryPropertyFlagBits::eHostCoherent))) {
            memory_type = i;
            break;
        }
    }

    if (memory_type == UINT32_MAX) {
        m_device.destroyBuffer(m_view_buffer);
        m_view_buffer = nullptr;
        return std::unexpected("Failed to find suitable memory type for view buffer");
    }

    auto alloc_info = vk::MemoryAllocateInfo()
        .setAllocationSize(mem_reqs.size)
        .setMemoryTypeIndex(memory_type);

	auto alloc_res = m_device.allocateMemory(alloc_info);
	if (alloc_res.result != vk::Result::eSuccess)
	{
		m_device.destroyBuffer(m_view_buffer);
		m_view_buffer = nullptr;
		return std::unexpected(std::format("Failed to allocate view memory: {}", to_string(alloc_res.result)));
	}
	m_view_memory = alloc_res.value;
	auto bind_res = m_device.bindBufferMemory(m_view_buffer, m_view_memory, 0);
	if (bind_res != vk::Result::eSuccess)
	{
		m_device.destroyBuffer(m_view_buffer);
		m_view_buffer = nullptr;
		return std::unexpected(std::format("Failed to bind view memory: {}", to_string(bind_res)));
	}

    // Create descriptor pool
    std::vector<vk::DescriptorPoolSize> pool_sizes = {
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1),
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1)
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
    auto alloc_info_desc = vk::DescriptorSetAllocateInfo()
        .setDescriptorPool(m_descriptor_pool)
        .setSetLayouts(m_descriptor_layout);

	auto descriptor_set_res = m_device.allocateDescriptorSets(alloc_info_desc);
	if (descriptor_set_res.result != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to allocate descriptor set: {}", to_string(descriptor_set_res.result)));
	}
	m_descriptor_set = descriptor_set_res.value[0];

    // Update descriptor set with view buffer (particle buffer will be updated in render())
    auto view_buffer_info = vk::DescriptorBufferInfo()
        .setBuffer(m_view_buffer)
        .setOffset(0)
        .setRange(sizeof(ViewShaderParams));

    auto write = vk::WriteDescriptorSet()
        .setDstSet(m_descriptor_set)
        .setDstBinding(1)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eUniformBuffer)
        .setDescriptorCount(1)
        .setBufferInfo(view_buffer_info);

    m_device.updateDescriptorSets(write, {});

    // Phase 3: Create graphics command infrastructure
    m_graphics_queue = m_context->graphics_queue();

    // Create command pool for graphics queue
    auto cmd_pool_info = vk::CommandPoolCreateInfo()
        .setQueueFamilyIndex(m_context->queue_indices().graphics)
        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

	auto cmd_pool_res = m_device.createCommandPool(cmd_pool_info);
	if (cmd_pool_res.result != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to create graphics command pool: {}", to_string(cmd_pool_res.result)));
	}
	m_graphics_command_pool = cmd_pool_res.value;

    // Create fences for frames-in-flight
    m_in_flight_fences.reserve(MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        auto fence_info = vk::FenceCreateInfo()
            .setFlags(vk::FenceCreateFlagBits::eSignaled);  // Start signaled

		auto fence_res = m_device.createFence(fence_info);
		if (fence_res.result != vk::Result::eSuccess)
		{
			return std::unexpected(std::format("Failed to create in-flight fence: {}", to_string(fence_res.result)));
		}
		m_in_flight_fences.push_back(fence_res.value);
    }

    // Note: Command buffers and semaphores will be created when swapchain is known
    // They are NOT created here because we need to know the swapchain image count first

    return {};
}

std::expected<void, std::string> ParticleRenderer::create_descriptor_layout() {
    if (!m_vertex_shader) {
        return std::unexpected("Vertex shader not loaded");
    }

    // Get descriptor info from shader reflection
    auto& descriptors = m_vertex_shader->get_descriptor_infos();

    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    for (const auto& desc : descriptors) {
        bindings.push_back(vk::DescriptorSetLayoutBinding()
            .setBinding(static_cast<uint32_t>(desc.binding))
            .setDescriptorType(desc.type)
            .setDescriptorCount(static_cast<uint32_t>(desc.descriptor_count))
            .setStageFlags(vk::ShaderStageFlagBits::eVertex)
        );
    }

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

std::expected<void, std::string> ParticleRenderer::create_pipeline() {
    if (!m_descriptor_layout) {
        return std::unexpected("Descriptor layout not created");
    }

    // Create pipeline layout
    auto pipeline_layout_info = vk::PipelineLayoutCreateInfo()
        .setSetLayouts(m_descriptor_layout);

	auto pipeline_layout_res = m_device.createPipelineLayout(pipeline_layout_info);
	if (pipeline_layout_res.result != vk::Result::eSuccess)
	{
		return std::unexpected(std::format("Failed to create pipeline layout: {}", to_string(pipeline_layout_res.result)));
	}
	m_pipeline_layout = pipeline_layout_res.value;

    // Shader stages
    std::vector<vk::PipelineShaderStageCreateInfo> shader_stages = {
        vk::PipelineShaderStageCreateInfo()
            .setStage(vk::ShaderStageFlagBits::eVertex)
            .setModule(m_vertex_shader->get_shader_module())
            .setPName("main"),
        vk::PipelineShaderStageCreateInfo()
            .setStage(vk::ShaderStageFlagBits::eFragment)
            .setModule(m_fragment_shader->get_shader_module())
            .setPName("main")
    };

    // Vertex input (empty - using SSBO in shader)
    auto vertex_input_info = vk::PipelineVertexInputStateCreateInfo();

    // Input assembly
    auto input_assembly = vk::PipelineInputAssemblyStateCreateInfo()
        .setTopology(vk::PrimitiveTopology::ePointList)
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
        .setFrontFace(vk::FrontFace::eCounterClockwise)
        .setDepthBiasEnable(false);

    // Multisampling
    auto multisampling = vk::PipelineMultisampleStateCreateInfo()
        .setSampleShadingEnable(false)
        .setRasterizationSamples(vk::SampleCountFlagBits::e1);

    // Color blending
    auto color_blend_attachment = vk::PipelineColorBlendAttachmentState()
        .setColorWriteMask(vk::ColorComponentFlagBits::eR |
                          vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB |
                          vk::ColorComponentFlagBits::eA)
        .setBlendEnable(true)
        .setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
        .setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
        .setColorBlendOp(vk::BlendOp::eAdd)
        .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
        .setDstAlphaBlendFactor(vk::BlendFactor::eZero)
        .setAlphaBlendOp(vk::BlendOp::eAdd);

    auto color_blending = vk::PipelineColorBlendStateCreateInfo()
        .setLogicOpEnable(false)
        .setAttachments(color_blend_attachment);

    // Depth/stencil state
    auto depth_stencil = vk::PipelineDepthStencilStateCreateInfo()
        .setDepthTestEnable(true)
        .setDepthWriteEnable(true)
        .setDepthCompareOp(vk::CompareOp::eLess)
        .setDepthBoundsTestEnable(false)
        .setStencilTestEnable(false);

    // Dynamic state
    std::vector<vk::DynamicState> dynamic_states = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };

    auto dynamic_state = vk::PipelineDynamicStateCreateInfo()
        .setDynamicStates(dynamic_states);

    // Create pipeline
    auto pipeline_info = vk::GraphicsPipelineCreateInfo()
        .setStages(shader_stages)
        .setPVertexInputState(&vertex_input_info)
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

void ParticleRenderer::cleanup() {
    // Phase 3: Cleanup graphics infrastructure
    // Wait for any pending rendering operations
    if (!m_in_flight_fences.empty()) {
        [[maybe_unused]] auto result = m_device.waitForFences(m_in_flight_fences, true, UINT64_MAX);
    }

    for (auto& fence : m_in_flight_fences) {
        m_device.destroyFence(fence);
    }
    m_in_flight_fences.clear();

    for (auto& sem : m_render_finished_semaphores) {
        m_device.destroySemaphore(sem);
    }
    m_render_finished_semaphores.clear();

    if (m_graphics_command_pool) {
        // Command buffers are freed when pool is destroyed
        m_device.destroyCommandPool(m_graphics_command_pool);
        m_graphics_command_pool = nullptr;
        m_command_buffers.clear();
    }

    m_images_in_flight.clear();

    if (m_descriptor_pool) {
        m_device.destroyDescriptorPool(m_descriptor_pool);
        m_descriptor_pool = nullptr;
    }
    if (m_graphics_pipeline) {
        m_device.destroyPipeline(m_graphics_pipeline);
        m_graphics_pipeline = nullptr;
    }
    if (m_pipeline_layout) {
        m_device.destroyPipelineLayout(m_pipeline_layout);
        m_pipeline_layout = nullptr;
    }
    if (m_descriptor_layout) {
        m_device.destroyDescriptorSetLayout(m_descriptor_layout);
        m_descriptor_layout = nullptr;
    }
    if (m_view_buffer) {
        m_device.destroyBuffer(m_view_buffer);
        m_view_buffer = nullptr;
    }
    if (m_view_memory) {
        m_device.freeMemory(m_view_memory);
        m_view_memory = nullptr;
    }
}

void ParticleRenderer::update_particle_buffer(vk::Buffer particle_buffer) {
    // Update descriptor set with particle buffer
    auto particle_buffer_info = vk::DescriptorBufferInfo()
        .setBuffer(particle_buffer)
        .setOffset(0)
        .setRange(VK_WHOLE_SIZE);

    auto write = vk::WriteDescriptorSet()
        .setDstSet(m_descriptor_set)
        .setDstBinding(0)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setDescriptorCount(1)
        .setBufferInfo(particle_buffer_info);

    m_device.updateDescriptorSets(write, {});
}

void ParticleRenderer::render(
    vk::CommandBuffer cmd,
    [[maybe_unused]] vk::Buffer particle_buffer,
    uint32_t particle_count,
    Camera& camera,
    const vk::Extent2D* extent
) {
    // Use provided extent or fall back to stored extent
    const auto& render_extent = extent ? *extent : m_extent;
    // Update view parameters
    ViewShaderParams view_params{
        .view_projection = camera.view_projection_matrix(),
        .screen_size = glm::vec2(m_extent.width, m_extent.height),
        .point_size = m_point_size,
        .padding = 0.0f
    };

    auto data_res = m_device.mapMemory(m_view_memory, 0, sizeof(ViewShaderParams));
	auto data = data_res.value;
    std::memcpy(data, &view_params, sizeof(ViewShaderParams));
    m_device.unmapMemory(m_view_memory);

    // Set dynamic viewport and scissor
    auto viewport = vk::Viewport()
        .setX(0.0f)
        .setY(0.0f)
        .setWidth(static_cast<float>(render_extent.width))
        .setHeight(static_cast<float>(render_extent.height))
        .setMinDepth(0.0f)
        .setMaxDepth(1.0f);

    auto scissor = vk::Rect2D()
        .setOffset({0, 0})
        .setExtent(render_extent);

    cmd.setViewport(0, viewport);
    cmd.setScissor(0, scissor);

    // Bind pipeline and draw
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_graphics_pipeline);
    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        m_pipeline_layout,
        0,
        m_descriptor_set,
        {}
    );

    cmd.draw(particle_count, 1, 0, 0);
}

void ParticleRenderer::resize(const vk::Extent2D& new_extent) {
    m_extent = new_extent;
    // Viewport and scissor are dynamic, so no pipeline recreation needed
}

void ParticleRenderer::handle_swapchain_recreation(uint32_t new_image_count) {
    // Wait for device to finish before recreating resources
    auto _ = m_device.waitIdle();

    // Destroy old semaphores and command buffers
    for (auto& sem : m_render_finished_semaphores) {
        m_device.destroySemaphore(sem);
    }
    m_render_finished_semaphores.clear();

    // Free command buffers (will be reallocated below)
    if (!m_command_buffers.empty() && m_graphics_command_pool) {
        m_device.freeCommandBuffers(m_graphics_command_pool, m_command_buffers);
        m_command_buffers.clear();
    }

    // Create new semaphores for each swapchain image
    m_render_finished_semaphores.reserve(new_image_count);
    for (uint32_t i = 0; i < new_image_count; i++) {
		auto semaphore_res = m_device.createSemaphore({});
		if (semaphore_res.result != vk::Result::eSuccess)
		{
			Logger::instance().error("Failed to create render finished semaphore: {}", to_string(semaphore_res.result));
			return;
		}
		m_render_finished_semaphores.push_back(semaphore_res.value);
    }

    // Allocate command buffers (one per swapchain image)
    auto cmd_alloc_info = vk::CommandBufferAllocateInfo()
        .setCommandPool(m_graphics_command_pool)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(new_image_count);

	auto cmd_buffers_res = m_device.allocateCommandBuffers(cmd_alloc_info);
	if (cmd_buffers_res.result != vk::Result::eSuccess)
	{
		Logger::instance().error("Failed to allocate command buffers: {}", to_string(cmd_buffers_res.result));
		return;
	}
	m_command_buffers = cmd_buffers_res.value;

    // Reset image-in-flight tracking
    m_images_in_flight.clear();
    m_images_in_flight.resize(new_image_count, nullptr);

    Logger::instance().info("Frontend swapchain resources recreated for {} images", new_image_count);
}

vk::Semaphore ParticleRenderer::render_frame(
    const FrameRenderInfo& info,
    vk::Queue graphics_queue
) {
    // Wait for the current frame's fence
    [[maybe_unused]] auto wait_result = m_device.waitForFences(
        m_in_flight_fences[info.current_frame], true, UINT64_MAX);

    // If this image is still being used by a previous frame, wait for it
    if (m_images_in_flight[info.image_index]) {
        [[maybe_unused]] auto fence_wait_result = m_device.waitForFences(
            m_images_in_flight[info.image_index], true, UINT64_MAX);
    }

    // Mark this image as now being used by this frame
    m_images_in_flight[info.image_index] = m_in_flight_fences[info.current_frame];

    // Now we can reset the fence for the current frame
    m_device.resetFences(m_in_flight_fences[info.current_frame]);

    // Record command buffer
    auto& cmd = m_command_buffers[info.image_index];
    auto _ = cmd.reset();
    auto begin_info = vk::CommandBufferBeginInfo();
    auto _ = cmd.begin(begin_info);

    // Acquire buffer ownership if needed
    if (info.needs_ownership_acquire) {
        acquire_buffer_ownership(cmd, info.particle_buffer,
            info.compute_queue_family, info.graphics_queue_family);
    }

    // Begin render pass
    auto render_pass_begin = vk::RenderPassBeginInfo()
        .setRenderPass(info.render_pass)
        .setFramebuffer(info.framebuffer)
        .setRenderArea(vk::Rect2D({0, 0}, info.extent))
        .setClearValues(info.clear_values);

    cmd.beginRenderPass(render_pass_begin, vk::SubpassContents::eInline);

    // Render particles (pass the extent to ensure correct viewport/scissor)
    render(cmd, info.particle_buffer, info.particle_count, info.camera, &info.extent);

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

    // Return the semaphore to wait on for presentation
    return m_render_finished_semaphores[info.image_index];
}

} // namespace ifs
