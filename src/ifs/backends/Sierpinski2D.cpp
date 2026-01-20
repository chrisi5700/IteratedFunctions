#include <ifs/backends/Sierpinski2D.hpp>
#include <ifs/Logger.hpp>
#include <format>
#include <random>

namespace ifs {

// Parameter structure matching shader layout
struct alignas(16) IFSShaderParams {
    uint32_t iteration_count;
    uint32_t particle_count;
    float scale;
    uint32_t random_seed;
};

Sierpinski2D::Sierpinski2D(
    const VulkanContext& context,
    vk::Device device
)
    : m_context(&context)
    , m_device(device)
    , m_particle_buffer(nullptr)
    , m_particle_count(100000)  // Default particle count
    , m_compute_shader(nullptr)
    , m_descriptor_layout(nullptr)
    , m_pipeline_layout(nullptr)
    , m_compute_pipeline(nullptr)
    , m_descriptor_pool(nullptr)
    , m_descriptor_set(nullptr)
    , m_param_buffer(nullptr)
    , m_param_memory(nullptr)
    , m_compute_command_pool(nullptr)
    , m_compute_command_buffer(nullptr)
    , m_compute_fence(nullptr)
    , m_compute_queue(nullptr)
{}

std::expected<std::unique_ptr<Sierpinski2D>, std::string> Sierpinski2D::create(
    const VulkanContext& context,
    vk::Device device
) {
    auto backend = std::unique_ptr<Sierpinski2D>(new Sierpinski2D(context, device));

    if (auto result = backend->initialize(); !result) {
        return std::unexpected(result.error());
    }

    Logger::instance().info("Created Sierpinski2D backend");
    return backend;
}

Sierpinski2D::~Sierpinski2D() {
    cleanup();
}

Sierpinski2D::Sierpinski2D(Sierpinski2D&& other) noexcept
    : m_context(other.m_context)
    , m_device(other.m_device)
    , m_particle_buffer(std::move(other.m_particle_buffer))
    , m_particle_count(other.m_particle_count)
    , m_compute_shader(std::move(other.m_compute_shader))
    , m_descriptor_layout(other.m_descriptor_layout)
    , m_pipeline_layout(other.m_pipeline_layout)
    , m_compute_pipeline(other.m_compute_pipeline)
    , m_descriptor_pool(other.m_descriptor_pool)
    , m_descriptor_set(other.m_descriptor_set)
    , m_param_buffer(other.m_param_buffer)
    , m_param_memory(other.m_param_memory)
    , m_compute_command_pool(other.m_compute_command_pool)
    , m_compute_command_buffer(other.m_compute_command_buffer)
    , m_compute_fence(other.m_compute_fence)
    , m_compute_queue(other.m_compute_queue)
{
    other.m_particle_count = 0;
    other.m_descriptor_layout = nullptr;
    other.m_pipeline_layout = nullptr;
    other.m_compute_pipeline = nullptr;
    other.m_descriptor_pool = nullptr;
    other.m_descriptor_set = nullptr;
    other.m_param_buffer = nullptr;
    other.m_param_memory = nullptr;
    other.m_compute_command_pool = nullptr;
    other.m_compute_command_buffer = nullptr;
    other.m_compute_fence = nullptr;
    other.m_compute_queue = nullptr;
}

Sierpinski2D& Sierpinski2D::operator=(Sierpinski2D&& other) noexcept {
    if (this != &other) {
        cleanup();

        m_context = other.m_context;
        m_device = other.m_device;
        m_particle_buffer = std::move(other.m_particle_buffer);
        m_particle_count = other.m_particle_count;
        m_compute_shader = std::move(other.m_compute_shader);
        m_descriptor_layout = other.m_descriptor_layout;
        m_pipeline_layout = other.m_pipeline_layout;
        m_compute_pipeline = other.m_compute_pipeline;
        m_descriptor_pool = other.m_descriptor_pool;
        m_descriptor_set = other.m_descriptor_set;
        m_param_buffer = other.m_param_buffer;
        m_param_memory = other.m_param_memory;
        m_compute_command_pool = other.m_compute_command_pool;
        m_compute_command_buffer = other.m_compute_command_buffer;
        m_compute_fence = other.m_compute_fence;
        m_compute_queue = other.m_compute_queue;

        other.m_descriptor_layout = nullptr;
        other.m_pipeline_layout = nullptr;
        other.m_compute_pipeline = nullptr;
        other.m_descriptor_pool = nullptr;
        other.m_descriptor_set = nullptr;
        other.m_param_buffer = nullptr;
        other.m_param_memory = nullptr;
        other.m_compute_command_pool = nullptr;
        other.m_compute_command_buffer = nullptr;
        other.m_compute_fence = nullptr;
        other.m_compute_queue = nullptr;
    }
    return *this;
}

std::expected<void, std::string> Sierpinski2D::initialize() {
    // Load compute shader
    auto shader_result = Shader::create_shader(
        m_device,
        "ifs_modular/backends/sierpinski_2d",
        "main"
    );
    if (!shader_result) {
        return std::unexpected(std::format("Failed to load shader: {}", shader_result.error()));
    }
    m_compute_shader = std::make_unique<Shader>(std::move(*shader_result));

    // Create descriptor layout
    if (auto result = create_descriptor_layout(); !result) {
        return std::unexpected(result.error());
    }

    // Create pipeline
    if (auto result = create_pipeline(); !result) {
        return std::unexpected(result.error());
    }

    // Create parameter buffer
    auto buffer_info = vk::BufferCreateInfo()
        .setSize(sizeof(IFSShaderParams))
        .setUsage(vk::BufferUsageFlagBits::eUniformBuffer)
        .setSharingMode(vk::SharingMode::eExclusive);

    try {
        m_param_buffer = m_device.createBuffer(buffer_info);
    } catch (const vk::SystemError& e) {
        return std::unexpected(std::format("Failed to create parameter buffer: {}", e.what()));
    }

    // Allocate host-visible memory
    auto mem_reqs = m_device.getBufferMemoryRequirements(m_param_buffer);
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
        m_device.destroyBuffer(m_param_buffer);
        m_param_buffer = nullptr;
        return std::unexpected("Failed to find suitable memory type for parameter buffer");
    }

    auto alloc_info = vk::MemoryAllocateInfo()
        .setAllocationSize(mem_reqs.size)
        .setMemoryTypeIndex(memory_type);

    try {
        m_param_memory = m_device.allocateMemory(alloc_info);
        m_device.bindBufferMemory(m_param_buffer, m_param_memory, 0);
    } catch (const vk::SystemError& e) {
        m_device.destroyBuffer(m_param_buffer);
        m_param_buffer = nullptr;
        return std::unexpected(std::format("Failed to allocate parameter memory: {}", e.what()));
    }

    // Create descriptor pool
    std::vector<vk::DescriptorPoolSize> pool_sizes = {
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1),
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1)
    };

    auto pool_info = vk::DescriptorPoolCreateInfo()
        .setMaxSets(1)
        .setPoolSizes(pool_sizes);

    try {
        m_descriptor_pool = m_device.createDescriptorPool(pool_info);
    } catch (const vk::SystemError& e) {
        return std::unexpected(std::format("Failed to create descriptor pool: {}", e.what()));
    }

    // Allocate descriptor set
    auto alloc_info_desc = vk::DescriptorSetAllocateInfo()
        .setDescriptorPool(m_descriptor_pool)
        .setSetLayouts(m_descriptor_layout);

    try {
        m_descriptor_set = m_device.allocateDescriptorSets(alloc_info_desc)[0];
    } catch (const vk::SystemError& e) {
        return std::unexpected(std::format("Failed to allocate descriptor set: {}", e.what()));
    }

    // Update descriptor set with parameter buffer (particle buffer will be updated in dispatch())
    auto param_buffer_info = vk::DescriptorBufferInfo()
        .setBuffer(m_param_buffer)
        .setOffset(0)
        .setRange(sizeof(IFSShaderParams));

    auto write = vk::WriteDescriptorSet()
        .setDstSet(m_descriptor_set)
        .setDstBinding(1)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eUniformBuffer)
        .setDescriptorCount(1)
        .setBufferInfo(param_buffer_info);

    m_device.updateDescriptorSets(write, {});

    // Phase 2: Create compute command infrastructure
    m_compute_queue = m_context->compute_queue();

    // Create command pool for compute queue
    auto cmd_pool_info = vk::CommandPoolCreateInfo()
        .setQueueFamilyIndex(m_context->queue_indices().compute)
        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

    try {
        m_compute_command_pool = m_device.createCommandPool(cmd_pool_info);
    } catch (const vk::SystemError& e) {
        return std::unexpected(std::format("Failed to create compute command pool: {}", e.what()));
    }

    // Allocate command buffer
    auto cmd_alloc_info = vk::CommandBufferAllocateInfo()
        .setCommandPool(m_compute_command_pool)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(1);

    try {
        m_compute_command_buffer = m_device.allocateCommandBuffers(cmd_alloc_info)[0];
    } catch (const vk::SystemError& e) {
        return std::unexpected(std::format("Failed to allocate compute command buffer: {}", e.what()));
    }

    // Create fence for synchronization
    auto fence_info = vk::FenceCreateInfo()
        .setFlags(vk::FenceCreateFlagBits::eSignaled);  // Start signaled

    try {
        m_compute_fence = m_device.createFence(fence_info);
    } catch (const vk::SystemError& e) {
        return std::unexpected(std::format("Failed to create compute fence: {}", e.what()));
    }

    // Create particle buffer with initial particle count
    ParticleBufferConfig buffer_config{
        .particle_count = m_particle_count,
        .support_dynamic_resize = false
    };

    auto particle_buffer_result = ParticleBuffer::create(*m_context, m_device, buffer_config);
    if (!particle_buffer_result) {
        return std::unexpected(std::format("Failed to create particle buffer: {}", particle_buffer_result.error()));
    }
    m_particle_buffer = std::make_unique<ParticleBuffer>(std::move(particle_buffer_result.value()));

    // Initialize with random data
    std::random_device rd;
    if (auto result = m_particle_buffer->initialize_random(m_compute_command_pool, m_compute_queue, rd()); !result) {
        return std::unexpected(std::format("Failed to initialize particle buffer: {}", result.error()));
    }

    // Update descriptor set with particle buffer
    auto particle_buffer_info = vk::DescriptorBufferInfo()
        .setBuffer(m_particle_buffer->buffer())
        .setOffset(0)
        .setRange(VK_WHOLE_SIZE);

    auto particle_write = vk::WriteDescriptorSet()
        .setDstSet(m_descriptor_set)
        .setDstBinding(0)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setDescriptorCount(1)
        .setBufferInfo(particle_buffer_info);

    m_device.updateDescriptorSets(particle_write, {});

    return {};
}

std::expected<void, std::string> Sierpinski2D::create_descriptor_layout() {
    if (!m_compute_shader) {
        return std::unexpected("Compute shader not loaded");
    }

    // Get descriptor info from shader reflection
    auto& descriptors = m_compute_shader->get_descriptor_infos();

    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    for (const auto& desc : descriptors) {
        bindings.push_back(vk::DescriptorSetLayoutBinding()
            .setBinding(static_cast<uint32_t>(desc.binding))
            .setDescriptorType(desc.type)
            .setDescriptorCount(static_cast<uint32_t>(desc.descriptor_count))
            .setStageFlags(vk::ShaderStageFlagBits::eCompute)
        );
    }

    auto layout_info = vk::DescriptorSetLayoutCreateInfo()
        .setBindings(bindings);

    try {
        m_descriptor_layout = m_device.createDescriptorSetLayout(layout_info);
    } catch (const vk::SystemError& e) {
        return std::unexpected(std::format("Failed to create descriptor layout: {}", e.what()));
    }

    return {};
}

std::expected<void, std::string> Sierpinski2D::create_pipeline() {
    if (!m_descriptor_layout) {
        return std::unexpected("Descriptor layout not created");
    }

    // Create pipeline layout
    auto pipeline_layout_info = vk::PipelineLayoutCreateInfo()
        .setSetLayouts(m_descriptor_layout);

    try {
        m_pipeline_layout = m_device.createPipelineLayout(pipeline_layout_info);
    } catch (const vk::SystemError& e) {
        return std::unexpected(std::format("Failed to create pipeline layout: {}", e.what()));
    }

    // Get compute shader module and verify it's a compute shader
    auto& shader_details = m_compute_shader->get_details();
    if (!std::holds_alternative<ComputeDetails>(shader_details)) {
        return std::unexpected("Shader is not a compute shader");
    }

    auto stage_info = vk::PipelineShaderStageCreateInfo()
        .setStage(vk::ShaderStageFlagBits::eCompute)
        .setModule(m_compute_shader->get_shader_module())
        .setPName("main");

    auto pipeline_info = vk::ComputePipelineCreateInfo()
        .setStage(stage_info)
        .setLayout(m_pipeline_layout);

    try {
        auto result = m_device.createComputePipeline(nullptr, pipeline_info);
        if (result.result != vk::Result::eSuccess) {
            return std::unexpected(std::format("Failed to create compute pipeline: {}",
                vk::to_string(result.result)));
        }
        m_compute_pipeline = result.value;
    } catch (const vk::SystemError& e) {
        return std::unexpected(std::format("Failed to create compute pipeline: {}", e.what()));
    }

    return {};
}

void Sierpinski2D::cleanup() {
    // Phase 2: Cleanup compute infrastructure
    if (m_compute_fence) {
        // Wait for any pending compute operations before cleanup
        [[maybe_unused]] auto result = m_device.waitForFences(m_compute_fence, true, UINT64_MAX);
        m_device.destroyFence(m_compute_fence);
        m_compute_fence = nullptr;
    }
    if (m_compute_command_pool) {
        // Command buffer is freed when pool is destroyed
        m_device.destroyCommandPool(m_compute_command_pool);
        m_compute_command_pool = nullptr;
        m_compute_command_buffer = nullptr;
    }

    if (m_descriptor_pool) {
        m_device.destroyDescriptorPool(m_descriptor_pool);
        m_descriptor_pool = nullptr;
    }
    if (m_compute_pipeline) {
        m_device.destroyPipeline(m_compute_pipeline);
        m_compute_pipeline = nullptr;
    }
    if (m_pipeline_layout) {
        m_device.destroyPipelineLayout(m_pipeline_layout);
        m_pipeline_layout = nullptr;
    }
    if (m_descriptor_layout) {
        m_device.destroyDescriptorSetLayout(m_descriptor_layout);
        m_descriptor_layout = nullptr;
    }
    if (m_param_buffer) {
        m_device.destroyBuffer(m_param_buffer);
        m_param_buffer = nullptr;
    }
    if (m_param_memory) {
        m_device.freeMemory(m_param_memory);
        m_param_memory = nullptr;
    }
}

void Sierpinski2D::dispatch(
    vk::CommandBuffer cmd,
    vk::Buffer particle_buffer,
    uint32_t particle_count,
    const IFSParameters& params
) {
    // Update parameter buffer
    IFSShaderParams shader_params{
        .iteration_count = params.iteration_count,
        .particle_count = particle_count,
        .scale = params.scale,
        .random_seed = params.random_seed
    };

    void* data = m_device.mapMemory(m_param_memory, 0, sizeof(IFSShaderParams));
    std::memcpy(data, &shader_params, sizeof(IFSShaderParams));
    m_device.unmapMemory(m_param_memory);

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

    // Bind pipeline and dispatch
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_compute_pipeline);
    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        m_pipeline_layout,
        0,
        m_descriptor_set,
        {}
    );

    // Calculate dispatch size (256 threads per workgroup)
    uint32_t workgroup_count = (particle_count + 255) / 256;
    cmd.dispatch(workgroup_count, 1, 1);
}

void Sierpinski2D::compute(
    vk::Buffer particle_buffer,
    uint32_t particle_count,
    const IFSParameters& params
) {
    // Note: particle_buffer and particle_count parameters are ignored
    // Backend owns its own particle buffer now
    (void)particle_buffer;
    (void)particle_count;

    // Wait for previous compute to finish (if any)
    wait_compute_complete();

    // Reset fence before recording
    m_device.resetFences(m_compute_fence);

    // Reset and begin command buffer
    m_compute_command_buffer.reset();
    m_compute_command_buffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // Record dispatch using backend's own particle buffer
    dispatch(m_compute_command_buffer, m_particle_buffer->buffer(), m_particle_count, params);

    // Issue ownership release barrier if different queue families
    bool different_queue_families = m_context->queue_indices().has_dedicated_compute();
    if (different_queue_families) {
        release_buffer_ownership(
            m_compute_command_buffer,
            m_particle_buffer->buffer(),
            m_context->queue_indices().compute,
            m_context->queue_indices().graphics
        );
    }

    m_compute_command_buffer.end();

    // Submit to compute queue with fence
    auto submit_info = vk::SubmitInfo()
        .setCommandBuffers(m_compute_command_buffer);

    m_compute_queue.submit(submit_info, m_compute_fence);

    // Return immediately - asynchronous execution
}

void Sierpinski2D::wait_compute_complete() {
    if (m_compute_fence) {
        [[maybe_unused]] auto result = m_device.waitForFences(m_compute_fence, true, UINT64_MAX);
    }
}

std::vector<UICallback> Sierpinski2D::get_ui_callbacks() {
    std::vector<UICallback> callbacks;

    // Particle count slider (logarithmic, 10K to 100M in steps of 10K)
    callbacks.emplace_back("Particle Count", DiscreteCallback{
        .setter = [this](int v) {
            // Round to nearest multiple of 10,000
            uint32_t new_count = static_cast<uint32_t>(v);
            new_count = (new_count / 10000) * 10000;
            if (new_count < 10000) new_count = 10000;
            if (new_count > 100000000) new_count = 100000000;

            if (new_count != m_particle_count) {
                reallocate_particle_buffer(new_count);
            }
        },
        .getter = [this]() { return static_cast<int>(m_particle_count); },
        .min = 10000,
        .max = 100000000
    });

    return callbacks;
}

void Sierpinski2D::reallocate_particle_buffer(uint32_t new_count) {
    Logger::instance().info("Reallocating particle buffer: {} -> {} particles", m_particle_count, new_count);

    // Wait for any pending compute operations
    wait_compute_complete();
    m_device.waitIdle();

    // Update particle count
    m_particle_count = new_count;

    // Recreate particle buffer
    ParticleBufferConfig buffer_config{
        .particle_count = m_particle_count,
        .support_dynamic_resize = false
    };

    auto particle_buffer_result = ParticleBuffer::create(*m_context, m_device, buffer_config);
    if (!particle_buffer_result) {
        Logger::instance().error("Failed to reallocate particle buffer: {}", particle_buffer_result.error());
        return;
    }
    m_particle_buffer = std::make_unique<ParticleBuffer>(std::move(particle_buffer_result.value()));

    // Initialize with random data
    std::random_device rd;
    if (auto result = m_particle_buffer->initialize_random(m_compute_command_pool, m_compute_queue, rd()); !result) {
        Logger::instance().error("Failed to initialize particle buffer: {}", result.error());
        return;
    }

    // Update descriptor set with new particle buffer
    auto particle_buffer_info = vk::DescriptorBufferInfo()
        .setBuffer(m_particle_buffer->buffer())
        .setOffset(0)
        .setRange(VK_WHOLE_SIZE);

    auto particle_write = vk::WriteDescriptorSet()
        .setDstSet(m_descriptor_set)
        .setDstBinding(0)
        .setDstArrayElement(0)
        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
        .setDescriptorCount(1)
        .setBufferInfo(particle_buffer_info);

    m_device.updateDescriptorSets(particle_write, {});

    Logger::instance().info("Particle buffer reallocated successfully");
}

} // namespace ifs
