#include <ifs/ParticleBuffer.hpp>
#include <ifs/Logger.hpp>
#include <format>

namespace ifs {

ParticleBuffer::ParticleBuffer(
    const VulkanContext& context,
    vk::Device device,
    const ParticleBufferConfig& config
)
    : m_context(&context)
    , m_device(device)
    , m_config(config)
    , m_buffer(nullptr)
    , m_memory(nullptr)
    , m_particle_count(config.particle_count)
    , m_buffer_size(config.particle_count * sizeof(Particle))
{}

std::expected<ParticleBuffer, std::string> ParticleBuffer::create(
    const VulkanContext& context,
    vk::Device device,
    const ParticleBufferConfig& config
) {
    ParticleBuffer buffer(context, device, config);

    if (auto result = buffer.create_buffer(); !result) {
        return std::unexpected(result.error());
    }

    Logger::instance().info("Created particle buffer: {} particles ({} MB)",
        config.particle_count,
        buffer.m_buffer_size / (1024.0 * 1024.0));

    return buffer;
}

ParticleBuffer::~ParticleBuffer() {
    destroy_buffer();
}

ParticleBuffer::ParticleBuffer(ParticleBuffer&& other) noexcept
    : m_context(other.m_context)
    , m_device(other.m_device)
    , m_config(other.m_config)
    , m_buffer(other.m_buffer)
    , m_memory(other.m_memory)
    , m_particle_count(other.m_particle_count)
    , m_buffer_size(other.m_buffer_size)
{
    other.m_buffer = nullptr;
    other.m_memory = nullptr;
}

ParticleBuffer& ParticleBuffer::operator=(ParticleBuffer&& other) noexcept {
    if (this != &other) {
        destroy_buffer();

        m_context = other.m_context;
        m_device = other.m_device;
        m_config = other.m_config;
        m_buffer = other.m_buffer;
        m_memory = other.m_memory;
        m_particle_count = other.m_particle_count;
        m_buffer_size = other.m_buffer_size;

        other.m_buffer = nullptr;
        other.m_memory = nullptr;
    }
    return *this;
}

std::expected<void, std::string> ParticleBuffer::create_buffer() {
    // Create device-local buffer
    auto buffer_info = vk::BufferCreateInfo()
        .setSize(m_buffer_size)
        .setUsage(vk::BufferUsageFlagBits::eStorageBuffer |
                  vk::BufferUsageFlagBits::eVertexBuffer |
                  vk::BufferUsageFlagBits::eTransferDst |
                  m_config.additional_usage_flags)
        .setSharingMode(vk::SharingMode::eExclusive);

    try {
        m_buffer = m_device.createBuffer(buffer_info);
    } catch (const vk::SystemError& e) {
        return std::unexpected(std::format("Failed to create buffer: {}", e.what()));
    }

    // Allocate device-local memory
    auto mem_reqs = m_device.getBufferMemoryRequirements(m_buffer);

    auto memory_type_result = find_memory_type(
        mem_reqs.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );
    if (!memory_type_result) {
        m_device.destroyBuffer(m_buffer);
        m_buffer = nullptr;
        return std::unexpected(memory_type_result.error());
    }

    auto alloc_info = vk::MemoryAllocateInfo()
        .setAllocationSize(mem_reqs.size)
        .setMemoryTypeIndex(*memory_type_result);

    try {
        m_memory = m_device.allocateMemory(alloc_info);
        m_device.bindBufferMemory(m_buffer, m_memory, 0);
    } catch (const vk::SystemError& e) {
        m_device.destroyBuffer(m_buffer);
        m_buffer = nullptr;
        return std::unexpected(std::format("Failed to allocate memory: {}", e.what()));
    }

    return {};
}

void ParticleBuffer::destroy_buffer() {
    if (m_buffer) {
        m_device.destroyBuffer(m_buffer);
        m_buffer = nullptr;
    }
    if (m_memory) {
        m_device.freeMemory(m_memory);
        m_memory = nullptr;
    }
}

std::expected<uint32_t, std::string> ParticleBuffer::find_memory_type(
    uint32_t type_filter,
    vk::MemoryPropertyFlags properties
) const {
    auto mem_props = m_context->physical_device().getMemoryProperties();

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    return std::unexpected("Failed to find suitable memory type");
}

std::expected<void, std::string> ParticleBuffer::resize(uint32_t new_particle_count) {
    Logger::instance().info("Resizing particle buffer: {} -> {} particles",
        m_particle_count, new_particle_count);

    // Destroy old buffer
    destroy_buffer();

    // Update size
    m_particle_count = new_particle_count;
    m_buffer_size = new_particle_count * sizeof(Particle);

    // Create new buffer
    if (auto result = create_buffer(); !result) {
        return std::unexpected(result.error());
    }

    return {};
}

std::expected<void, std::string> ParticleBuffer::initialize_random(
    vk::CommandPool cmd_pool,
    vk::Queue queue,
    uint32_t seed
) {
    // Generate random particles on CPU
    std::vector<Particle> particles(m_particle_count);

    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto& particle : particles) {
        particle.position = glm::vec3(dist(rng), dist(rng), dist(rng));
        particle.padding1 = 0.0f;
        particle.color = glm::vec4(dist(rng), dist(rng), dist(rng), 1.0f);
    }

    // Create staging buffer
    auto staging_buffer_info = vk::BufferCreateInfo()
        .setSize(m_buffer_size)
        .setUsage(vk::BufferUsageFlagBits::eTransferSrc)
        .setSharingMode(vk::SharingMode::eExclusive);

    vk::Buffer staging_buffer;
    try {
        staging_buffer = m_device.createBuffer(staging_buffer_info);
    } catch (const vk::SystemError& e) {
        return std::unexpected(std::format("Failed to create staging buffer: {}", e.what()));
    }

    // Allocate host-visible memory
    auto staging_mem_reqs = m_device.getBufferMemoryRequirements(staging_buffer);

    auto memory_type_result = find_memory_type(
        staging_mem_reqs.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );
    if (!memory_type_result) {
        m_device.destroyBuffer(staging_buffer);
        return std::unexpected(memory_type_result.error());
    }

    auto staging_alloc_info = vk::MemoryAllocateInfo()
        .setAllocationSize(staging_mem_reqs.size)
        .setMemoryTypeIndex(*memory_type_result);

    vk::DeviceMemory staging_memory;
    try {
        staging_memory = m_device.allocateMemory(staging_alloc_info);
        m_device.bindBufferMemory(staging_buffer, staging_memory, 0);
    } catch (const vk::SystemError& e) {
        m_device.destroyBuffer(staging_buffer);
        return std::unexpected(std::format("Failed to allocate staging memory: {}", e.what()));
    }

    // Copy particle data to staging buffer
    void* data = m_device.mapMemory(staging_memory, 0, m_buffer_size);
    std::memcpy(data, particles.data(), m_buffer_size);
    m_device.unmapMemory(staging_memory);

    // Create one-time command buffer
    auto cmd_alloc_info = vk::CommandBufferAllocateInfo()
        .setCommandPool(cmd_pool)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(1);

    vk::CommandBuffer cmd;
    try {
        cmd = m_device.allocateCommandBuffers(cmd_alloc_info)[0];
    } catch (const vk::SystemError& e) {
        m_device.destroyBuffer(staging_buffer);
        m_device.freeMemory(staging_memory);
        return std::unexpected(std::format("Failed to allocate command buffer: {}", e.what()));
    }

    // Record transfer command
    auto begin_info = vk::CommandBufferBeginInfo()
        .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    try {
        cmd.begin(begin_info);

        auto copy_region = vk::BufferCopy()
            .setSrcOffset(0)
            .setDstOffset(0)
            .setSize(m_buffer_size);

        cmd.copyBuffer(staging_buffer, m_buffer, copy_region);

        cmd.end();
    } catch (const vk::SystemError& e) {
        m_device.freeCommandBuffers(cmd_pool, cmd);
        m_device.destroyBuffer(staging_buffer);
        m_device.freeMemory(staging_memory);
        return std::unexpected(std::format("Failed to record transfer command: {}", e.what()));
    }

    // Submit and wait
    auto submit_info = vk::SubmitInfo()
        .setCommandBuffers(cmd);

    try {
        queue.submit(submit_info);
        queue.waitIdle();
    } catch (const vk::SystemError& e) {
        m_device.freeCommandBuffers(cmd_pool, cmd);
        m_device.destroyBuffer(staging_buffer);
        m_device.freeMemory(staging_memory);
        return std::unexpected(std::format("Failed to submit transfer: {}", e.what()));
    }

    // Cleanup
    m_device.freeCommandBuffers(cmd_pool, cmd);
    m_device.destroyBuffer(staging_buffer);
    m_device.freeMemory(staging_memory);

    Logger::instance().info("Initialized particle buffer with random data");

    return {};
}

} // namespace ifs
