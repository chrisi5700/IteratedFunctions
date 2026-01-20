#pragma once

#include "ParticleData.hpp"
#include "VulkanContext.hpp"
#include <expected>
#include <string>
#include <random>

namespace ifs {

/**
 * @brief RAII wrapper for particle storage buffer
 *
 * Manages a device-local storage buffer containing particle data.
 * Provides initialization, resizing, and descriptor binding functionality.
 * Used as the shared interface between IFS backends (compute) and frontends (rendering).
 */
class ParticleBuffer {
public:
    /**
     * @brief Create a particle buffer
     *
     * @param context Vulkan context
     * @param device Vulkan device
     * @param config Buffer configuration
     * @return ParticleBuffer on success, error message on failure
     */
    static std::expected<ParticleBuffer, std::string> create(
        const VulkanContext& context,
        vk::Device device,
        const ParticleBufferConfig& config
    );

    ~ParticleBuffer();

    // Non-copyable
    ParticleBuffer(const ParticleBuffer&) = delete;
    ParticleBuffer& operator=(const ParticleBuffer&) = delete;

    // Movable
    ParticleBuffer(ParticleBuffer&& other) noexcept;
    ParticleBuffer& operator=(ParticleBuffer&& other) noexcept;

    /**
     * @brief Get the underlying Vulkan buffer
     */
    [[nodiscard]] vk::Buffer buffer() const { return m_buffer; }

    /**
     * @brief Get the current particle count
     */
    [[nodiscard]] uint32_t particle_count() const { return m_particle_count; }

    /**
     * @brief Get descriptor buffer info for binding
     */
    [[nodiscard]] vk::DescriptorBufferInfo get_descriptor_info() const {
        return vk::DescriptorBufferInfo()
            .setBuffer(m_buffer)
            .setOffset(0)
            .setRange(VK_WHOLE_SIZE);
    }

    /**
     * @brief Resize the particle buffer
     *
     * This will destroy the current buffer and create a new one.
     * The new buffer will be initialized with random particle positions.
     *
     * @param new_particle_count New number of particles
     * @return void on success, error message on failure
     */
    std::expected<void, std::string> resize(uint32_t new_particle_count);

    /**
     * @brief Initialize buffer with random particle positions
     *
     * Particles are distributed randomly in [0,1]³ space with random colors.
     * Uses a staging buffer for CPU-side generation followed by GPU transfer.
     *
     * @param cmd_pool Command pool for transfer commands
     * @param queue Queue for transfer submission (must support transfer operations)
     * @param seed Random seed (0 = use random device)
     * @return void on success, error message on failure
     */
    std::expected<void, std::string> initialize_random(
        vk::CommandPool cmd_pool,
        vk::Queue queue,
        uint32_t seed = 0
    );

private:
    // Private constructor - use create() factory
    ParticleBuffer(
        const VulkanContext& context,
        vk::Device device,
        const ParticleBufferConfig& config
    );

    /**
     * @brief Internal buffer creation
     */
    std::expected<void, std::string> create_buffer();

    /**
     * @brief Internal buffer cleanup
     */
    void destroy_buffer();

    /**
     * @brief Find suitable memory type
     */
    std::expected<uint32_t, std::string> find_memory_type(
        uint32_t type_filter,
        vk::MemoryPropertyFlags properties
    ) const;

    const VulkanContext* m_context;
    vk::Device m_device;
    ParticleBufferConfig m_config;

    vk::Buffer m_buffer;
    vk::DeviceMemory m_memory;
    uint32_t m_particle_count;
    vk::DeviceSize m_buffer_size;
};

} // namespace ifs
