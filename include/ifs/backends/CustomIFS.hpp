#pragma once

#include "../IFSBackend.hpp"
#include "../Shader.hpp"
#include "../ParticleBuffer.hpp"
#include <memory>

namespace ifs {

/**
 * @brief 2D Sierpinski Triangle IFS backend
 *
 * Generates a classic Sierpinski triangle fractal using 3 affine transformations.
 * This is a 2D fractal, so all particles will have z=0.
 *
 * Algorithm:
 * - Start with random position in [0,1]²
 * - For each iteration:
 *   1. Randomly select one of 3 transforms (corner vertices)
 *   2. Apply transform: pos = (pos + vertex) * 0.5
 *   3. Repeat
 *
 * After sufficient iterations, points converge to the Sierpinski triangle attractor.
 *
 * Transform vertices (triangle corners):
 * - T0: (0.0, 0.0)
 * - T1: (1.0, 0.0)
 * - T2: (0.5, 0.866) [equilateral triangle]
 */
class CustomIFS : public IFSBackend {
public:
    /**
     * @brief Create CustomIFS backend
     *
     * @param context Vulkan context
     * @param device Vulkan device
     * @return CustomIFS instance or error message
     */
    static std::expected<std::unique_ptr<CustomIFS>, std::string> create(
        const VulkanContext& context,
        vk::Device device
    );

    ~CustomIFS() override;

    // Non-copyable, movable
    CustomIFS(const CustomIFS&) = delete;
    CustomIFS& operator=(const CustomIFS&) = delete;
    CustomIFS(CustomIFS&&) noexcept;
    CustomIFS& operator=(CustomIFS&&) noexcept;

    // IFSBackend interface
    [[nodiscard]] std::string_view name() const override { return "Barnsley Fern"; }
    [[nodiscard]] uint32_t dimension() const override { return 2; }

    // Phase 2: Backend owns compute infrastructure
    void compute(
        vk::Buffer particle_buffer,
        uint32_t particle_count,
        const IFSParameters& params
    ) override;

    void wait_compute_complete() override;

    // Legacy method for manual command buffer recording
    void dispatch(
        vk::CommandBuffer cmd,
        vk::Buffer particle_buffer,
        uint32_t particle_count,
        const IFSParameters& params
    ) override;

    // Buffer ownership and query methods
    [[nodiscard]] vk::Buffer get_particle_buffer() const override {
        return m_particle_buffer ? m_particle_buffer->buffer() : nullptr;
    }

    [[nodiscard]] uint32_t get_particle_count() const override {
        return m_particle_count;
    }

    [[nodiscard]] std::vector<UICallback> get_ui_callbacks() override;

private:
    // Private constructor - use create() factory
    CustomIFS(
        const VulkanContext& context,
        vk::Device device
    );

    /**
     * @brief Initialize compute pipeline and resources
     */
    std::expected<void, std::string> initialize();

    /**
     * @brief Create descriptor set layout from shader reflection
     */
    std::expected<void, std::string> create_descriptor_layout();

    /**
     * @brief Create compute pipeline
     */
    std::expected<void, std::string> create_pipeline();

    /**
     * @brief Cleanup Vulkan resources
     */
    void cleanup();

    /**
     * @brief Reallocate particle buffer with new count
     *
     * Called when particle_count UI parameter changes.
     * Recreates the buffer and reinitializes with random data.
     *
     * @param new_count New particle count
     */
    void reallocate_particle_buffer(uint32_t new_count);

    const VulkanContext* m_context;
    vk::Device m_device;

    // Particle data (backend owns this)
    std::unique_ptr<ParticleBuffer> m_particle_buffer;
    uint32_t m_particle_count;
	uint32_t m_iteration_count = 100;

    // Shader and pipeline
    std::unique_ptr<Shader> m_compute_shader;
    vk::DescriptorSetLayout m_descriptor_layout;
    vk::PipelineLayout m_pipeline_layout;
    vk::Pipeline m_compute_pipeline;

    // Descriptor management
    vk::DescriptorPool m_descriptor_pool;
    vk::DescriptorSet m_descriptor_set;

    // Parameter buffer (uniform buffer for IFSParams)
    vk::Buffer m_param_buffer;
    vk::DeviceMemory m_param_memory;

    // Phase 2: Compute command infrastructure (backend owns compute resources)
    vk::CommandPool m_compute_command_pool;
    vk::CommandBuffer m_compute_command_buffer;
    vk::Fence m_compute_fence;
    vk::Queue m_compute_queue;
};

} // namespace ifs
