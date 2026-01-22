#pragma once

#include "../IFSFrontend.hpp"
#include "../Shader.hpp"
#include "../VulkanContext.hpp"
#include <memory>
#include <expected>

namespace ifs {

/**
 * @brief Particle point rendering frontend
 *
 * Renders particles as GL_POINTS with configurable point size.
 * Uses the particle color directly from the buffer.
 *
 * Features:
 * - 3D camera support via view-projection matrix
 * - Adjustable point size
 * - Per-particle colors
 * - Dynamic viewport/scissor handling
 */
class ParticleRenderer : public IFSFrontend {
public:
    /**
     * @brief Create ParticleRenderer frontend
     *
     * @param context Vulkan context
     * @param device Vulkan device
     * @param render_pass Render pass for graphics pipeline
     * @param initial_extent Initial framebuffer extent
     * @return ParticleRenderer instance or error message
     */
    static std::expected<std::unique_ptr<ParticleRenderer>, std::string> create(
        const VulkanContext& context,
        vk::Device device,
        vk::RenderPass render_pass,
        const vk::Extent2D& initial_extent
    );

    ~ParticleRenderer() override;

    // Non-copyable, movable
    ParticleRenderer(const ParticleRenderer&) = delete;
    ParticleRenderer& operator=(const ParticleRenderer&) = delete;
    ParticleRenderer(ParticleRenderer&&) noexcept;
    ParticleRenderer& operator=(ParticleRenderer&&) noexcept;

    // IFSFrontend interface
    [[nodiscard]] std::string_view name() const override { return "Point Particles"; }

    // Phase 3: Frontend owns graphics infrastructure
    [[nodiscard]] vk::Semaphore render_frame(
        const FrameRenderInfo& info,
        vk::Queue graphics_queue
    ) override;

    void handle_swapchain_recreation(uint32_t new_image_count) override;

    // Legacy method for manual command buffer recording
    void render(
        vk::CommandBuffer cmd,
        vk::Buffer particle_buffer,
        uint32_t particle_count,
        Camera& camera,
        const vk::Extent2D* extent = nullptr
    ) override;

    void resize(const vk::Extent2D& new_extent) override;

    [[nodiscard]] std::vector<std::pair<std::string, std::pair<float, float>>>
    get_render_parameters() const override {
        return {
            {"point_size", {1.0f, 10.0f}}
        };
    }

    [[nodiscard]] std::vector<UICallback> get_ui_callbacks() override {
        std::vector<UICallback> callbacks;
        callbacks.emplace_back("Point Size", ContinuousCallback{
            .setter = [this](float v) { set_point_size(v); },
            .getter = [this]() { return point_size(); },
            .min = 1.0f,
            .max = 10.0f,
            .logarithmic = false
        });
        return callbacks;
    }

    /**
     * @brief Set point size for rendering
     */
    void set_point_size(float size) { m_point_size = size; }

    [[nodiscard]] float point_size() const { return m_point_size; }

    /**
     * @brief Update particle buffer binding in descriptor set
     *
     * Call this after creating the particle buffer or when it changes (e.g., resize).
     * Do not call every frame - descriptor sets should not be updated while in use.
     */
    void update_particle_buffer(vk::Buffer particle_buffer) override;

private:
    // Private constructor - use create() factory
    ParticleRenderer(
        const VulkanContext& context,
        vk::Device device,
        vk::RenderPass render_pass,
        const vk::Extent2D& initial_extent
    );

    /**
     * @brief Initialize graphics pipeline and resources
     */
    std::expected<void, std::string> initialize();

    /**
     * @brief Create descriptor set layout from shader reflection
     */
    std::expected<void, std::string> create_descriptor_layout();

    /**
     * @brief Create graphics pipeline
     */
    std::expected<void, std::string> create_pipeline();

    /**
     * @brief Cleanup Vulkan resources
     */
    void cleanup();

    const VulkanContext* m_context;
    vk::Device m_device;
    vk::RenderPass m_render_pass;
    vk::Extent2D m_extent;

    // Shaders and pipeline
    std::unique_ptr<Shader> m_vertex_shader;
    std::unique_ptr<Shader> m_fragment_shader;
    vk::DescriptorSetLayout m_descriptor_layout;
    vk::PipelineLayout m_pipeline_layout;
    vk::Pipeline m_graphics_pipeline;

    // Descriptor management
    vk::DescriptorPool m_descriptor_pool;
    vk::DescriptorSet m_descriptor_set;

    // View parameter buffer (uniform buffer for camera/view data)
    vk::Buffer m_view_buffer;
    vk::DeviceMemory m_view_memory;

    // Rendering parameters
    float m_point_size;

    // Phase 3: Graphics command infrastructure (frontend owns graphics resources)
    vk::CommandPool m_graphics_command_pool;
    std::vector<vk::CommandBuffer> m_command_buffers;  // One per swapchain image
    std::vector<vk::Semaphore> m_render_finished_semaphores;  // One per swapchain image
    std::vector<vk::Fence> m_in_flight_fences;  // For frames-in-flight
    std::vector<vk::Fence> m_images_in_flight;  // Track which fence is using which image
    vk::Queue m_graphics_queue;

    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 2;
};

} // namespace ifs
