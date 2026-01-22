#pragma once

#include <ifs/IFSFrontend.hpp>
#include <ifs/VulkanContext.hpp>
#include <ifs/Shader.hpp>
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <expected>
#include <string>
#include <string_view>
#include <cstdint>

namespace ifs {

/**
 * @brief Frontend that renders particles as instanced 3D spheres
 *
 * Uses GPU instancing to render each particle as a small sphere with proper
 * 3D geometry, lighting, and depth testing. Much better visual quality than
 * point sprites or billboards.
 */
class SphereRenderer : public IFSFrontend {
public:
    ~SphereRenderer() override;

    /**
     * @brief Create a SphereRenderer
     * @param context Vulkan context
     * @param device Vulkan device
     * @param render_pass Render pass to use
     * @param extent Initial swapchain extent
     * @param sphere_subdivisions Number of subdivisions for sphere mesh (higher = smoother)
     * @return SphereRenderer instance or error
     */
    [[nodiscard]] static std::expected<std::unique_ptr<SphereRenderer>, std::string> create(
        const VulkanContext& context,
        vk::Device device,
        vk::RenderPass render_pass,
        const vk::Extent2D& extent,
        uint32_t sphere_subdivisions = 2
    );

    // IFSFrontend interface implementation
    [[nodiscard]] std::string_view name() const override { return "SphereRenderer"; }

    void resize(const vk::Extent2D& new_extent) override;

    /**
     * @brief Update the particle buffer binding (call once, not every frame)
     */
    void update_particle_buffer(vk::Buffer particle_buffer) override;

    /**
     * @brief Set the sphere radius for rendering
     */
    void set_sphere_radius(float radius) { m_sphere_radius = radius; }

    /**
     * @brief Get the current sphere radius
     */
    float sphere_radius() const { return m_sphere_radius; }

    void render(
        vk::CommandBuffer cmd,
        vk::Buffer particle_buffer,
        uint32_t particle_count,
        Camera& camera,
        const vk::Extent2D* extent = nullptr
    ) override;

    [[nodiscard]] vk::Semaphore render_frame(
        const FrameRenderInfo& info,
        vk::Queue graphics_queue
    ) override;

    void handle_swapchain_recreation(uint32_t new_image_count) override;

private:
    SphereRenderer(const VulkanContext& context, vk::Device device);

    /**
     * @brief Generate sphere mesh geometry
     */
    void generate_sphere_mesh(uint32_t subdivisions);

    /**
     * @brief Create vertex and index buffers for sphere mesh
     */
    [[nodiscard]] std::expected<void, std::string> create_sphere_buffers();

    /**
     * @brief Create descriptor set layout
     */
    [[nodiscard]] std::expected<void, std::string> create_descriptor_layout();

    /**
     * @brief Create graphics pipeline
     */
    [[nodiscard]] std::expected<void, std::string> create_pipeline();

    /**
     * @brief Create descriptor set
     */
    [[nodiscard]] std::expected<void, std::string> create_descriptor_set();

    // Vulkan context
    const VulkanContext* m_context;
    vk::Device m_device;
    vk::RenderPass m_render_pass;
    vk::Extent2D m_extent;

    // Shaders
    std::unique_ptr<Shader> m_vertex_shader;
    std::unique_ptr<Shader> m_fragment_shader;

    // Pipeline
    vk::PipelineLayout m_pipeline_layout;
    vk::Pipeline m_graphics_pipeline;

    // Descriptor sets
    vk::DescriptorSetLayout m_descriptor_layout;
    vk::DescriptorPool m_descriptor_pool;
    vk::DescriptorSet m_descriptor_set;

    // Sphere mesh data
    struct Vertex {
        glm::vec3 position;
        glm::vec3 normal;
    };
    std::vector<Vertex> m_sphere_vertices;
    std::vector<uint32_t> m_sphere_indices;

    // Sphere mesh buffers
    vk::Buffer m_vertex_buffer;
    vk::DeviceMemory m_vertex_memory;
    vk::Buffer m_index_buffer;
    vk::DeviceMemory m_index_memory;

    // View parameters buffer
    vk::Buffer m_view_buffer;
    vk::DeviceMemory m_view_memory;
    void* m_view_mapped = nullptr;

    // Graphics infrastructure (Phase 3: owned by frontend)
    vk::CommandPool m_graphics_command_pool;
    std::vector<vk::CommandBuffer> m_command_buffers;
    std::vector<vk::Semaphore> m_render_finished_semaphores;
    std::vector<vk::Fence> m_in_flight_fences;
    std::vector<vk::Fence> m_images_in_flight;
    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 2;

    // Shader parameters
    struct ViewParams {
        glm::mat4 view_projection;
        glm::vec3 camera_pos;
        float sphere_radius;
        glm::vec3 light_dir;
        float padding;
    };

    // Rendering parameters
    float m_sphere_radius = 0.003f;
};

} // namespace ifs
