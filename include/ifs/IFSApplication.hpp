#pragma once

#include "VulkanContext.hpp"
#include "Window.hpp"
#include "Camera3D.hpp"
#include "ParticleBuffer.hpp"
#include "IFSBackend.hpp"
#include "IFSFrontend.hpp"
#include <expected>
#include <memory>
#include <vector>

namespace ifs {

/**
 * @brief Main application orchestrator for modular IFS visualization
 *
 * Manages the entire rendering pipeline:
 * - Vulkan initialization (context, swapchain, render pass)
 * - Backend registry (fractal generators)
 * - Frontend registry (rendering systems)
 * - Particle buffer management
 * - ImGui UI for controls
 * - Main render loop
 */
class IFSApplication {
public:
    struct Config {
        uint32_t initial_particle_count = 1'000'000;
        uint32_t window_width = 1280;
        uint32_t window_height = 720;
        std::string_view window_title = "IFS Visualizer - Modular";
    };

    /**
     * @brief Create IFS application
     */
    static std::expected<IFSApplication, std::string> create(const Config& config);

    ~IFSApplication();

    // Non-copyable, movable
    IFSApplication(const IFSApplication&) = delete;
    IFSApplication& operator=(const IFSApplication&) = delete;
    IFSApplication(IFSApplication&&) noexcept;
    IFSApplication& operator=(IFSApplication&&) noexcept;

    /**
     * @brief Run the main application loop
     */
    void run();

private:
    // Private constructor - use create() factory
    IFSApplication(const Config& config);

    /**
     * @brief Initialize Vulkan resources
     */
    std::expected<void, std::string> initialize();

    /**
     * @brief Create swapchain and related resources
     */
    std::expected<void, std::string> create_swapchain();

    /**
     * @brief Create render pass
     */
    std::expected<void, std::string> create_render_pass();

    /**
     * @brief Create framebuffers for swapchain images
     */
    std::expected<void, std::string> create_framebuffers();

    /**
     * @brief Create command pools and buffers
     */
    std::expected<void, std::string> create_command_resources();

    /**
     * @brief Create synchronization objects
     */
    std::expected<void, std::string> create_sync_objects();

    /**
     * @brief Initialize ImGui
     */
    std::expected<void, std::string> initialize_imgui();

    /**
     * @brief Register all backends
     */
    std::expected<void, std::string> register_backends();

    /**
     * @brief Register all frontends
     */
    std::expected<void, std::string> register_frontends();

    /**
     * @brief Handle window resize
     */
    void recreate_swapchain();

    /**
     * @brief Render one frame
     */
    void render_frame();

    /**
     * @brief Build ImGui UI
     */
    void build_ui();

    /**
     * @brief Cleanup Vulkan resources
     */
    void cleanup();
    void cleanup_swapchain();

    // Configuration
    Config m_config;

    // Core Vulkan components
    std::unique_ptr<VulkanContext> m_context;
    std::unique_ptr<Window> m_window;

    // Swapchain and rendering
    vk::SwapchainKHR m_swapchain;
    vk::Format m_swapchain_format;
    vk::Extent2D m_swapchain_extent;
    std::vector<vk::Image> m_swapchain_images;
    std::vector<vk::ImageView> m_swapchain_image_views;
    std::vector<vk::Framebuffer> m_framebuffers;
    vk::RenderPass m_render_pass;

    // Command resources
    vk::CommandPool m_command_pool;
    vk::CommandPool m_compute_command_pool;
    std::vector<vk::CommandBuffer> m_command_buffers;

    // Synchronization
    std::vector<vk::Semaphore> m_image_available_semaphores;
    std::vector<vk::Semaphore> m_render_finished_semaphores;
    std::vector<vk::Fence> m_in_flight_fences;
    uint32_t m_current_frame = 0;
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    // IFS components
    std::unique_ptr<ParticleBuffer> m_particle_buffer;
    std::unique_ptr<Camera3D> m_camera;
    std::vector<std::unique_ptr<IFSBackend>> m_backends;
    std::vector<std::unique_ptr<IFSFrontend>> m_frontends;

    // Current selections
    size_t m_current_backend_index = 0;
    size_t m_current_frontend_index = 0;

    // IFS parameters
    IFSParameters m_ifs_params;
    bool m_needs_recompute = true;

    // ImGui
    vk::DescriptorPool m_imgui_descriptor_pool;

    // State
    bool m_framebuffer_resized = false;
};

} // namespace ifs
