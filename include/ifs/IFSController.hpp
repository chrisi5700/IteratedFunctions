#pragma once

#include "IFSBackend.hpp"
#include "IFSFrontend.hpp"
#include "Camera3D.hpp"
#include "ParticleBuffer.hpp"
#include "VulkanContext.hpp"
#include "Window.hpp"
#include <GLFW/glfw3.h>
#include <memory>
#include <expected>
#include <functional>

namespace ifs {

/**
 * @brief Configuration for IFS application
 */
struct IFSConfig {
    uint32_t window_width = 1280;
    uint32_t window_height = 720;
    const char* window_title = "IFS Visualizer";
};

/**
 * @brief Controller for IFS visualization (MVC pattern)
 *
 * This class acts as the Controller in an MVC architecture:
 * - Model: IFSBackend (computes fractal data)
 * - View: IFSFrontend (renders particles)
 * - Controller: IFSController (manages interaction, UI, input)
 *
 * The controller handles:
 * - Window and Vulkan context lifecycle
 * - Camera and input management
 * - UI rendering and parameter updates
 * - Coordination between backend (compute) and frontend (graphics)
 * - Main render loop
 */
class IFSController {
public:
    /**
     * @brief Create an IFS controller
     *
     * @param config Application configuration
     * @return Controller instance or error message
     */
    static std::expected<std::unique_ptr<IFSController>, std::string> create(
        const IFSConfig& config = {}
    );

    ~IFSController();

    // Non-copyable, movable
    IFSController(const IFSController&) = delete;
    IFSController& operator=(const IFSController&) = delete;
    IFSController(IFSController&&) noexcept = default;
    IFSController& operator=(IFSController&&) noexcept = default;

    /**
     * @brief Set the backend (model) - must be called before run()
     *
     * @param backend Backend instance
     */
    void set_backend(std::unique_ptr<IFSBackend> backend);

    /**
     * @brief Set the frontend (view) - must be called before run()
     *
     * @param frontend Frontend instance
     */
    void set_frontend(std::unique_ptr<IFSFrontend> frontend);

    /**
     * @brief Run the main application loop
     *
     * Blocks until the window is closed.
     * Backend and frontend must be set before calling this.
     *
     * @return Error message if something goes wrong, or void on success
     */
    std::expected<void, std::string> run();

    /**
     * @brief Get the Vulkan context (for creating backends/frontends)
     */
    [[nodiscard]] const VulkanContext& context() const { return *m_context; }

    /**
     * @brief Get the Vulkan device (for creating backends/frontends)
     */
    [[nodiscard]] vk::Device device() const { return m_context->device(); }

    /**
     * @brief Get the render pass (for creating frontends)
     */
    [[nodiscard]] vk::RenderPass render_pass() const { return m_window->render_pass(); }

    /**
     * @brief Get the current extent (for creating frontends)
     */
    [[nodiscard]] vk::Extent2D extent() const { return m_window->extent(); }

private:
    IFSController(const IFSConfig& config);

    /**
     * @brief Initialize Vulkan, window, and resources
     */
    std::expected<void, std::string> initialize();

    /**
     * @brief Setup ImGui
     */
    std::expected<void, std::string> setup_imgui();

    /**
     * @brief Handle input for camera control
     */
    void handle_input(float delta_time);

    /**
     * @brief Render ImGui UI
     */
    void render_ui();

    /**
     * @brief Render UI callbacks generically
     */
    void render_ui_callbacks(const std::vector<UICallback>& callbacks);

    /**
     * @brief Cleanup resources
     */
    void cleanup();

    // GLFW callbacks (friends to access private members)
    friend void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    friend void glfw_mouse_callback(GLFWwindow* window, double xpos, double ypos);
    friend void glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

    // Configuration
    IFSConfig m_config;

    // Core Vulkan resources
    std::unique_ptr<VulkanContext> m_context;
    std::unique_ptr<Window> m_window;

    // MVC components
    std::unique_ptr<IFSBackend> m_backend;   // Model (owns particle buffer)
    std::unique_ptr<IFSFrontend> m_frontend; // View

    // Camera and input
    std::unique_ptr<Camera3D> m_camera;
    bool m_keys_pressed[512] = {false};
    double m_last_mouse_x = 0.0;
    double m_last_mouse_y = 0.0;
    bool m_first_mouse = true;
    bool m_mouse_captured = false;

    // IFS parameters
    IFSParameters m_ifs_params;
    bool m_needs_recompute = true;
    bool m_needs_ownership_acquire = false;
    bool m_needs_buffer_rebind = false;  // Frontend needs to rebind particle buffer

    // ImGui descriptor pool
    vk::DescriptorPool m_imgui_descriptor_pool;

    // Timing
    double m_last_frame_time = 0.0;

    // Frame counter for in-flight synchronization
    uint32_t m_current_frame = 0;
    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 2;
};

} // namespace ifs
