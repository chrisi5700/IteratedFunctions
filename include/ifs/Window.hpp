#pragma once

#include <ifs/Common.hpp>
#include <ifs/VulkanContext.hpp>
#include <vulkan/vulkan.hpp>
#include <string_view>
#include <string>
#include <vector>
#include <expected>
#include <optional>

namespace ifs {

/**
 * @brief GLFW window with integrated Vulkan presentation
 *
 * Owns the complete presentation stack: surface, swapchain, render pass,
 * framebuffers, and image views. Handles swapchain recreation automatically
 * on resize or out-of-date conditions.
 *
 * Responsibilities:
 * - GLFW window lifecycle
 * - Vulkan surface creation
 * - Swapchain management (creation, recreation, acquisition)
 * - Render pass creation
 * - Framebuffer management
 * - Present operations
 */
class Window {
public:
    /**
     * @brief Create a Window with Vulkan presentation
     *
     * @param context Vulkan context for device/queue access
     * @param width Initial window width
     * @param height Initial window height
     * @param title Window title
     * @return Window instance or error message
     */
    static std::expected<Window, std::string> create(
        const VulkanContext& context,
        int width,
        int height,
        std::string_view title
    );

    ~Window();

    // Non-copyable, movable
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    Window(Window&&) noexcept;
    Window& operator=(Window&&) noexcept;

    /**
     * @brief Check if window should close
     */
    [[nodiscard]] bool should_close() const;

    /**
     * @brief Get current window dimensions
     */
    [[nodiscard]] int get_width() const { return m_width; }
    [[nodiscard]] int get_height() const { return m_height; }

    /**
     * @brief Get GLFW window handle (for input/ImGui)
     */
    [[nodiscard]] GLFWwindow* get_window_handle() const { return m_window_handle; }

    /**
     * @brief Get render pass for pipeline creation
     */
    [[nodiscard]] vk::RenderPass render_pass() const { return m_render_pass; }

    /**
     * @brief Get current swapchain extent
     */
    [[nodiscard]] vk::Extent2D extent() const { return m_extent; }

    /**
     * @brief Get number of swapchain images
     */
    [[nodiscard]] uint32_t image_count() const {
        return static_cast<uint32_t>(m_swapchain_images.size());
    }

    /**
     * @brief Get framebuffer for a specific image index
     */
    [[nodiscard]] vk::Framebuffer get_framebuffer(uint32_t index) const {
        return m_framebuffers[index];
    }

    /**
     * @brief Get the current framebuffer (last acquired image)
     */
    [[nodiscard]] vk::Framebuffer get_current_framebuffer() const {
        return m_framebuffers[m_current_image_index];
    }

    /**
     * @brief Acquire next swapchain image
     *
     * Automatically handles swapchain recreation if out-of-date or suboptimal.
     * If recreation occurs, returns nullopt to signal the caller to retry.
     *
     * @param signal_semaphore Semaphore to signal when image is available
     * @param timeout Timeout in nanoseconds
     * @return Image index, or nullopt if swapchain was recreated
     */
    [[nodiscard]] std::optional<uint32_t> acquire_next_image(
        vk::Semaphore signal_semaphore,
        uint64_t timeout = UINT64_MAX
    );

    /**
     * @brief Present rendered image to screen
     *
     * Automatically handles swapchain recreation if out-of-date.
     *
     * @param present_queue Queue to present on (usually graphics queue)
     * @param wait_semaphore Semaphore to wait on before presenting
     * @param image_index Index of image to present
     * @return true if presented successfully, false if swapchain was recreated
     */
    [[nodiscard]] bool present(
        vk::Queue present_queue,
        vk::Semaphore wait_semaphore,
        uint32_t image_index
    );

    /**
     * @brief Mark that window needs resize handling
     *
     * Called by application when window resize is detected.
     * Next acquire_next_image() will recreate the swapchain.
     */
    void mark_resize_needed() { m_needs_resize = true; }

private:
    // Private constructor - use create() factory
    Window(
        const VulkanContext& context,
        int width,
        int height,
        std::string_view title
    );

    /**
     * @brief Ensure GLFW is initialized (static initializer)
     */
    static void ensure_glfw_initialized();

    /**
     * @brief Initialize Vulkan surface and swapchain
     */
    std::expected<void, std::string> initialize();

    /**
     * @brief Create Vulkan surface from GLFW window
     */
    std::expected<void, std::string> create_surface();

    /**
     * @brief Create swapchain and associated resources
     */
    std::expected<void, std::string> create_swapchain();

    /**
     * @brief Create render pass for the swapchain format
     */
    std::expected<void, std::string> create_render_pass();

    /**
     * @brief Create framebuffers for each swapchain image
     */
    std::expected<void, std::string> create_framebuffers();

    /**
     * @brief Create depth buffer resources
     */
    std::expected<void, std::string> create_depth_resources();

    /**
     * @brief Find supported depth format
     */
    vk::Format find_depth_format() const;

    /**
     * @brief Find supported format from candidates
     */
    vk::Format find_supported_format(
        const std::vector<vk::Format>& candidates,
        vk::ImageTiling tiling,
        vk::FormatFeatureFlags features
    ) const;

    /**
     * @brief Recreate swapchain (on resize or out-of-date)
     */
    std::expected<void, std::string> recreate_swapchain();

    /**
     * @brief Cleanup swapchain resources (for recreation)
     */
    void cleanup_swapchain();

    /**
     * @brief Cleanup all Vulkan resources
     */
    void cleanup();

    /**
     * @brief Choose swapchain surface format
     */
    vk::SurfaceFormatKHR choose_surface_format(
        const std::vector<vk::SurfaceFormatKHR>& available_formats
    ) const;

    /**
     * @brief Choose swapchain present mode
     */
    vk::PresentModeKHR choose_present_mode(
        const std::vector<vk::PresentModeKHR>& available_modes
    ) const;

    /**
     * @brief Choose swapchain extent based on capabilities
     */
    vk::Extent2D choose_extent(const vk::SurfaceCapabilitiesKHR& capabilities) const;

    // GLFW initialization (must happen before VulkanContext creation)
    // This static initializer ensures glfwInit() is called before main()
    static inline bool s_glfw_init_guard = []() {
        ensure_glfw_initialized();
        return true;
    }();

    // GLFW window
    GLFWwindow* m_window_handle;
    int m_width;
    int m_height;

    // Vulkan context reference
    const VulkanContext* m_context;
    vk::Device m_device;

    // Presentation resources
    vk::SurfaceKHR m_surface;
    vk::SurfaceFormatKHR m_surface_format;
    vk::PresentModeKHR m_present_mode;
    vk::SwapchainKHR m_swapchain;
    vk::Extent2D m_extent;

    std::vector<vk::Image> m_swapchain_images;
    std::vector<vk::ImageView> m_image_views;
    std::vector<vk::Framebuffer> m_framebuffers;
    vk::RenderPass m_render_pass;

    // Depth buffer resources
    vk::Image m_depth_image;
    vk::DeviceMemory m_depth_memory;
    vk::ImageView m_depth_image_view;
    vk::Format m_depth_format;

    // State tracking
    uint32_t m_current_image_index;
    bool m_needs_resize;
};

} // namespace ifs
