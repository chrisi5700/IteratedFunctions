#pragma once

#include "Camera.hpp"
#include "UICallback.hpp"
#include <vulkan/vulkan.hpp>
#include <string_view>
#include <string>
#include <vector>
#include <utility>
#include <array>

namespace ifs {

/**
 * @brief Information needed to render a frame (Phase 3: Frontend owns graphics infrastructure)
 */
struct FrameRenderInfo {
    uint32_t image_index;                      ///< Swapchain image index
    uint32_t current_frame;                    ///< Frame-in-flight index (for fence cycling)
    vk::Semaphore image_available_semaphore;   ///< Semaphore signaled when image is available
    vk::Framebuffer framebuffer;               ///< Target framebuffer
    vk::Extent2D extent;                       ///< Render area extent
    vk::RenderPass render_pass;                ///< Render pass to use
    std::array<vk::ClearValue, 2> clear_values;///< Clear values for render pass (color, depth)
    vk::Buffer particle_buffer;                ///< Particle buffer to render
    uint32_t particle_count;                   ///< Number of particles
    Camera& camera;                            ///< Camera for view/projection
    bool needs_ownership_acquire;              ///< Whether buffer ownership needs to be acquired
    uint32_t compute_queue_family;             ///< Compute queue family (for ownership transfer)
    uint32_t graphics_queue_family;            ///< Graphics queue family (for ownership transfer)
    void* imgui_draw_data;                     ///< ImGui draw data (optional)
};

/**
 * @brief Abstract base class for IFS frontends (rendering systems)
 *
 * An IFS frontend is responsible for rendering particles from a particle buffer.
 * The frontend reads particle data and visualizes it using various techniques.
 *
 * Contract:
 * - Frontends must read from the particle buffer (as SSBO or vertex input)
 * - Must handle variable particle counts
 * - Must acquire buffer ownership if queue families differ
 * - Must use Camera3D for view/projection matrices in 3D rendering
 * - Must handle viewport/scissor updates on resize
 *
 * Example frontends:
 * - ParticleRenderer: Renders particles as GL_POINTS with point size control
 * - InstancedRenderer: Renders instanced spheres/cubes at particle positions
 */
class IFSFrontend {
public:
    virtual ~IFSFrontend() = default;

    /**
     * @brief Get the frontend name for UI display
     */
    [[nodiscard]] virtual std::string_view name() const = 0;

    /**
     * @brief Render a complete frame (Phase 3: Frontend owns graphics infrastructure)
     *
     * Records commands, submits to graphics queue with proper synchronization,
     * and returns the semaphore to wait on for presentation.
     *
     * The frontend handles:
     * 1. Fence synchronization (wait for frame-in-flight fence)
     * 2. Command buffer recording
     * 3. Buffer ownership acquisition (if needed)
     * 4. Render pass begin/end
     * 5. Particle rendering
     * 6. ImGui rendering (if provided)
     * 7. Queue submission with semaphores
     *
     * @param info Frame rendering information
     * @param graphics_queue Graphics queue for submission
     * @return Semaphore signaled when rendering is complete (for presentation)
     */
    [[nodiscard]] virtual vk::Semaphore render_frame(
        const FrameRenderInfo& info,
        vk::Queue graphics_queue
    ) = 0;

    /**
     * @brief Handle swapchain recreation
     *
     * Called when swapchain is recreated (e.g., window resize).
     * Frontend should recreate per-image resources (semaphores, command buffers).
     *
     * @param new_image_count New number of swapchain images
     */
    virtual void handle_swapchain_recreation(uint32_t new_image_count) = 0;

    /**
     * @brief Render particles to the current render pass (DEPRECATED - use render_frame() instead)
     *
     * The frontend should:
     * 1. Bind graphics pipeline
     * 2. Bind descriptor sets (particle buffer, camera UBO, etc.)
     * 3. Update dynamic state (viewport, scissor)
     * 4. Issue draw call(s)
     *
     * This is called inside an active render pass. The frontend must not
     * begin or end the render pass.
     *
     * @param cmd Command buffer to record into
     * @param particle_buffer Device buffer containing particles
     * @param particle_count Number of particles to render
     * @param camera Camera for view/projection matrices
     * @param extent Viewport extent (if not provided, uses stored extent)
     */
    virtual void render(
        vk::CommandBuffer cmd,
        vk::Buffer particle_buffer,
        uint32_t particle_count,
        Camera& camera,
        const vk::Extent2D* extent = nullptr
    ) = 0;

    /**
     * @brief Acquire particle buffer ownership (compute → graphics)
     *
     * Issues a queue family ownership transfer barrier when compute and graphics
     * queues are from different families. If they're the same queue, does nothing
     * (the backend's release barrier is sufficient).
     *
     * This is called before render() if the backend and frontend use different
     * queue families. Frontends can override this if they need custom behavior.
     *
     * @param cmd Command buffer to record into
     * @param particle_buffer Device buffer containing particles
     * @param compute_queue_family Compute queue family index
     * @param graphics_queue_family Graphics queue family index
     */
    virtual void acquire_buffer_ownership(
        vk::CommandBuffer cmd,
        vk::Buffer particle_buffer,
        uint32_t compute_queue_family,
        uint32_t graphics_queue_family
    ) const {
        if (compute_queue_family != graphics_queue_family) {
            // Different queues: ownership transfer (acquire)
            auto barrier = vk::BufferMemoryBarrier()
                .setSrcAccessMask({})  // Release already set this
                .setDstAccessMask(vk::AccessFlagBits::eVertexAttributeRead)
                .setSrcQueueFamilyIndex(compute_queue_family)
                .setDstQueueFamilyIndex(graphics_queue_family)
                .setBuffer(particle_buffer)
                .setOffset(0)
                .setSize(VK_WHOLE_SIZE);

            cmd.pipelineBarrier(
                vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eVertexInput,
                {},
                {},
                barrier,
                {}
            );
        }
    }

    /**
     * @brief Handle viewport/framebuffer resize
     *
     * Called when the window or render target changes size.
     * Frontends should update internal state (viewport, scissor, etc.)
     * and may need to recreate pipelines if they have baked-in dimensions.
     *
     * @param new_extent New framebuffer dimensions
     */
    virtual void resize(const vk::Extent2D& new_extent) = 0;

    /**
     * @brief Update the particle buffer binding in descriptor sets
     *
     * Called when the backend creates or recreates the particle buffer.
     * Frontends should update their descriptor sets to point to the new buffer.
     *
     * @param particle_buffer The particle buffer to bind
     */
    virtual void update_particle_buffer(vk::Buffer particle_buffer) = 0;

    /**
     * @brief Get render parameter ranges for UI (DEPRECATED - use get_ui_callbacks() instead)
     *
     * Frontends can override this to provide adjustable rendering parameters.
     *
     * @return Vector of (parameter_name, (min_value, max_value)) pairs
     *
     * Example:
     *   return {
     *       {"point_size", {1.0f, 10.0f}},
     *       {"instance_scale", {0.001f, 0.1f}}
     *   };
     */
    [[nodiscard]] virtual std::vector<std::pair<std::string, std::pair<float, float>>>
    get_render_parameters() const {
        return {};
    }

    /**
     * @brief Get UI callbacks for frontend-specific rendering parameters
     *
     * Frontends can override this to expose rendering parameters to the UI.
     * The UI rendering code will automatically create appropriate controls
     * based on the callback type (continuous sliders, discrete sliders, toggles).
     *
     * @return Vector of UICallback objects
     *
     * Example:
     *   std::vector<UICallback> callbacks;
     *   callbacks.emplace_back("Point Size", ContinuousCallback{
     *       .setter = [this](float v) { m_point_size = v; },
     *       .getter = [this]() { return m_point_size; },
     *       .min = 1.0f,
     *       .max = 10.0f
     *   });
     *   return callbacks;
     */
    [[nodiscard]] virtual std::vector<UICallback> get_ui_callbacks() {
        return {};
    }
};

} // namespace ifs
