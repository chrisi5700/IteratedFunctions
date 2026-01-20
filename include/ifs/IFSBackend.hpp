#pragma once

#include "VulkanContext.hpp"
#include "UICallback.hpp"
#include <string_view>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

namespace ifs {

/**
 * @brief Parameters for IFS computation
 *
 * Common parameters shared by all IFS backends.
 * Backends can provide additional custom parameters via get_custom_parameter_ranges().
 */
struct IFSParameters {
    uint32_t iteration_count = 20;        ///< Number of iterations per particle
    float scale = 1.0f;                   ///< Global scale factor
    uint32_t random_seed = 0;             ///< Random seed (0 = use random device)
    std::unordered_map<std::string, float> custom_params;  ///< Backend-specific parameters
};

/**
 * @brief Abstract base class for IFS backends (fractal generators)
 *
 * An IFS backend is responsible for computing particle positions and colors
 * via a compute shader. The backend fills a particle buffer with fractal data.
 *
 * Contract:
 * - Backends must fill the particle buffer via compute shader
 * - Must support both 2D (z=0) and 3D positions using the unified Particle structure
 * - Must issue ownership transfer barriers if queue families differ
 * - Can provide custom UI parameters via get_custom_parameter_ranges()
 *
 * Example backends:
 * - Sierpinski2D: 2D triangle fractal (3 affine transforms, z=0)
 * - Sierpinski3D: 3D tetrahedron fractal (4 affine transforms)
 * - MengerSponge: 3D cube fractal (20 affine transforms)
 */
class IFSBackend {
public:
    virtual ~IFSBackend() = default;

    /**
     * @brief Get the backend name for UI display
     */
    [[nodiscard]] virtual std::string_view name() const = 0;

    /**
     * @brief Get the dimension of the fractal
     *
     * @return 2 for 2D fractals (set z=0), 3 for true 3D fractals
     */
    [[nodiscard]] virtual uint32_t dimension() const = 0;

    /**
     * @brief Execute compute shader to fill particle buffer (Phase 2: Backend owns compute infrastructure)
     *
     * Records compute dispatch into backend's own command buffer, submits to compute queue
     * with fence, and returns immediately (asynchronous). Call wait_compute_complete() to
     * ensure completion before rendering.
     *
     * The backend:
     * 1. Records dispatch into its own command buffer
     * 2. Issues ownership release barriers if needed
     * 3. Submits to compute queue with fence
     * 4. Returns immediately
     *
     * @param particle_buffer Device buffer containing particles
     * @param particle_count Number of particles in buffer
     * @param params IFS parameters (iterations, scale, seed, custom)
     */
    virtual void compute(
        vk::Buffer particle_buffer,
        uint32_t particle_count,
        const IFSParameters& params
    ) = 0;

    /**
     * @brief Wait for compute operation to complete (Phase 2: Backend owns compute infrastructure)
     *
     * Blocks until the last compute() call has finished executing.
     * This should be called before rendering to ensure buffer ownership is transferred.
     */
    virtual void wait_compute_complete() = 0;

    /**
     * @brief Dispatch compute shader to fill particle buffer (DEPRECATED - use compute() instead)
     *
     * Legacy method for manual command buffer recording. Prefer using compute() which
     * handles command buffer management and synchronization internally.
     *
     * The backend should:
     * 1. Bind compute pipeline
     * 2. Bind descriptor sets (particle buffer, parameter UBO, etc.)
     * 3. Dispatch compute shader
     * 4. Issue memory barrier if on same queue family
     *    (or call release_buffer_ownership() if different queue families)
     *
     * @param cmd Command buffer to record into
     * @param particle_buffer Device buffer containing particles
     * @param particle_count Number of particles in buffer
     * @param params IFS parameters (iterations, scale, seed, custom)
     */
    virtual void dispatch(
        vk::CommandBuffer cmd,
        vk::Buffer particle_buffer,
        uint32_t particle_count,
        const IFSParameters& params
    ) = 0;

    /**
     * @brief Release particle buffer ownership (compute → graphics)
     *
     * Issues a queue family ownership transfer barrier when compute and graphics
     * queues are from different families. If they're the same queue, issues a
     * simple memory barrier.
     *
     * This is called after dispatch() if the backend and frontend use different
     * queue families. Backends can override this if they need custom behavior.
     *
     * @param cmd Command buffer to record into
     * @param particle_buffer Device buffer containing particles
     * @param compute_queue_family Compute queue family index
     * @param graphics_queue_family Graphics queue family index
     */
    virtual void release_buffer_ownership(
        vk::CommandBuffer cmd,
        vk::Buffer particle_buffer,
        uint32_t compute_queue_family,
        uint32_t graphics_queue_family
    ) const {
        if (compute_queue_family == graphics_queue_family) {
            // Same queue: simple memory barrier
            auto barrier = vk::MemoryBarrier()
                .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                .setDstAccessMask(vk::AccessFlagBits::eVertexAttributeRead);

            cmd.pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eVertexInput,
                {},
                barrier,
                {},
                {}
            );
        } else {
            // Different queues: ownership transfer (release)
            auto barrier = vk::BufferMemoryBarrier()
                .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                .setDstAccessMask({})  // Acquire will set this
                .setSrcQueueFamilyIndex(compute_queue_family)
                .setDstQueueFamilyIndex(graphics_queue_family)
                .setBuffer(particle_buffer)
                .setOffset(0)
                .setSize(VK_WHOLE_SIZE);

            cmd.pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eBottomOfPipe,
                {},
                {},
                barrier,
                {}
            );
        }
    }

    /**
     * @brief Get custom parameter ranges for UI (DEPRECATED - use get_ui_callbacks() instead)
     *
     * Backends can override this to provide additional parameters beyond
     * the standard iteration_count, scale, and random_seed.
     *
     * @return Vector of (parameter_name, (min_value, max_value)) pairs
     *
     * Example:
     *   return {
     *       {"rotation_speed", {0.0f, 10.0f}},
     *       {"hollow_factor", {0.0f, 1.0f}}
     *   };
     */
    [[nodiscard]] virtual std::vector<std::pair<std::string, std::pair<float, float>>>
    get_custom_parameter_ranges() const {
        return {};
    }

    /**
     * @brief Get UI callbacks for backend-specific parameters
     *
     * Backends can override this to expose custom parameters to the UI.
     * The UI rendering code will automatically create appropriate controls
     * based on the callback type (continuous sliders, discrete sliders, toggles).
     *
     * @return Vector of UICallback objects
     *
     * Example:
     *   std::vector<UICallback> callbacks;
     *   callbacks.emplace_back("Rotation Speed", ContinuousCallback{
     *       .setter = [this](float v) { m_rotation_speed = v; },
     *       .getter = [this]() { return m_rotation_speed; },
     *       .min = 0.0f,
     *       .max = 10.0f
     *   });
     *   return callbacks;
     */
    [[nodiscard]] virtual std::vector<UICallback> get_ui_callbacks() {
        return {};
    }

    /**
     * @brief Get the particle buffer for rendering
     *
     * Frontend queries this to get the buffer to render.
     * Backend owns and manages the particle buffer allocation.
     *
     * @return Vulkan buffer handle containing particle data
     */
    [[nodiscard]] virtual vk::Buffer get_particle_buffer() const = 0;

    /**
     * @brief Get the current particle count
     *
     * Frontend queries this to know how many particles to render.
     *
     * @return Number of particles in the buffer
     */
    [[nodiscard]] virtual uint32_t get_particle_count() const = 0;
};

} // namespace ifs
