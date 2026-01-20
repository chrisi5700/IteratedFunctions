#pragma once

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>
#include <cstdint>

namespace ifs {

/**
 * @brief Unified particle structure for IFS visualization
 *
 * This structure is used as the common interface between backends (fractal generators)
 * and frontends (rendering systems). Backends write particle positions and colors,
 * frontends read them for visualization.
 *
 * Layout is designed for:
 * - Cache efficiency (32-byte alignment)
 * - Support for both 2D (z=0) and 3D fractals
 * - Per-particle color variation
 */
struct Particle {
    glm::vec3 position;  ///< World-space position (2D backends set z=0)
    float padding1;      ///< Padding for 16-byte alignment
    glm::vec4 color;     ///< RGBA color (0.0-1.0 range)

    // Total size: 32 bytes (cache-line friendly on most architectures)
};
static_assert(sizeof(Particle) == 32, "Particle must be exactly 32 bytes");

/**
 * @brief Configuration for particle buffer creation
 */
struct ParticleBufferConfig {
    /// Number of particles to allocate
    uint32_t particle_count = 1'000'000;

    /// Whether the buffer should support dynamic resizing
    /// (If false, resize operations will destroy and recreate the buffer)
    bool support_dynamic_resize = false;

    /// Additional usage flags beyond the default
    /// Default flags: STORAGE_BUFFER_BIT | VERTEX_BUFFER_BIT
    vk::BufferUsageFlags additional_usage_flags = {};
};

} // namespace ifs
