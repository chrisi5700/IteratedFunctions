#pragma once

#include <glm/glm.hpp>
#include <cstdint>

namespace ifs {

/**
 * @brief Abstract camera interface
 *
 * Provides a common interface for both 2D and 3D cameras.
 * All cameras must provide a view-projection matrix.
 */
class Camera {
public:
    virtual ~Camera() = default;

    /**
     * @brief Get the combined view-projection matrix
     *
     * This matrix transforms world coordinates to clip space.
     */
    [[nodiscard]] virtual glm::mat4 view_projection_matrix() = 0;

    /**
     * @brief Handle window/viewport resize
     *
     * @param width New viewport width
     * @param height New viewport height
     */
    virtual void handle_resize(uint32_t width, uint32_t height) = 0;

    /**
     * @brief Get the camera's position in world space
     *
     * @return Camera position
     */
    [[nodiscard]] virtual glm::vec3 position() = 0;
};

} // namespace ifs
