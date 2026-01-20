#pragma once

#include "Camera.hpp"

// Force GLM to use Vulkan's depth range [0, 1] instead of OpenGL's [-1, 1]
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cstdint>

namespace ifs {

/**
 * @brief Orbital 3D camera with focus point movement
 *
 * Implements an orbital camera that rotates around a focus point:
 * - WASD/QE movement (moves the focus point in camera-relative directions)
 * - Mouse drag for orbit rotation (azimuth/elevation)
 * - Scroll wheel for zoom (distance from focus)
 * - Perspective projection
 * - Automatic aspect ratio handling
 * - Lazy matrix computation with dirty flags
 */
class Camera3D : public Camera {
public:
    /**
     * @brief Construct camera with default parameters
     *
     * Default camera:
     * - Target: (0.5, 0.5, 0.0) - center of Sierpinski triangle
     * - Distance: 1.5 units from target
     * - Azimuth: -90° (facing negative Z)
     * - Elevation: -35° (looking down)
     * - FOV: 60°
     * - Move speed: 0.5 units/sec
     */
    Camera3D(uint32_t viewport_width = 1280, uint32_t viewport_height = 720);

    /**
     * @brief Get the view matrix (world to camera space)
     */
    [[nodiscard]] glm::mat4 view_matrix();

    /**
     * @brief Get the projection matrix (camera to clip space)
     */
    [[nodiscard]] glm::mat4 projection_matrix();

    /**
     * @brief Get combined view-projection matrix (Camera interface)
     */
    [[nodiscard]] glm::mat4 view_projection_matrix() override;

    /**
     * @brief Get the camera's world position
     */
    [[nodiscard]] glm::vec3 position();

    /**
     * @brief Handle mouse drag for orbit rotation
     *
     * @param xoffset Mouse X delta
     * @param yoffset Mouse Y delta
     */
    void handle_mouse_movement(double xoffset, double yoffset);

    /**
     * @brief Handle mouse scroll event (zoom in/out)
     *
     * @param yoffset Scroll amount (positive = zoom in, negative = zoom out)
     */
    void handle_mouse_scroll(double yoffset);

    /**
     * @brief Handle window/viewport resize (Camera interface)
     *
     * @param width New viewport width
     * @param height New viewport height
     */
    void handle_resize(uint32_t width, uint32_t height) override;

    /**
     * @brief Move target forward (in camera's forward direction projected to XZ plane)
     *
     * @param delta_time Time delta in seconds
     * @param direction 1.0 = forward, -1.0 = backward
     */
    void move_target_forward(float delta_time, float direction = 1.0f);

    /**
     * @brief Move target right (in camera's right direction)
     *
     * @param delta_time Time delta in seconds
     * @param direction 1.0 = right, -1.0 = left
     */
    void move_target_right(float delta_time, float direction = 1.0f);

    /**
     * @brief Move target up (in world Y direction)
     *
     * @param delta_time Time delta in seconds
     * @param direction 1.0 = up, -1.0 = down
     */
    void move_target_up(float delta_time, float direction = 1.0f);

    /**
     * @brief Set the target point (focus point the camera orbits around)
     *
     * @param target Target position in world space
     */
    void set_target(const glm::vec3& target);

    /**
     * @brief Set camera distance from target
     *
     * @param distance Distance in world units
     */
    void set_distance(float distance);

    /**
     * @brief Set camera rotation angles
     *
     * @param azimuth Horizontal angle in degrees
     * @param elevation Vertical angle in degrees (clamped to [-89, 89])
     */
    void set_rotation(float azimuth, float elevation);

    /**
     * @brief Set movement speed for target movement
     *
     * @param speed Movement speed in units per second
     */
    void set_move_speed(float speed);

    /**
     * @brief Reset camera to default parameters
     */
    void reset();

    // Getters for camera parameters
    [[nodiscard]] glm::vec3 get_position() const;
    [[nodiscard]] glm::vec3 target() const;
    [[nodiscard]] float distance() const;
    [[nodiscard]] float azimuth() const;
    [[nodiscard]] float elevation() const;
    [[nodiscard]] float move_speed() const;

private:
    /**
     * @brief Recompute view matrix if dirty
     */
    void update_view_matrix();

    /**
     * @brief Recompute projection matrix if dirty
     */
    void update_projection_matrix();

    // Camera orbital parameters (spherical coordinates)
    glm::vec3 m_target;       ///< Point the camera orbits around
    float m_distance;         ///< Distance from target
    float m_azimuth;          ///< Horizontal rotation (degrees)
    float m_elevation;        ///< Vertical rotation (degrees)

    // Movement parameters
    float m_move_speed;       ///< Movement speed in units per second

    // Projection parameters
    float m_fov;              ///< Field of view (degrees)
    float m_aspect_ratio;     ///< Width / height
    float m_near_plane;       ///< Near clipping plane
    float m_far_plane;        ///< Far clipping plane

    // Cached matrices
    glm::mat4 m_view_matrix;
    glm::mat4 m_projection_matrix;
    glm::mat4 m_view_projection_matrix;

    // Dirty flags for lazy computation
    bool m_view_dirty;
    bool m_projection_dirty;

    // Sensitivity parameters
    float m_mouse_sensitivity;    ///< Degrees per pixel for orbit rotation
    float m_scroll_sensitivity;   ///< Distance change per scroll unit
};

} // namespace ifs
