#include <ifs/Camera3D.hpp>
#include <glm/gtc/constants.hpp>
#include <algorithm>

namespace ifs {

Camera3D::Camera3D(uint32_t viewport_width, uint32_t viewport_height)
    : m_target(0.5f, 0.5f, 0.0f)   // Center of Sierpinski triangle
    , m_distance(1.5f)              // Distance from target
    , m_azimuth(-90.0f)             // Facing negative Z
    , m_elevation(-35.0f)           // Looking down
    , m_move_speed(0.5f)
    , m_fov(60.0f)
    , m_aspect_ratio(static_cast<float>(viewport_width) / static_cast<float>(viewport_height))
    , m_near_plane(0.1f)           // Near plane
    , m_far_plane(100.0f)
    , m_view_matrix(1.0f)
    , m_projection_matrix(1.0f)
    , m_view_projection_matrix(1.0f)
    , m_view_dirty(true)
    , m_projection_dirty(true)
    , m_mouse_sensitivity(0.25f)
    , m_scroll_sensitivity(0.1f)
{}

void Camera3D::update_view_matrix() {
    if (!m_view_dirty) return;

    // Convert spherical coordinates to Cartesian position
    float azimuth_rad = glm::radians(m_azimuth);
    float elevation_rad = glm::radians(m_elevation);

    glm::vec3 camera_pos;
    camera_pos.x = m_target.x + m_distance * std::cos(elevation_rad) * std::cos(azimuth_rad);
    camera_pos.y = m_target.y + m_distance * std::sin(elevation_rad);
    camera_pos.z = m_target.z + m_distance * std::cos(elevation_rad) * std::sin(azimuth_rad);

    // Look at target from camera position
    m_view_matrix = glm::lookAt(camera_pos, m_target, glm::vec3(0.0f, 1.0f, 0.0f));
    m_view_dirty = false;
}

void Camera3D::update_projection_matrix() {
    if (!m_projection_dirty) return;

    m_projection_matrix = glm::perspective(
        glm::radians(m_fov),
        m_aspect_ratio,
        m_near_plane,
        m_far_plane
    );

    // Note: GLM_FORCE_DEPTH_ZERO_TO_ONE handles the depth range [0,1] automatically
    // Note: Y-axis flip is now handled by negative viewport height in renderer

    m_projection_dirty = false;
}

glm::mat4 Camera3D::view_matrix() {
    update_view_matrix();
    return m_view_matrix;
}

glm::mat4 Camera3D::projection_matrix() {
    update_projection_matrix();
    return m_projection_matrix;
}

glm::mat4 Camera3D::view_projection_matrix() {
    update_view_matrix();
    update_projection_matrix();
    return m_projection_matrix * m_view_matrix;
}

glm::vec3 Camera3D::position() {
    // Calculate position from orbital parameters
    float azimuth_rad = glm::radians(m_azimuth);
    float elevation_rad = glm::radians(m_elevation);

    glm::vec3 pos;
    pos.x = m_target.x + m_distance * std::cos(elevation_rad) * std::cos(azimuth_rad);
    pos.y = m_target.y + m_distance * std::sin(elevation_rad);
    pos.z = m_target.z + m_distance * std::cos(elevation_rad) * std::sin(azimuth_rad);

    return pos;
}

void Camera3D::handle_mouse_movement(double xoffset, double yoffset) {
    // Update orbital angles
    m_azimuth -= static_cast<float>(xoffset) * m_mouse_sensitivity;
    m_elevation += static_cast<float>(yoffset) * m_mouse_sensitivity;

    // Wrap azimuth to [0, 360)
    while (m_azimuth < 0.0f) m_azimuth += 360.0f;
    while (m_azimuth >= 360.0f) m_azimuth -= 360.0f;

    // Clamp elevation to avoid gimbal lock
    m_elevation = std::clamp(m_elevation, -89.0f, 89.0f);

    m_view_dirty = true;
}

void Camera3D::handle_mouse_scroll(double yoffset) {
    // Zoom in/out (adjust distance from target)
    m_distance -= static_cast<float>(yoffset) * m_scroll_sensitivity;
    m_distance = std::clamp(m_distance, 0.5f, 10.0f);
    m_view_dirty = true;
}

void Camera3D::handle_resize(uint32_t width, uint32_t height) {
    m_aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
    m_projection_dirty = true;
}

void Camera3D::move_target_forward(float delta_time, float direction) {
    // Move target in camera's forward direction (projected to XZ plane)
    float azimuth_rad = glm::radians(m_azimuth);
    glm::vec3 forward(std::cos(azimuth_rad), 0.0f, std::sin(azimuth_rad));
    m_target += forward * m_move_speed * delta_time * direction;
    m_view_dirty = true;
}

void Camera3D::move_target_right(float delta_time, float direction) {
    // Move target in camera's right direction
    float azimuth_rad = glm::radians(m_azimuth);
    glm::vec3 right(-std::sin(azimuth_rad), 0.0f, std::cos(azimuth_rad));
    m_target += right * m_move_speed * delta_time * direction;
    m_view_dirty = true;
}

void Camera3D::move_target_up(float delta_time, float direction) {
    // Move target in world up direction
    m_target.y += m_move_speed * delta_time * direction;
    m_view_dirty = true;
}

void Camera3D::set_target(const glm::vec3& target) {
    m_target = target;
    m_view_dirty = true;
}

void Camera3D::set_distance(float distance) {
    m_distance = std::clamp(distance, 0.5f, 10.0f);
    m_view_dirty = true;
}

void Camera3D::set_rotation(float azimuth, float elevation) {
    m_azimuth = azimuth;
    m_elevation = std::clamp(elevation, -89.0f, 89.0f);

    // Normalize azimuth to [0, 360)
    while (m_azimuth < 0.0f) m_azimuth += 360.0f;
    while (m_azimuth >= 360.0f) m_azimuth -= 360.0f;

    m_view_dirty = true;
}

void Camera3D::set_move_speed(float speed) {
    m_move_speed = std::clamp(speed, 0.1f, 10.0f);
}

void Camera3D::reset() {
    m_target = glm::vec3(0.5f, 0.5f, 0.0f);
    m_distance = 1.5f;
    m_azimuth = -90.0f;
    m_elevation = -35.0f;
    m_move_speed = 0.5f;
    m_view_dirty = true;
}

// Getter implementations
glm::vec3 Camera3D::get_position() const {
    // Calculate position from orbital parameters
    float azimuth_rad = glm::radians(m_azimuth);
    float elevation_rad = glm::radians(m_elevation);

    glm::vec3 pos;
    pos.x = m_target.x + m_distance * std::cos(elevation_rad) * std::cos(azimuth_rad);
    pos.y = m_target.y + m_distance * std::sin(elevation_rad);
    pos.z = m_target.z + m_distance * std::cos(elevation_rad) * std::sin(azimuth_rad);

    return pos;
}

glm::vec3 Camera3D::target() const {
    return m_target;
}

float Camera3D::distance() const {
    return m_distance;
}

float Camera3D::azimuth() const {
    return m_azimuth;
}

float Camera3D::elevation() const {
    return m_elevation;
}

float Camera3D::move_speed() const {
    return m_move_speed;
}

} // namespace ifs
