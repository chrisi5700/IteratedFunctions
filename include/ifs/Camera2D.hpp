#pragma once

#include "Camera.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cstdint>

namespace ifs {

/**
 * @brief Simple 2D orthographic camera
 *
 * Maps a 2D world space rectangle to screen space using orthographic projection.
 * Perfect for 2D fractals like Sierpinski triangle.
 */
class Camera2D : public Camera {
public:
    /**
     * @brief Construct 2D camera with default [0,1] x [0,1] view
     *
     * @param viewport_width Viewport width in pixels
     * @param viewport_height Viewport height in pixels
     */
    Camera2D(uint32_t viewport_width, uint32_t viewport_height)
        : m_viewport_width(viewport_width)
        , m_viewport_height(viewport_height)
        , m_view_min(0.0f, 0.0f)
        , m_view_max(1.0f, 1.0f)
        , m_matrix_dirty(true)
        , m_move_speed(0.5f)
        , m_zoom_sensitivity(0.1f)
        , m_last_mouse_pos(0.0, 0.0)
        , m_mouse_drag_active(false)
    {}

    /**
     * @brief Get the combined view-projection matrix
     *
     * For 2D orthographic projection, this maps the view rectangle
     * [view_min, view_max] to clip space [-1, 1]^2
     */
    [[nodiscard]] glm::mat4 view_projection_matrix() override {
        if (m_matrix_dirty) {
            update_matrix();
        }
        return m_view_projection;
    }

    /**
     * @brief Set the visible world space rectangle
     *
     * @param min Bottom-left corner of view (world space)
     * @param max Top-right corner of view (world space)
     */
    void set_view_rect(const glm::vec2& min, const glm::vec2& max) {
        m_view_min = min;
        m_view_max = max;
        m_matrix_dirty = true;
    }

    /**
     * @brief Handle viewport resize
     */
    void handle_resize(uint32_t width, uint32_t height) override {
        m_viewport_width = width;
        m_viewport_height = height;
        // Orthographic projection doesn't depend on aspect ratio if we want to maintain aspect
        // For now, we don't mark dirty - the view rect stays the same
    }

    /**
     * @brief Reset to default [0,1] x [0,1] view
     */
    void reset() {
        m_view_min = glm::vec2(0.0f, 0.0f);
        m_view_max = glm::vec2(1.0f, 1.0f);
        m_matrix_dirty = true;
    }

    /**
     * @brief Pan the camera (move the view rectangle)
     *
     * @param delta Movement in world space (positive = move view right/up)
     */
    void pan(const glm::vec2& delta) {
        m_view_min += delta;
        m_view_max += delta;
        m_matrix_dirty = true;
    }

    /**
     * @brief Zoom the camera around a world space point
     *
     * @param factor Zoom factor (>1 = zoom in, <1 = zoom out)
     * @param center_world World space point to zoom around
     */
    void zoom(float factor, const glm::vec2& center_world) {
        // Zoom by scaling the view rectangle around the center point
        glm::vec2 view_size = m_view_max - m_view_min;
        glm::vec2 new_size = view_size / factor;

        // Calculate offset from center to maintain it in place
        glm::vec2 center_offset = center_world - m_view_min;
        glm::vec2 center_ratio = center_offset / view_size;

        m_view_min = center_world - center_ratio * new_size;
        m_view_max = m_view_min + new_size;
        m_matrix_dirty = true;
    }

    /**
     * @brief Zoom around the center of the current view
     *
     * @param factor Zoom factor (>1 = zoom in, <1 = zoom out)
     */
    void zoom_at_center(float factor) {
        glm::vec2 center = (m_view_min + m_view_max) * 0.5f;
        zoom(factor, center);
    }

    /**
     * @brief Get current view bounds
     */
    [[nodiscard]] glm::vec2 view_min() const { return m_view_min; }
    [[nodiscard]] glm::vec2 view_max() const { return m_view_max; }
    [[nodiscard]] glm::vec2 view_size() const { return m_view_max - m_view_min; }
    [[nodiscard]] glm::vec2 view_center() const { return (m_view_min + m_view_max) * 0.5f; }

private:
    void update_matrix() {
        // Create orthographic projection matrix
        // Maps [view_min.x, view_max.x] x [view_min.y, view_max.y] x [-1, 1] to clip space
        m_view_projection = glm::ortho(
            m_view_min.x, m_view_max.x,   // left, right
            m_view_min.y, m_view_max.y,   // bottom, top
            -1.0f, 1.0f                    // near, far
        );

        // Vulkan clip space has Y pointing down, flip it
        m_view_projection[1][1] *= -1.0f;

        m_matrix_dirty = false;
    }

    uint32_t m_viewport_width;
    uint32_t m_viewport_height;
    glm::vec2 m_view_min;
    glm::vec2 m_view_max;
    glm::mat4 m_view_projection;
    bool m_matrix_dirty;
};

} // namespace ifs
