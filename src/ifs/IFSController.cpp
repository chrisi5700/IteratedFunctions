#include <ifs/IFSController.hpp>
#include <ifs/Logger.hpp>
#include <ifs/frontends/ParticleRenderer.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <chrono>
#include <random>
#include <format>

namespace ifs {

// GLFW callback wrappers
void glfw_key_callback(GLFWwindow* window, int key, [[maybe_unused]] int scancode, int action,[[maybe_unused]]  int mods) {
    auto* controller = static_cast<IFSController*>(glfwGetWindowUserPointer(window));
    if (!controller) return;

    if (action == GLFW_PRESS) {
        controller->m_keys_pressed[key] = true;

        // Toggle mouse capture with Tab key
        if (key == GLFW_KEY_TAB) {
            controller->m_mouse_captured = !controller->m_mouse_captured;
            if (controller->m_mouse_captured) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                controller->m_first_mouse = true;  // Reset to avoid jump
            } else {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }
        }
    } else if (action == GLFW_RELEASE) {
        controller->m_keys_pressed[key] = false;
    }
}

void glfw_mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    auto* controller = static_cast<IFSController*>(glfwGetWindowUserPointer(window));
    if (!controller || !controller->m_camera) return;

    if (!controller->m_mouse_captured) return;

    if (controller->m_first_mouse) {
        controller->m_last_mouse_x = xpos;
        controller->m_last_mouse_y = ypos;
        controller->m_first_mouse = false;
        return;
    }

    double xoffset = xpos - controller->m_last_mouse_x;
    double yoffset = ypos - controller->m_last_mouse_y;
    controller->m_last_mouse_x = xpos;
    controller->m_last_mouse_y = ypos;

    controller->m_camera->handle_mouse_movement(xoffset, yoffset);
}

void glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    (void)xoffset;  // Unused
    auto* controller = static_cast<IFSController*>(glfwGetWindowUserPointer(window));
    if (!controller || !controller->m_camera) return;

    controller->m_camera->handle_mouse_scroll(yoffset);
}

IFSController::IFSController(const IFSConfig& config)
    : m_config(config)
    , m_context(nullptr)
    , m_window(nullptr)
    , m_backend(nullptr)
    , m_frontend(nullptr)
    , m_camera(nullptr)
    , m_imgui_descriptor_pool(nullptr)
{
    m_ifs_params.iteration_count = 100;
    m_ifs_params.scale = 1.0f;
    std::random_device rd;
    m_ifs_params.random_seed = rd();
}

IFSController::~IFSController() {
    cleanup();
}

std::expected<std::unique_ptr<IFSController>, std::string> IFSController::create(const IFSConfig& config) {
    auto controller = std::unique_ptr<IFSController>(new IFSController(config));

    if (auto result = controller->initialize(); !result) {
        return std::unexpected(result.error());
    }

    return controller;
}

std::expected<void, std::string> IFSController::initialize() {
    Logger::instance().info("Initializing IFS Controller...");

    // Create Vulkan context
    try {
        m_context = std::make_unique<VulkanContext>("IFS Controller");
    } catch (const std::exception& e) {
        return std::unexpected(std::format("Failed to create Vulkan context: {}", e.what()));
    }

    // Create window
    auto window_result = Window::create(
        *m_context,
        m_config.window_width,
        m_config.window_height,
        m_config.window_title
    );
    if (!window_result) {
        return std::unexpected(std::format("Failed to create window: {}", window_result.error()));
    }
    m_window = std::make_unique<Window>(std::move(window_result.value()));

    // Note: Particle buffer is now owned by the backend
    // It will be created when the backend is initialized

    // Create 3D camera
    m_camera = std::make_unique<Camera3D>(m_config.window_width, m_config.window_height);

    // Set up input handling
    m_mouse_captured = true;  // Start with mouse captured
    glfwSetWindowUserPointer(m_window->get_window_handle(), this);
    glfwSetKeyCallback(m_window->get_window_handle(), glfw_key_callback);
    glfwSetCursorPosCallback(m_window->get_window_handle(), glfw_mouse_callback);
    glfwSetScrollCallback(m_window->get_window_handle(), glfw_scroll_callback);

    // Enable mouse capture at startup
    glfwSetInputMode(m_window->get_window_handle(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Setup ImGui
    if (auto result = setup_imgui(); !result) {
        return std::unexpected(result.error());
    }

    Logger::instance().info("IFS Controller initialized successfully");
    return {};
}

std::expected<void, std::string> IFSController::setup_imgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO();
    ImGui::StyleColorsDark();

    // ImGui descriptor pool
    std::vector<vk::DescriptorPoolSize> pool_sizes = {
        {vk::DescriptorType::eSampler, 1000},
        {vk::DescriptorType::eCombinedImageSampler, 1000},
        {vk::DescriptorType::eSampledImage, 1000},
        {vk::DescriptorType::eStorageImage, 1000},
        {vk::DescriptorType::eUniformTexelBuffer, 1000},
        {vk::DescriptorType::eStorageTexelBuffer, 1000},
        {vk::DescriptorType::eUniformBuffer, 1000},
        {vk::DescriptorType::eStorageBuffer, 1000},
        {vk::DescriptorType::eUniformBufferDynamic, 1000},
        {vk::DescriptorType::eStorageBufferDynamic, 1000},
        {vk::DescriptorType::eInputAttachment, 1000}
    };

    auto imgui_pool_info = vk::DescriptorPoolCreateInfo()
        .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
        .setMaxSets(1000)
        .setPoolSizes(pool_sizes);

    try {
        m_imgui_descriptor_pool = m_context->device().createDescriptorPool(imgui_pool_info);
    } catch (const vk::SystemError& e) {
        return std::unexpected(std::format("Failed to create ImGui descriptor pool: {}", e.what()));
    }

    ImGui_ImplGlfw_InitForVulkan(m_window->get_window_handle(), true);

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance = static_cast<VkInstance>(m_context->instance());
    init_info.PhysicalDevice = static_cast<VkPhysicalDevice>(m_context->physical_device());
    init_info.Device = static_cast<VkDevice>(m_context->device());
    init_info.QueueFamily = m_context->queue_indices().graphics;
    init_info.Queue = static_cast<VkQueue>(m_context->graphics_queue());
    init_info.DescriptorPool = static_cast<VkDescriptorPool>(m_imgui_descriptor_pool);
    init_info.MinImageCount = 2;
    init_info.ImageCount = m_window->image_count();
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.RenderPass = static_cast<VkRenderPass>(m_window->render_pass());
    init_info.Allocator = nullptr;
    init_info.CheckVkResultFn = nullptr;

    ImGui_ImplVulkan_Init(&init_info);
    ImGui_ImplVulkan_CreateFontsTexture();

    return {};
}

void IFSController::set_backend(std::unique_ptr<IFSBackend> backend) {
    m_backend = std::move(backend);
    m_needs_recompute = true;
}

void IFSController::set_frontend(std::unique_ptr<IFSFrontend> frontend) {
    m_frontend = std::move(frontend);

    // Initialize frontend graphics infrastructure for current swapchain size
    if (m_frontend && m_window) {
        m_frontend->handle_swapchain_recreation(m_window->image_count());
    }
}

void IFSController::handle_input(float delta_time) {
    if (!m_camera) return;

    // WASD moves the focus/target point
    if (m_keys_pressed[GLFW_KEY_W]) {
        m_camera->move_target_forward(delta_time, -1.0f);
    }
    if (m_keys_pressed[GLFW_KEY_S]) {
        m_camera->move_target_forward(delta_time, 1.0f);
    }
    if (m_keys_pressed[GLFW_KEY_A]) {
        m_camera->move_target_right(delta_time, 1.0f);
    }
    if (m_keys_pressed[GLFW_KEY_D]) {
        m_camera->move_target_right(delta_time, -1.0f);
    }

    // QE for up/down movement of target
    if (m_keys_pressed[GLFW_KEY_Q]) {
        m_camera->move_target_up(delta_time, 1.0f);
    }
    if (m_keys_pressed[GLFW_KEY_E]) {
        m_camera->move_target_up(delta_time, -1.0f);
    }
}

void IFSController::render_ui_callbacks(const std::vector<UICallback>& callbacks) {
    for (const auto& callback : callbacks) {
        switch (callback.get_callback_type()) {
            case CallbackType::Continuous: {
                if (auto* cb = callback.as_continuous()) {
                    float value = cb->getter();
                    int flags = cb->logarithmic ? ImGuiSliderFlags_Logarithmic : 0;
                    if (ImGui::SliderFloat(callback.field_name.c_str(), &value, cb->min, cb->max, "%.3f", flags)) {
                        cb->setter(value);
                    }
                }
                break;
            }
            case CallbackType::Discrete: {
                if (auto* cb = callback.as_discrete()) {
                    int value = cb->getter();
                    if (ImGui::SliderInt(callback.field_name.c_str(), &value, cb->min, cb->max)) {
                        cb->setter(value);
                    }
                }
                break;
            }
            case CallbackType::Toggle: {
                if (auto* cb = callback.as_toggle()) {
                    bool value = cb->getter();
                    if (ImGui::Checkbox(callback.field_name.c_str(), &value)) {
                        cb->setter(value);
                    }
                }
                break;
            }
        }
    }
}

void IFSController::render_ui() {
    ImGui::Begin("IFS Controls");

    if (m_backend) {
        ImGui::Text("Backend: %s", m_backend->name().data());
    } else {
        ImGui::TextDisabled("Backend: (none)");
    }

    if (m_frontend) {
        ImGui::Text("Frontend: %s", m_frontend->name().data());
    } else {
        ImGui::TextDisabled("Frontend: (none)");
    }

    if (m_backend) {
        ImGui::Text("Particles: %u", m_backend->get_particle_count());
    } else {
        ImGui::TextDisabled("Particles: (backend not set)");
    }

    ImGui::Separator();
    ImGui::Text("Camera Controls:");
    ImGui::Text("  TAB: Toggle mouse capture");
    ImGui::Text("  WASD: Move focus point");
    ImGui::Text("  QE: Move focus up/down");
    ImGui::Text("  Mouse: Orbit around focus (when captured)");
    ImGui::Text("  Scroll: Zoom in/out");

    if (m_camera) {
        ImGui::Separator();
        auto cam_target = m_camera->target();
        auto cam_pos = m_camera->get_position();
        ImGui::Text("Focus: (%.2f, %.2f, %.2f)", cam_target.x, cam_target.y, cam_target.z);
        ImGui::Text("Position: (%.2f, %.2f, %.2f)", cam_pos.x, cam_pos.y, cam_pos.z);
        ImGui::Text("Distance: %.2f", m_camera->distance());
        ImGui::Text("Azimuth: %.1f  Elevation: %.1f", m_camera->azimuth(), m_camera->elevation());
        ImGui::Text("Move Speed: %.2f", m_camera->move_speed());
        ImGui::Text("Mouse: %s", m_mouse_captured ? "Captured" : "Free");
    }

    ImGui::Separator();
    if (ImGui::Button("Reset IFS")) {
        std::random_device rd;
        m_ifs_params.random_seed = rd();
        m_needs_recompute = true;
    }

    if (ImGui::SliderFloat("Scale", &m_ifs_params.scale, 0.1f, 10.0f)) {
        m_needs_recompute = true;
    }

    ImGui::Separator();

    // Backend-specific UI controls
    if (m_backend) {
        ImGui::Text("Backend Parameters:");
        auto backend_callbacks = m_backend->get_ui_callbacks();
        if (!backend_callbacks.empty()) {
            render_ui_callbacks(backend_callbacks);
            m_needs_recompute = true;  // Backend parameters might affect computation
            m_needs_buffer_rebind = true;  // Buffer might have changed (e.g., particle count)
        } else {
            ImGui::TextDisabled("(No backend parameters)");
        }
    }

    ImGui::Separator();

    // Frontend-specific UI controls
    if (m_frontend) {
        ImGui::Text("Frontend Parameters:");
        auto frontend_callbacks = m_frontend->get_ui_callbacks();
        if (!frontend_callbacks.empty()) {
            render_ui_callbacks(frontend_callbacks);
        } else {
            ImGui::TextDisabled("(No frontend parameters)");
        }
    }

    ImGui::Separator();

    if (ImGui::Button("Reset Camera")) {
        if (m_camera) {
            m_camera->reset();
        }
    }

    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::End();
}

std::expected<void, std::string> IFSController::run() {
    if (!m_backend) {
        return std::unexpected("Backend not set - call set_backend() before run()");
    }
    if (!m_frontend) {
        return std::unexpected("Frontend not set - call set_frontend() before run()");
    }

    Logger::instance().info("Starting main loop...");

    // Check if we need queue family transfers
    bool different_queue_families = m_context->queue_indices().has_dedicated_compute();

    // Dispatch initial compute (backend owns particle buffer)
    m_backend->compute(nullptr, 0, m_ifs_params);  // Parameters ignored by backend
    m_backend->wait_compute_complete();
    m_needs_recompute = false;
    m_needs_ownership_acquire = different_queue_families;

    // IMPORTANT: Bind particle buffer to frontend descriptor set
    // ParticleRenderer needs this to access particle data in shaders
    // Query backend for particle buffer
    if (auto* particle_renderer = dynamic_cast<ParticleRenderer*>(m_frontend.get())) {
        particle_renderer->update_particle_buffer(m_backend->get_particle_buffer());
    }

    // Create image available semaphores (one per swapchain image)
    std::vector<vk::Semaphore> image_available_sems;
    for (uint32_t i = 0; i < m_window->image_count(); i++) {
        image_available_sems.push_back(m_context->device().createSemaphore({}));
    }

    // Delta time tracking
    auto last_frame_time = std::chrono::high_resolution_clock::now();

    // Semaphore cycling - use a simple counter to cycle through all available semaphores
    uint32_t semaphore_index = 0;

    // Main loop
    while (!m_window->should_close()) {
        // Calculate delta time
        auto current_frame_time = std::chrono::high_resolution_clock::now();
        float delta_time = std::chrono::duration<float>(current_frame_time - last_frame_time).count();
        last_frame_time = current_frame_time;

        glfwPollEvents();

        // Process camera input
        handle_input(delta_time);

        // Start ImGui frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Build UI
        render_ui();

        ImGui::Render();

        // Rebind frontend buffer if backend reallocated it
        if (m_needs_buffer_rebind) {
            // Wait for all GPU work to complete before updating descriptor sets
            // (descriptor sets cannot be updated while in use by pending command buffers)
            m_context->device().waitIdle();

            // Update frontend's descriptor set with (potentially new) particle buffer
            if (auto* particle_renderer = dynamic_cast<ParticleRenderer*>(m_frontend.get())) {
                particle_renderer->update_particle_buffer(m_backend->get_particle_buffer());
            }
            m_needs_buffer_rebind = false;
        }

        // Recompute if needed
        if (m_needs_recompute) {
            m_backend->compute(nullptr, 0, m_ifs_params);  // Parameters ignored by backend
            m_backend->wait_compute_complete();
            m_needs_recompute = false;
            m_needs_ownership_acquire = different_queue_families;
        }

        // Acquire next image
        // Cycle through available semaphores to avoid reuse conflicts
        auto acquire_result = m_window->acquire_next_image(image_available_sems[semaphore_index]);

        if (!acquire_result) {
            // Swapchain out of date - will be recreated
            m_frontend->handle_swapchain_recreation(m_window->image_count());

            // Rebind particle buffer after swapchain recreation (query from backend)
            if (auto* particle_renderer = dynamic_cast<ParticleRenderer*>(m_frontend.get())) {
                particle_renderer->update_particle_buffer(m_backend->get_particle_buffer());
            }
            continue;
        }

        uint32_t image_index = *acquire_result;

        // Prepare frame render info (query backend for buffer/count)
        FrameRenderInfo render_info{
            .image_index = image_index,
            .current_frame = m_current_frame,
            .image_available_semaphore = image_available_sems[semaphore_index],
            .framebuffer = m_window->get_framebuffer(image_index),
            .extent = m_window->extent(),
            .render_pass = m_window->render_pass(),
            .clear_values = {
                vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}),
                vk::ClearDepthStencilValue(1.0f, 0)
            },
            .particle_buffer = m_backend->get_particle_buffer(),
            .particle_count = m_backend->get_particle_count(),
            .camera = *m_camera,
            .needs_ownership_acquire = m_needs_ownership_acquire,
            .compute_queue_family = m_context->queue_indices().compute,
            .graphics_queue_family = m_context->queue_indices().graphics,
            .imgui_draw_data = ImGui::GetDrawData()
        };

        // Render frame (frontend handles everything)
        auto render_finished_sem = m_frontend->render_frame(render_info, m_context->graphics_queue());

        // Present (correct argument order: queue, semaphore, image_index)
        auto present_result = m_window->present(m_context->graphics_queue(), render_finished_sem, image_index);

        if (!present_result) {
            // Swapchain out of date - recreate
            m_frontend->handle_swapchain_recreation(m_window->image_count());
        }

        m_needs_ownership_acquire = false;
        m_current_frame = (m_current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        // Advance semaphore index for next frame
        semaphore_index = (semaphore_index + 1) % image_available_sems.size();
    }

    // Wait for all operations to complete before cleanup
    m_context->device().waitIdle();

    // Cleanup semaphores
    for (auto& sem : image_available_sems) {
        m_context->device().destroySemaphore(sem);
    }

    Logger::instance().info("Shutdown complete");
    return {};
}

void IFSController::cleanup() {
    if (m_context && m_context->device()) {
        m_context->device().waitIdle();
    }

    // Cleanup ImGui
    if (m_imgui_descriptor_pool) {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        if (m_context && m_context->device()) {
            m_context->device().destroyDescriptorPool(m_imgui_descriptor_pool);
        }
        m_imgui_descriptor_pool = nullptr;
    }

    // Resources are automatically cleaned up by unique_ptr destructors
}

} // namespace ifs
