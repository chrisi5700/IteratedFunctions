// IFS Modular Main - MVC Architecture Demo
// Model (Backend): Sierpinski2D fractal generator
// View (Frontend): ParticleRenderer point cloud visualizer
// Controller: IFSController manages interaction and coordination

#include <ifs/backends/Sierpinski2D.hpp>
#include <ifs/frontends/ParticleRenderer.hpp>
#include <ifs/IFSController.hpp>
#include <ifs/Logger.hpp>
#include <spdlog/spdlog.h>

#include "ifs/backends/CustomIFS.hpp"
#include "ifs/frontends/SphereRenderer.hpp"

int main() {
    Logger::instance().set_level(spdlog::level::trace);
    Logger::instance().info("Starting IFS Modular Visualizer...");

    try {
        // Configure application
        ifs::IFSConfig config{
            .window_width = 1280,
            .window_height = 720,
            .window_title = "IFS Modular - MVC Architecture"
        };

        // Create controller
        auto controller_result = ifs::IFSController::create(config);
        if (!controller_result) {
            Logger::instance().error("Failed to create controller: {}", controller_result.error());
            return 1;
        }
        auto& controller = *controller_result;

        auto backend = ifs::CustomIFS::create(controller->context(), controller->device());
        if (!backend) {
            Logger::instance().error("Failed to create backend: {}", backend.error());
            return 1;
        }

        // Create frontend (View) - Point particle renderer
        auto frontend = ifs::ParticleRenderer::create(
            controller->context(),
            controller->device(),
            controller->render_pass(),
            controller->extent()
        );
        if (!frontend) {
            Logger::instance().error("Failed to create frontend: {}", frontend.error());
            return 1;
        }

        // Set MVC components on controller
        controller->set_backend(std::move(*backend));
        controller->set_frontend(std::move(*frontend));

        // Run main loop (blocks until window closes)
        if (auto result = controller->run(); !result) {
            Logger::instance().error("Runtime error: {}", result.error());
            return 1;
        }

        Logger::instance().info("Application exited successfully");
        return 0;

    } catch (const std::exception& e) {
        Logger::instance().error("Unhandled exception: {}", e.what());
        return 1;
    }
}
