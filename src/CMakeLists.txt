# Add library definitions here
# See README.md for CMake library patterns and examples

# IFS (Iterated Function System) visualization library
target_add_library(IFSLib STATIC
    ifs/VulkanContext.cpp
    ifs/Window.cpp
    ifs/Shader.cpp
    ifs/ParticleBuffer.cpp
    ifs/Camera3D.cpp
    ifs/IFSController.cpp
    ifs/backends/Sierpinski2D.cpp
    ifs/frontends/ParticleRenderer.cpp
    ifs/frontends/SphereRenderer.cpp
)
target_link_libraries(IFSLib PUBLIC Vulkan::Vulkan glfw spdlog::spdlog slang::slang glm::glm)
target_include_directories(IFSLib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Combined definitions
target_compile_definitions(IFSLib PUBLIC
    LOG_FILE="${CMAKE_SOURCE_DIR}/logs/logs.txt"
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL=1
    SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders"
        GLM_FORCE_DEPTH_ZERO_TO_ONE
)
