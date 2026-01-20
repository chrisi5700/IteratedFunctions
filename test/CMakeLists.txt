find_package(Catch2 3 REQUIRED)

include(CTest)
include(Catch)

# VulkanContext tests
add_executable(VulkanContextTests VulkanContext/VulkanContextTests.cpp)
target_link_libraries(VulkanContextTests PRIVATE IFSLib Catch2::Catch2WithMain)


add_executable(ShaderLoadingTests Shader/ShaderLoadingTests.cpp)
target_link_libraries(ShaderLoadingTests PRIVATE IFSLib Catch2::Catch2WithMain)
target_compile_options(ShaderLoadingTests PRIVATE -O0)

add_executable(ShaderValidationTests Shader/ShaderValidationTests.cpp)
target_link_libraries(ShaderValidationTests PRIVATE IFSLib Catch2::Catch2WithMain)

catch_discover_tests(VulkanContextTests)
catch_discover_tests(ShaderLoadingTests)
catch_discover_tests(ShaderValidationTests)

