//
// IFS (Iterated Function System) Visualizer
// Simple implementation using direct Vulkan HPP calls
//

#include <ifs/VulkanContext.hpp>
#include <ifs/Window.hpp>
#include <ifs/Shader.hpp>
#include <ifs/Logger.hpp>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>

// Particle structure (16-byte aligned)
struct Particle {
    glm::vec2 position;
    glm::vec2 padding;
};

// IFS parameters for compute shader
struct IFSParams {
    uint32_t iteration_count;
    uint32_t particle_count;
    float scale;
    uint32_t random_seed;
};

constexpr uint32_t PARTICLE_COUNT = 1000000;
constexpr uint32_t WORK_GROUP_SIZE = 256;

struct SwapchainData {
    vk::SwapchainKHR swapchain;
    std::vector<vk::Image> images;
    std::vector<vk::ImageView> image_views;
    std::vector<vk::Framebuffer> framebuffers;
    vk::Extent2D extent;
    vk::SurfaceFormatKHR format;
};

bool framebuffer_resized = false;

void framebuffer_resize_callback(GLFWwindow*, int, int) {
    framebuffer_resized = true;
}

SwapchainData create_swapchain(vk::Device device, vk::PhysicalDevice physical_device,
                               vk::SurfaceKHR surface, vk::RenderPass render_pass,
                               GLFWwindow* window, vk::SwapchainKHR old_swapchain = nullptr) {
    SwapchainData data;

    auto capabilities = physical_device.getSurfaceCapabilitiesKHR(surface);
    auto formats = physical_device.getSurfaceFormatsKHR(surface);
    auto present_modes = physical_device.getSurfacePresentModesKHR(surface);

    // Choose surface format
    data.format = formats[0];
    for (const auto& format : formats) {
        if (format.format == vk::Format::eB8G8R8A8Srgb &&
            format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            data.format = format;
            break;
        }
    }

    // Choose extent
    if (capabilities.currentExtent.width != UINT32_MAX) {
        data.extent = capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        data.extent = vk::Extent2D{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
        data.extent.width = std::clamp(data.extent.width, capabilities.minImageExtent.width,
                                      capabilities.maxImageExtent.width);
        data.extent.height = std::clamp(data.extent.height, capabilities.minImageExtent.height,
                                       capabilities.maxImageExtent.height);
    }

    // Handle minimization (zero-sized framebuffer)
    while (data.extent.width == 0 || data.extent.height == 0) {
        glfwGetFramebufferSize(window, reinterpret_cast<int*>(&data.extent.width),
                               reinterpret_cast<int*>(&data.extent.height));
        glfwWaitEvents();
    }

    // Choose present mode
    vk::PresentModeKHR present_mode = vk::PresentModeKHR::eFifo;
    for (const auto& mode : present_modes) {
        if (mode == vk::PresentModeKHR::eMailbox) {
            present_mode = mode;
            break;
        }
        if (mode == vk::PresentModeKHR::eImmediate) {
            present_mode = mode;
        }
    }

    uint32_t image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount) {
        image_count = capabilities.maxImageCount;
    }

    auto swapchain_info = vk::SwapchainCreateInfoKHR()
        .setSurface(surface)
        .setMinImageCount(image_count)
        .setImageFormat(data.format.format)
        .setImageColorSpace(data.format.colorSpace)
        .setImageExtent(data.extent)
        .setImageArrayLayers(1)
        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
        .setImageSharingMode(vk::SharingMode::eExclusive)
        .setPreTransform(capabilities.currentTransform)
        .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
        .setPresentMode(present_mode)
        .setClipped(true)
        .setOldSwapchain(old_swapchain);

    data.swapchain = device.createSwapchainKHR(swapchain_info);
    data.images = device.getSwapchainImagesKHR(data.swapchain);

    // Create image views
    for (const auto& image : data.images) {
        auto view_info = vk::ImageViewCreateInfo()
            .setImage(image)
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(data.format.format)
            .setComponents(vk::ComponentMapping())
            .setSubresourceRange(vk::ImageSubresourceRange()
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setBaseMipLevel(0)
                .setLevelCount(1)
                .setBaseArrayLayer(0)
                .setLayerCount(1));
        data.image_views.push_back(device.createImageView(view_info));
    }

    // Create framebuffers
    for (const auto& view : data.image_views) {
        auto framebuffer_info = vk::FramebufferCreateInfo()
            .setRenderPass(render_pass)
            .setAttachments(view)
            .setWidth(data.extent.width)
            .setHeight(data.extent.height)
            .setLayers(1);
        data.framebuffers.push_back(device.createFramebuffer(framebuffer_info));
    }

    return data;
}

void cleanup_swapchain(vk::Device device, const SwapchainData& data) {
    for (auto fb : data.framebuffers) device.destroyFramebuffer(fb);
    for (auto view : data.image_views) device.destroyImageView(view);
    device.destroySwapchainKHR(data.swapchain);
}

int main() {
    try {
        // Initialize GLFW and Vulkan context
        VulkanContext context("IFS Visualizer");
        Window window(1280, 720, "Sierpinski Triangle - IFS");

        auto device = context.device();
        auto physical_device = context.physical_device();

        // Set resize callback
        glfwSetFramebufferSizeCallback(window.get_window_handle(), framebuffer_resize_callback);

        Logger::instance().info("Creating swapchain...");

        // Create surface
        VkSurfaceKHR surface_c;
        if (glfwCreateWindowSurface(static_cast<VkInstance>(context.instance()),
                                     window.get_window_handle(),
                                     nullptr,
                                     &surface_c) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface");
        }
        vk::SurfaceKHR surface(surface_c);

        // Check surface support
        if (!physical_device.getSurfaceSupportKHR(context.queue_indices().graphics, surface)) {
            throw std::runtime_error("Surface not supported");
        }

        // Create render pass (needs to be created before swapchain for helper function)
        // We'll use a placeholder format that will be overridden
        auto temp_format = vk::Format::eB8G8R8A8Srgb;

        auto color_attachment = vk::AttachmentDescription()
            .setFormat(temp_format)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

        auto color_attachment_ref = vk::AttachmentReference()
            .setAttachment(0)
            .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

        auto subpass = vk::SubpassDescription()
            .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
            .setColorAttachments(color_attachment_ref);

        auto dependency = vk::SubpassDependency()
            .setSrcSubpass(VK_SUBPASS_EXTERNAL)
            .setDstSubpass(0)
            .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
            .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
            .setSrcAccessMask(vk::AccessFlags{})
            .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);

        auto render_pass_info = vk::RenderPassCreateInfo()
            .setAttachments(color_attachment)
            .setSubpasses(subpass)
            .setDependencies(dependency);

        auto render_pass = device.createRenderPass(render_pass_info);
        Logger::instance().info("Created render pass");

        // Create swapchain
        auto swapchain_data = create_swapchain(device, physical_device, surface,
                                               render_pass, window.get_window_handle());
        Logger::instance().info("Created swapchain with {} images", swapchain_data.images.size());

        // Create particle buffer
        std::vector<Particle> particles(PARTICLE_COUNT);
        {
            std::random_device rd;
            std::default_random_engine rng(rd());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (auto& p : particles) {
                p.position = glm::vec2(dist(rng), dist(rng));
                p.padding = glm::vec2(0.0f);
            }
        }

        vk::DeviceSize buffer_size = sizeof(Particle) * PARTICLE_COUNT;

        // Create staging buffer
        auto staging_buffer_info = vk::BufferCreateInfo()
            .setSize(buffer_size)
            .setUsage(vk::BufferUsageFlagBits::eTransferSrc)
            .setSharingMode(vk::SharingMode::eExclusive);

        auto staging_buffer = device.createBuffer(staging_buffer_info);

        auto staging_mem_req = device.getBufferMemoryRequirements(staging_buffer);
        auto mem_props = physical_device.getMemoryProperties();

        uint32_t staging_mem_type = 0;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
            if ((staging_mem_req.memoryTypeBits & (1 << i)) &&
                (mem_props.memoryTypes[i].propertyFlags &
                 (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent))) {
                staging_mem_type = i;
                break;
            }
        }

        auto staging_mem_alloc = vk::MemoryAllocateInfo()
            .setAllocationSize(staging_mem_req.size)
            .setMemoryTypeIndex(staging_mem_type);

        auto staging_memory = device.allocateMemory(staging_mem_alloc);
        device.bindBufferMemory(staging_buffer, staging_memory, 0);

        // Copy data to staging buffer
        void* data = device.mapMemory(staging_memory, 0, buffer_size);
        std::memcpy(data, particles.data(), buffer_size);
        device.unmapMemory(staging_memory);

        // Create device-local particle buffer
        auto particle_buffer_info = vk::BufferCreateInfo()
            .setSize(buffer_size)
            .setUsage(vk::BufferUsageFlagBits::eStorageBuffer |
                      vk::BufferUsageFlagBits::eTransferDst)
            .setSharingMode(vk::SharingMode::eExclusive);

        auto particle_buffer = device.createBuffer(particle_buffer_info);
        auto particle_mem_req = device.getBufferMemoryRequirements(particle_buffer);

        uint32_t particle_mem_type = 0;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
            if ((particle_mem_req.memoryTypeBits & (1 << i)) &&
                (mem_props.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
                particle_mem_type = i;
                break;
            }
        }

        auto particle_mem_alloc = vk::MemoryAllocateInfo()
            .setAllocationSize(particle_mem_req.size)
            .setMemoryTypeIndex(particle_mem_type);

        auto particle_memory = device.allocateMemory(particle_mem_alloc);
        device.bindBufferMemory(particle_buffer, particle_memory, 0);

        // Copy staging to particle buffer
        auto cmd_pool_info = vk::CommandPoolCreateInfo()
            .setQueueFamilyIndex(context.queue_indices().graphics)
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        auto command_pool = device.createCommandPool(cmd_pool_info);

        auto cmd_alloc_info = vk::CommandBufferAllocateInfo()
            .setCommandPool(command_pool)
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(1);
        auto copy_cmd_buffers = device.allocateCommandBuffers(cmd_alloc_info);
        auto copy_cmd = copy_cmd_buffers[0];

        copy_cmd.begin(vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
        auto copy_region = vk::BufferCopy().setSize(buffer_size);
        copy_cmd.copyBuffer(staging_buffer, particle_buffer, copy_region);
        copy_cmd.end();

        auto submit_info = vk::SubmitInfo().setCommandBuffers(copy_cmd);
        context.graphics_queue().submit(submit_info);
        context.graphics_queue().waitIdle();

        device.freeCommandBuffers(command_pool, copy_cmd_buffers);
        device.destroyBuffer(staging_buffer);
        device.freeMemory(staging_memory);

        Logger::instance().info("Created particle buffer with {} particles", PARTICLE_COUNT);

        // Load compute shader and create pipeline
        auto compute_shader_result = Shader::create_shader(device, "ifs/ifs_compute");
        if (!compute_shader_result.has_value()) {
            std::println(std::cerr, "Failed to load compute shader: {}", compute_shader_result.error());
            return 1;
        }
        auto compute_shader = std::move(*compute_shader_result);

        Logger::instance().info("Loaded compute shader");

        // Create compute descriptor set layout from shader reflection
        auto& compute_descriptors = compute_shader.get_descriptor_infos();

        std::vector<vk::DescriptorSetLayoutBinding> compute_bindings;
        for (const auto& desc : compute_descriptors) {
            compute_bindings.push_back(vk::DescriptorSetLayoutBinding()
                .setBinding(desc.binding)
                .setDescriptorType(desc.type)
                .setDescriptorCount(desc.descriptor_count)
                .setStageFlags(desc.stage));
        }

        auto compute_layout_info = vk::DescriptorSetLayoutCreateInfo()
            .setBindings(compute_bindings);
        auto compute_descriptor_layout = device.createDescriptorSetLayout(compute_layout_info);

        // Create compute pipeline layout
        auto compute_pipeline_layout_info = vk::PipelineLayoutCreateInfo()
            .setSetLayouts(compute_descriptor_layout);
        auto compute_pipeline_layout = device.createPipelineLayout(compute_pipeline_layout_info);

        // Create compute pipeline
        auto compute_stage = compute_shader.create_pipeline_shader_stage_create_info();
        auto compute_pipeline_info = vk::ComputePipelineCreateInfo()
            .setStage(compute_stage)
            .setLayout(compute_pipeline_layout);

        auto compute_pipeline_result = device.createComputePipeline(nullptr, compute_pipeline_info);
        auto compute_pipeline = compute_pipeline_result.value;

        Logger::instance().info("Created compute pipeline");

        // Create IFS params uniform buffer
        IFSParams ifs_params{
            .iteration_count = 20,
            .particle_count = PARTICLE_COUNT,
            .scale = 1.0f,
            .random_seed = static_cast<uint32_t>(std::random_device{}())
        };

        auto params_buffer_info = vk::BufferCreateInfo()
            .setSize(sizeof(IFSParams))
            .setUsage(vk::BufferUsageFlagBits::eUniformBuffer)
            .setSharingMode(vk::SharingMode::eExclusive);
        auto params_buffer = device.createBuffer(params_buffer_info);

        auto params_mem_req = device.getBufferMemoryRequirements(params_buffer);
        uint32_t params_mem_type = 0;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
            if ((params_mem_req.memoryTypeBits & (1 << i)) &&
                (mem_props.memoryTypes[i].propertyFlags &
                 (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent))) {
                params_mem_type = i;
                break;
            }
        }

        auto params_mem_alloc = vk::MemoryAllocateInfo()
            .setAllocationSize(params_mem_req.size)
            .setMemoryTypeIndex(params_mem_type);
        auto params_memory = device.allocateMemory(params_mem_alloc);
        device.bindBufferMemory(params_buffer, params_memory, 0);

        // Upload params
        void* params_data = device.mapMemory(params_memory, 0, sizeof(IFSParams));
        std::memcpy(params_data, &ifs_params, sizeof(IFSParams));
        device.unmapMemory(params_memory);

        // Create descriptor pool for compute
        std::vector<vk::DescriptorPoolSize> compute_pool_sizes = {
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1),
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1)
        };
        auto compute_pool_info = vk::DescriptorPoolCreateInfo()
            .setMaxSets(1)
            .setPoolSizes(compute_pool_sizes);
        auto compute_descriptor_pool = device.createDescriptorPool(compute_pool_info);

        // Allocate descriptor set
        auto compute_set_alloc_info = vk::DescriptorSetAllocateInfo()
            .setDescriptorPool(compute_descriptor_pool)
            .setSetLayouts(compute_descriptor_layout);
        auto compute_descriptor_sets = device.allocateDescriptorSets(compute_set_alloc_info);
        auto compute_descriptor_set = compute_descriptor_sets[0];

        // Update descriptor set
        auto particle_buffer_info_desc = vk::DescriptorBufferInfo()
            .setBuffer(particle_buffer)
            .setOffset(0)
            .setRange(VK_WHOLE_SIZE);

        auto params_buffer_info_desc = vk::DescriptorBufferInfo()
            .setBuffer(params_buffer)
            .setOffset(0)
            .setRange(sizeof(IFSParams));

        std::vector<vk::WriteDescriptorSet> write_sets = {
            vk::WriteDescriptorSet()
                .setDstSet(compute_descriptor_set)
                .setDstBinding(0)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setBufferInfo(particle_buffer_info_desc),
            vk::WriteDescriptorSet()
                .setDstSet(compute_descriptor_set)
                .setDstBinding(1)
                .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                .setBufferInfo(params_buffer_info_desc)
        };
        device.updateDescriptorSets(write_sets, nullptr);

        Logger::instance().info("Created compute descriptor sets");

        // Create compute command pool
        auto compute_cmd_pool_info = vk::CommandPoolCreateInfo()
            .setQueueFamilyIndex(context.queue_indices().compute)
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        auto compute_command_pool = device.createCommandPool(compute_cmd_pool_info);

        // Dispatch compute shader
        auto compute_cmd_alloc = vk::CommandBufferAllocateInfo()
            .setCommandPool(compute_command_pool)
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(1);
        auto compute_cmd_buffers = device.allocateCommandBuffers(compute_cmd_alloc);
        auto compute_cmd = compute_cmd_buffers[0];

        compute_cmd.begin(vk::CommandBufferBeginInfo{});
        compute_cmd.bindPipeline(vk::PipelineBindPoint::eCompute, compute_pipeline);
        compute_cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, compute_pipeline_layout,
                                        0, compute_descriptor_set, nullptr);

        uint32_t dispatch_count = (PARTICLE_COUNT + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
        compute_cmd.dispatch(dispatch_count, 1, 1);

        // Queue family ownership transfer: compute -> graphics
        // Only needed if compute and graphics queue families differ
        if (context.queue_indices().compute != context.queue_indices().graphics) {
            auto buffer_barrier = vk::BufferMemoryBarrier()
                .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                .setDstAccessMask(vk::AccessFlags{})  // Graphics queue will acquire
                .setSrcQueueFamilyIndex(context.queue_indices().compute)
                .setDstQueueFamilyIndex(context.queue_indices().graphics)
                .setBuffer(particle_buffer)
                .setOffset(0)
                .setSize(VK_WHOLE_SIZE);

            compute_cmd.pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eBottomOfPipe,
                vk::DependencyFlags{}, nullptr, buffer_barrier, nullptr);
        } else {
            // Same queue family, just need a memory barrier
            auto barrier = vk::MemoryBarrier()
                .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                .setDstAccessMask(vk::AccessFlagBits::eVertexAttributeRead);
            compute_cmd.pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eVertexInput,
                vk::DependencyFlags{}, barrier, nullptr, nullptr);
        }

        compute_cmd.end();

        auto compute_submit = vk::SubmitInfo().setCommandBuffers(compute_cmd);
        context.compute_queue().submit(compute_submit);
        context.compute_queue().waitIdle();

        Logger::instance().info("Dispatched compute shader ({} work groups)", dispatch_count);

        // Load graphics shaders
        auto vert_shader_result = Shader::create_shader(device, "ifs/ifs_particle.vert");
        if (!vert_shader_result.has_value()) {
            std::println(std::cerr, "Failed to load vertex shader: {}", vert_shader_result.error());
            return 1;
        }
        auto vert_shader = std::move(*vert_shader_result);

        auto frag_shader_result = Shader::create_shader(device, "ifs/ifs_particle.frag");
        if (!frag_shader_result.has_value()) {
            std::println(std::cerr, "Failed to load fragment shader: {}", frag_shader_result.error());
            return 1;
        }
        auto frag_shader = std::move(*frag_shader_result);

        Logger::instance().info("Loaded graphics shaders");

        // Create graphics descriptor set layout
        auto& graphics_descriptors = vert_shader.get_descriptor_infos();
        auto& frag_descriptors = frag_shader.get_descriptor_infos();

        std::vector<vk::DescriptorSetLayoutBinding> graphics_bindings;
        for (const auto& desc : graphics_descriptors) {
            graphics_bindings.push_back(vk::DescriptorSetLayoutBinding()
                .setBinding(desc.binding)
                .setDescriptorType(desc.type)
                .setDescriptorCount(desc.descriptor_count)
                .setStageFlags(desc.stage));
        }
        for (const auto& desc : frag_descriptors) {
            graphics_bindings.push_back(vk::DescriptorSetLayoutBinding()
                .setBinding(desc.binding)
                .setDescriptorType(desc.type)
                .setDescriptorCount(desc.descriptor_count)
                .setStageFlags(desc.stage));
        }

        auto graphics_layout_info = vk::DescriptorSetLayoutCreateInfo()
            .setBindings(graphics_bindings);
        auto graphics_descriptor_layout = device.createDescriptorSetLayout(graphics_layout_info);

        // Create graphics pipeline layout
        auto graphics_pipeline_layout_info = vk::PipelineLayoutCreateInfo()
            .setSetLayouts(graphics_descriptor_layout);
        auto graphics_pipeline_layout = device.createPipelineLayout(graphics_pipeline_layout_info);

        // Create graphics pipeline
        auto vert_stage = vert_shader.create_pipeline_shader_stage_create_info();
        auto frag_stage = frag_shader.create_pipeline_shader_stage_create_info();
        std::array shader_stages = {vert_stage, frag_stage};

        auto vertex_input_info = vk::PipelineVertexInputStateCreateInfo();  // No vertex attributes

        auto input_assembly = vk::PipelineInputAssemblyStateCreateInfo()
            .setTopology(vk::PrimitiveTopology::ePointList)
            .setPrimitiveRestartEnable(false);

        auto viewport = vk::Viewport()
            .setX(0.0f).setY(0.0f)
            .setWidth(static_cast<float>(swapchain_data.extent.width))
            .setHeight(static_cast<float>(swapchain_data.extent.height))
            .setMinDepth(0.0f).setMaxDepth(1.0f);

        auto scissor = vk::Rect2D()
            .setOffset({0, 0})
            .setExtent(swapchain_data.extent);

        auto viewport_state = vk::PipelineViewportStateCreateInfo()
            .setViewports(viewport)
            .setScissors(scissor);

        auto rasterizer = vk::PipelineRasterizationStateCreateInfo()
            .setDepthClampEnable(false)
            .setRasterizerDiscardEnable(false)
            .setPolygonMode(vk::PolygonMode::eFill)
            .setCullMode(vk::CullModeFlagBits::eNone)
            .setFrontFace(vk::FrontFace::eCounterClockwise)
            .setDepthBiasEnable(false)
            .setLineWidth(1.0f);

        auto multisampling = vk::PipelineMultisampleStateCreateInfo()
            .setRasterizationSamples(vk::SampleCountFlagBits::e1)
            .setSampleShadingEnable(false);

        auto color_blend_attachment = vk::PipelineColorBlendAttachmentState()
            .setBlendEnable(false)
            .setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                               vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

        auto color_blending = vk::PipelineColorBlendStateCreateInfo()
            .setLogicOpEnable(false)
            .setAttachments(color_blend_attachment);

        // Enable dynamic viewport and scissor
        std::array<vk::DynamicState, 2> dynamic_states = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        auto dynamic_state = vk::PipelineDynamicStateCreateInfo()
            .setDynamicStates(dynamic_states);

        auto pipeline_info = vk::GraphicsPipelineCreateInfo()
            .setStages(shader_stages)
            .setPVertexInputState(&vertex_input_info)
            .setPInputAssemblyState(&input_assembly)
            .setPViewportState(&viewport_state)
            .setPRasterizationState(&rasterizer)
            .setPMultisampleState(&multisampling)
            .setPColorBlendState(&color_blending)
            .setPDynamicState(&dynamic_state)
            .setLayout(graphics_pipeline_layout)
            .setRenderPass(render_pass)
            .setSubpass(0);

        auto graphics_pipeline_result = device.createGraphicsPipeline(nullptr, pipeline_info);
        auto graphics_pipeline = graphics_pipeline_result.value;

        Logger::instance().info("Created graphics pipeline");

        // Create view params and color params buffers
        struct ViewParams {
            glm::vec2 screen_size;
            float point_size;
            float padding;
        } view_params{glm::vec2(swapchain_data.extent.width, swapchain_data.extent.height), 1.0f, 0.0f};

        struct ColorParams {
            glm::vec4 color;
        } color_params{glm::vec4(1.0f)};

        auto view_buffer_info = vk::BufferCreateInfo()
            .setSize(sizeof(ViewParams))
            .setUsage(vk::BufferUsageFlagBits::eUniformBuffer);
        auto view_buffer = device.createBuffer(view_buffer_info);

        auto view_mem_req = device.getBufferMemoryRequirements(view_buffer);
        uint32_t view_mem_type = 0;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
            if ((view_mem_req.memoryTypeBits & (1 << i)) &&
                (mem_props.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)) {
                view_mem_type = i;
                break;
            }
        }

        auto view_mem_alloc = vk::MemoryAllocateInfo()
            .setAllocationSize(view_mem_req.size)
            .setMemoryTypeIndex(view_mem_type);
        auto view_memory = device.allocateMemory(view_mem_alloc);
        device.bindBufferMemory(view_buffer, view_memory, 0);

        void* view_data = device.mapMemory(view_memory, 0, sizeof(ViewParams));
        std::memcpy(view_data, &view_params, sizeof(ViewParams));
        device.unmapMemory(view_memory);

        auto color_buffer_info = vk::BufferCreateInfo()
            .setSize(sizeof(ColorParams))
            .setUsage(vk::BufferUsageFlagBits::eUniformBuffer);
        auto color_buffer = device.createBuffer(color_buffer_info);

        auto color_mem_req = device.getBufferMemoryRequirements(color_buffer);
        auto color_mem_alloc = vk::MemoryAllocateInfo()
            .setAllocationSize(color_mem_req.size)
            .setMemoryTypeIndex(view_mem_type);
        auto color_memory = device.allocateMemory(color_mem_alloc);
        device.bindBufferMemory(color_buffer, color_memory, 0);

        void* color_data = device.mapMemory(color_memory, 0, sizeof(ColorParams));
        std::memcpy(color_data, &color_params, sizeof(ColorParams));
        device.unmapMemory(color_memory);

        // Create graphics descriptor pool and set
        std::vector<vk::DescriptorPoolSize> graphics_pool_sizes = {
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1),
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2)
        };
        auto graphics_pool_info = vk::DescriptorPoolCreateInfo()
            .setMaxSets(1)
            .setPoolSizes(graphics_pool_sizes);
        auto graphics_descriptor_pool = device.createDescriptorPool(graphics_pool_info);

        auto graphics_set_alloc_info = vk::DescriptorSetAllocateInfo()
            .setDescriptorPool(graphics_descriptor_pool)
            .setSetLayouts(graphics_descriptor_layout);
        auto graphics_descriptor_sets = device.allocateDescriptorSets(graphics_set_alloc_info);
        auto graphics_descriptor_set = graphics_descriptor_sets[0];

        // Update graphics descriptor set
        auto particle_buffer_graphics_info = vk::DescriptorBufferInfo()
            .setBuffer(particle_buffer)
            .setOffset(0)
            .setRange(VK_WHOLE_SIZE);

        auto view_buffer_info_desc = vk::DescriptorBufferInfo()
            .setBuffer(view_buffer)
            .setOffset(0)
            .setRange(sizeof(ViewParams));

        auto color_buffer_info_desc = vk::DescriptorBufferInfo()
            .setBuffer(color_buffer)
            .setOffset(0)
            .setRange(sizeof(ColorParams));

        std::vector<vk::WriteDescriptorSet> graphics_write_sets = {
            vk::WriteDescriptorSet()
                .setDstSet(graphics_descriptor_set)
                .setDstBinding(0)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setBufferInfo(particle_buffer_graphics_info),
            vk::WriteDescriptorSet()
                .setDstSet(graphics_descriptor_set)
                .setDstBinding(1)
                .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                .setBufferInfo(view_buffer_info_desc),
            vk::WriteDescriptorSet()
                .setDstSet(graphics_descriptor_set)
                .setDstBinding(2)
                .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                .setBufferInfo(color_buffer_info_desc)
        };
        device.updateDescriptorSets(graphics_write_sets, nullptr);

        Logger::instance().info("Created graphics descriptor sets");

        // One-time queue ownership acquire (only needed once since we're not giving it back)
        if (context.queue_indices().compute != context.queue_indices().graphics) {
            auto acquire_cmd_alloc = vk::CommandBufferAllocateInfo()
                .setCommandPool(command_pool)
                .setLevel(vk::CommandBufferLevel::ePrimary)
                .setCommandBufferCount(1);
            auto acquire_cmd_buffers = device.allocateCommandBuffers(acquire_cmd_alloc);
            auto acquire_cmd = acquire_cmd_buffers[0];

            acquire_cmd.begin(vk::CommandBufferBeginInfo{});

            auto acquire_barrier = vk::BufferMemoryBarrier()
                .setSrcAccessMask(vk::AccessFlags{})
                .setDstAccessMask(vk::AccessFlagBits::eVertexAttributeRead)
                .setSrcQueueFamilyIndex(context.queue_indices().compute)
                .setDstQueueFamilyIndex(context.queue_indices().graphics)
                .setBuffer(particle_buffer)
                .setOffset(0)
                .setSize(VK_WHOLE_SIZE);

            acquire_cmd.pipelineBarrier(
                vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eVertexInput,
                vk::DependencyFlags{}, nullptr, acquire_barrier, nullptr);

            acquire_cmd.end();

            auto acquire_submit = vk::SubmitInfo().setCommandBuffers(acquire_cmd);
            context.graphics_queue().submit(acquire_submit);
            context.graphics_queue().waitIdle();

            device.freeCommandBuffers(command_pool, acquire_cmd_buffers);
            Logger::instance().info("Acquired particle buffer ownership for graphics queue");
        }

        // Initialize ImGui
        Logger::instance().info("Initializing ImGui...");

        // Create ImGui descriptor pool
        std::vector<vk::DescriptorPoolSize> imgui_pool_sizes = {
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
            .setPoolSizes(imgui_pool_sizes);
        auto imgui_descriptor_pool = device.createDescriptorPool(imgui_pool_info);

        // Setup ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        // Setup ImGui style
        ImGui::StyleColorsDark();

        // Initialize ImGui for GLFW
        ImGui_ImplGlfw_InitForVulkan(window.get_window_handle(), true);

        // Initialize ImGui for Vulkan
        ImGui_ImplVulkan_InitInfo init_info{};
        init_info.Instance = static_cast<VkInstance>(context.instance());
        init_info.PhysicalDevice = static_cast<VkPhysicalDevice>(physical_device);
        init_info.Device = static_cast<VkDevice>(device);
        init_info.QueueFamily = context.queue_indices().graphics;
        init_info.Queue = static_cast<VkQueue>(context.graphics_queue());
        init_info.PipelineCache = VK_NULL_HANDLE;
        init_info.DescriptorPool = static_cast<VkDescriptorPool>(imgui_descriptor_pool);
        init_info.RenderPass = static_cast<VkRenderPass>(render_pass);
        init_info.Subpass = 0;
        init_info.MinImageCount = static_cast<uint32_t>(swapchain_data.images.size());
        init_info.ImageCount = static_cast<uint32_t>(swapchain_data.images.size());
        init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        init_info.Allocator = nullptr;
        init_info.CheckVkResultFn = nullptr;

        ImGui_ImplVulkan_Init(&init_info);

        // Upload fonts
        ImGui_ImplVulkan_CreateFontsTexture();

        Logger::instance().info("ImGui initialized");

        // UI state variables
        uint32_t current_iterations = 20;
        float current_scale = 1.0f;
        uint32_t current_particle_count = PARTICLE_COUNT;
        bool needs_recompute = false;
        bool needs_buffer_recreate = false;

        // Create synchronization primitives
        auto image_available_semaphore = device.createSemaphore(vk::SemaphoreCreateInfo{});
        auto render_finished_semaphore = device.createSemaphore(vk::SemaphoreCreateInfo{});

        // Main render loop
        Logger::instance().info("Starting render loop...");

        while (!glfwWindowShouldClose(window.get_window_handle())) {
            glfwPollEvents();

            // Start ImGui frame
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // Build UI
            ImGui::Begin("IFS Controls");

            ImGui::Text("Sierpinski Triangle Visualizer");
            ImGui::Separator();

            if (ImGui::Button("Reset")) {
                // Generate new random seed
                std::random_device rd;
                ifs_params.random_seed = rd();
                needs_recompute = true;
            }

            int iterations_int = static_cast<int>(current_iterations);
            if (ImGui::SliderInt("Iterations", &iterations_int, 1, 100)) {
                current_iterations = static_cast<uint32_t>(iterations_int);
                ifs_params.iteration_count = current_iterations;
                needs_recompute = true;
            }

            if (ImGui::SliderFloat("Scale", &current_scale, 0.5f, 2.0f)) {
                ifs_params.scale = current_scale;
                needs_recompute = true;
            }

            int particle_count_thousands = static_cast<int>(current_particle_count / 1000);
            if (ImGui::SliderInt("Particles (x1000)", &particle_count_thousands, 10, 2000)) {
                uint32_t new_count = static_cast<uint32_t>(particle_count_thousands) * 1000;
                if (new_count != current_particle_count) {
                    current_particle_count = new_count;
                    needs_buffer_recreate = true;
                }
            }

            ImGui::Separator();
            ImGui::Text("Current particles: %u", current_particle_count);
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                       1000.0f / io.Framerate, io.Framerate);

            ImGui::End();

            // Finish ImGui frame
            ImGui::Render();

            // Recreate particle buffer if count changed
            if (needs_buffer_recreate) {
                needs_buffer_recreate = false;
                device.waitIdle();

                Logger::instance().info("Recreating particle buffer with {} particles", current_particle_count);

                // Destroy old buffer and memory
                device.destroyBuffer(particle_buffer);
                device.freeMemory(particle_memory);

                // Create new particle data
                std::vector<Particle> new_particles(current_particle_count);
                {
                    std::random_device rd;
                    std::default_random_engine rng(rd());
                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                    for (auto& p : new_particles) {
                        p.position = glm::vec2(dist(rng), dist(rng));
                        p.padding = glm::vec2(0.0f);
                    }
                }

                vk::DeviceSize new_buffer_size = sizeof(Particle) * current_particle_count;

                // Create staging buffer
                auto new_staging_buffer_info = vk::BufferCreateInfo()
                    .setSize(new_buffer_size)
                    .setUsage(vk::BufferUsageFlagBits::eTransferSrc)
                    .setSharingMode(vk::SharingMode::eExclusive);
                auto new_staging_buffer = device.createBuffer(new_staging_buffer_info);

                auto new_staging_mem_req = device.getBufferMemoryRequirements(new_staging_buffer);
                uint32_t new_staging_mem_type = 0;
                for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
                    if ((new_staging_mem_req.memoryTypeBits & (1 << i)) &&
                        (mem_props.memoryTypes[i].propertyFlags &
                         (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent))) {
                        new_staging_mem_type = i;
                        break;
                    }
                }

                auto new_staging_mem_alloc = vk::MemoryAllocateInfo()
                    .setAllocationSize(new_staging_mem_req.size)
                    .setMemoryTypeIndex(new_staging_mem_type);
                auto new_staging_memory = device.allocateMemory(new_staging_mem_alloc);
                device.bindBufferMemory(new_staging_buffer, new_staging_memory, 0);

                void* staging_data = device.mapMemory(new_staging_memory, 0, new_buffer_size);
                std::memcpy(staging_data, new_particles.data(), new_buffer_size);
                device.unmapMemory(new_staging_memory);

                // Create device-local particle buffer
                auto new_particle_buffer_info = vk::BufferCreateInfo()
                    .setSize(new_buffer_size)
                    .setUsage(vk::BufferUsageFlagBits::eStorageBuffer |
                              vk::BufferUsageFlagBits::eTransferDst)
                    .setSharingMode(vk::SharingMode::eExclusive);
                particle_buffer = device.createBuffer(new_particle_buffer_info);

                auto new_particle_mem_req = device.getBufferMemoryRequirements(particle_buffer);
                uint32_t new_particle_mem_type = 0;
                for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
                    if ((new_particle_mem_req.memoryTypeBits & (1 << i)) &&
                        (mem_props.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
                        new_particle_mem_type = i;
                        break;
                    }
                }

                auto new_particle_mem_alloc = vk::MemoryAllocateInfo()
                    .setAllocationSize(new_particle_mem_req.size)
                    .setMemoryTypeIndex(new_particle_mem_type);
                particle_memory = device.allocateMemory(new_particle_mem_alloc);
                device.bindBufferMemory(particle_buffer, particle_memory, 0);

                // Copy staging to particle buffer
                auto copy_cmd_alloc = vk::CommandBufferAllocateInfo()
                    .setCommandPool(command_pool)
                    .setLevel(vk::CommandBufferLevel::ePrimary)
                    .setCommandBufferCount(1);
                auto copy_cmd_buffers = device.allocateCommandBuffers(copy_cmd_alloc);
                auto copy_cmd = copy_cmd_buffers[0];

                copy_cmd.begin(vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
                auto copy_region = vk::BufferCopy().setSize(new_buffer_size);
                copy_cmd.copyBuffer(new_staging_buffer, particle_buffer, copy_region);
                copy_cmd.end();

                auto copy_submit = vk::SubmitInfo().setCommandBuffers(copy_cmd);
                context.graphics_queue().submit(copy_submit);
                context.graphics_queue().waitIdle();

                device.freeCommandBuffers(command_pool, copy_cmd_buffers);
                device.destroyBuffer(new_staging_buffer);
                device.freeMemory(new_staging_memory);

                // Update descriptor sets
                auto new_particle_buffer_compute_info = vk::DescriptorBufferInfo()
                    .setBuffer(particle_buffer)
                    .setOffset(0)
                    .setRange(VK_WHOLE_SIZE);

                auto new_particle_buffer_graphics_info = vk::DescriptorBufferInfo()
                    .setBuffer(particle_buffer)
                    .setOffset(0)
                    .setRange(VK_WHOLE_SIZE);

                std::vector<vk::WriteDescriptorSet> update_writes = {
                    vk::WriteDescriptorSet()
                        .setDstSet(compute_descriptor_set)
                        .setDstBinding(0)
                        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                        .setBufferInfo(new_particle_buffer_compute_info),
                    vk::WriteDescriptorSet()
                        .setDstSet(graphics_descriptor_set)
                        .setDstBinding(0)
                        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                        .setBufferInfo(new_particle_buffer_graphics_info)
                };
                device.updateDescriptorSets(update_writes, nullptr);

                // Update params buffer
                ifs_params.particle_count = current_particle_count;
                void* params_data = device.mapMemory(params_memory, 0, sizeof(IFSParams));
                std::memcpy(params_data, &ifs_params, sizeof(IFSParams));
                device.unmapMemory(params_memory);

                // Trigger recompute to run compute shader on new buffer
                needs_recompute = true;

                Logger::instance().info("Particle buffer recreated");
            }

            // Re-compute if parameters changed
            if (needs_recompute) {
                needs_recompute = false;

                // Wait for all operations to complete
                device.waitIdle();

                // Update params buffer
                ifs_params.iteration_count = current_iterations;
                void* params_data = device.mapMemory(params_memory, 0, sizeof(IFSParams));
                std::memcpy(params_data, &ifs_params, sizeof(IFSParams));
                device.unmapMemory(params_memory);

                // Re-dispatch compute shader
                auto recompute_cmd_alloc = vk::CommandBufferAllocateInfo()
                    .setCommandPool(compute_command_pool)
                    .setLevel(vk::CommandBufferLevel::ePrimary)
                    .setCommandBufferCount(1);
                auto recompute_cmd_buffers = device.allocateCommandBuffers(recompute_cmd_alloc);
                auto recompute_cmd = recompute_cmd_buffers[0];

                recompute_cmd.begin(vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
                recompute_cmd.bindPipeline(vk::PipelineBindPoint::eCompute, compute_pipeline);
                recompute_cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, compute_pipeline_layout,
                                                  0, compute_descriptor_set, nullptr);

                uint32_t work_groups_x = (current_particle_count + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
                recompute_cmd.dispatch(work_groups_x, 1, 1);

                // Release particle buffer ownership
                if (context.queue_indices().compute != context.queue_indices().graphics) {
                    auto release_barrier = vk::BufferMemoryBarrier()
                        .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                        .setDstAccessMask(vk::AccessFlags{})
                        .setSrcQueueFamilyIndex(context.queue_indices().compute)
                        .setDstQueueFamilyIndex(context.queue_indices().graphics)
                        .setBuffer(particle_buffer)
                        .setOffset(0)
                        .setSize(VK_WHOLE_SIZE);

                    recompute_cmd.pipelineBarrier(
                        vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eBottomOfPipe,
                        vk::DependencyFlags{}, nullptr, release_barrier, nullptr);
                }

                recompute_cmd.end();

                auto recompute_submit = vk::SubmitInfo().setCommandBuffers(recompute_cmd);
                context.compute_queue().submit(recompute_submit);
                context.compute_queue().waitIdle();

                device.freeCommandBuffers(compute_command_pool, recompute_cmd_buffers);

                // Acquire ownership on graphics queue
                if (context.queue_indices().compute != context.queue_indices().graphics) {
                    auto acquire_cmd_alloc = vk::CommandBufferAllocateInfo()
                        .setCommandPool(command_pool)
                        .setLevel(vk::CommandBufferLevel::ePrimary)
                        .setCommandBufferCount(1);
                    auto acquire_cmd_buffers = device.allocateCommandBuffers(acquire_cmd_alloc);
                    auto acquire_cmd = acquire_cmd_buffers[0];

                    acquire_cmd.begin(vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

                    auto acquire_barrier = vk::BufferMemoryBarrier()
                        .setSrcAccessMask(vk::AccessFlags{})
                        .setDstAccessMask(vk::AccessFlagBits::eVertexAttributeRead)
                        .setSrcQueueFamilyIndex(context.queue_indices().compute)
                        .setDstQueueFamilyIndex(context.queue_indices().graphics)
                        .setBuffer(particle_buffer)
                        .setOffset(0)
                        .setSize(VK_WHOLE_SIZE);

                    acquire_cmd.pipelineBarrier(
                        vk::PipelineStageFlagBits::eTopOfPipe,
                        vk::PipelineStageFlagBits::eVertexInput,
                        vk::DependencyFlags{}, nullptr, acquire_barrier, nullptr);

                    acquire_cmd.end();

                    auto acquire_submit = vk::SubmitInfo().setCommandBuffers(acquire_cmd);
                    context.graphics_queue().submit(acquire_submit);
                    context.graphics_queue().waitIdle();

                    device.freeCommandBuffers(command_pool, acquire_cmd_buffers);
                }

                Logger::instance().info("Re-computed IFS with new parameters");
            }

            // Handle swapchain recreation from resize callback
            if (framebuffer_resized) {
                framebuffer_resized = false;
                device.waitIdle();

                // Save old swapchain handle before creating new one
                auto old_swapchain_handle = swapchain_data.swapchain;

                // Create new swapchain
                auto new_swapchain_data = create_swapchain(device, physical_device, surface,
                                                           render_pass, window.get_window_handle(),
                                                           old_swapchain_handle);

                // Cleanup old swapchain resources
                cleanup_swapchain(device, swapchain_data);

                // Use new swapchain
                swapchain_data = new_swapchain_data;

                // Update view params with new extent
                view_params.screen_size = glm::vec2(swapchain_data.extent.width, swapchain_data.extent.height);
                void* view_data = device.mapMemory(view_memory, 0, sizeof(ViewParams));
                std::memcpy(view_data, &view_params, sizeof(ViewParams));
                device.unmapMemory(view_memory);

                Logger::instance().info("Swapchain recreated with extent {}x{}",
                                       swapchain_data.extent.width, swapchain_data.extent.height);
                continue;
            }

            // Acquire next image
            uint32_t image_index;
            auto result = device.acquireNextImageKHR(swapchain_data.swapchain, UINT64_MAX,
                                                     image_available_semaphore, nullptr, &image_index);
            if (result == vk::Result::eErrorOutOfDateKHR) {
                framebuffer_resized = true;
                continue;
            } else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
                throw std::runtime_error("Failed to acquire swapchain image");
            }

            // Record command buffer
            auto cmd_alloc = vk::CommandBufferAllocateInfo()
                .setCommandPool(command_pool)
                .setLevel(vk::CommandBufferLevel::ePrimary)
                .setCommandBufferCount(1);
            auto cmd_buffers = device.allocateCommandBuffers(cmd_alloc);
            auto cmd = cmd_buffers[0];

            cmd.begin(vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

            vk::ClearValue clear_color(vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}));
            auto render_pass_begin = vk::RenderPassBeginInfo()
                .setRenderPass(render_pass)
                .setFramebuffer(swapchain_data.framebuffers[image_index])
                .setRenderArea(vk::Rect2D({0, 0}, swapchain_data.extent))
                .setClearValues(clear_color);

            cmd.beginRenderPass(render_pass_begin, vk::SubpassContents::eInline);
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipeline);

            // Set dynamic viewport and scissor
            vk::Viewport dynamic_viewport{
                0.0f, 0.0f,
                static_cast<float>(swapchain_data.extent.width),
                static_cast<float>(swapchain_data.extent.height),
                0.0f, 1.0f
            };
            cmd.setViewport(0, dynamic_viewport);

            vk::Rect2D dynamic_scissor{{0, 0}, swapchain_data.extent};
            cmd.setScissor(0, dynamic_scissor);

            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics_pipeline_layout,
                                    0, graphics_descriptor_set, nullptr);
            cmd.draw(current_particle_count, 1, 0, 0);

            // Render ImGui
            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), static_cast<VkCommandBuffer>(cmd));

            cmd.endRenderPass();

            cmd.end();

            // Submit
            vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
            auto submit = vk::SubmitInfo()
                .setWaitSemaphores(image_available_semaphore)
                .setWaitDstStageMask(wait_stage)
                .setCommandBuffers(cmd)
                .setSignalSemaphores(render_finished_semaphore);
            context.graphics_queue().submit(submit);

            // Present
            auto present_info = vk::PresentInfoKHR()
                .setWaitSemaphores(render_finished_semaphore)
                .setSwapchains(swapchain_data.swapchain)
                .setImageIndices(image_index);
            auto present_result = context.graphics_queue().presentKHR(present_info);

            if (present_result == vk::Result::eErrorOutOfDateKHR || present_result == vk::Result::eSuboptimalKHR) {
                framebuffer_resized = true;
            } else if (present_result != vk::Result::eSuccess) {
                throw std::runtime_error("Failed to present swapchain image");
            }

            // Simple synchronization - wait for queue to finish
            // (In production, use fences for better performance)
            context.graphics_queue().waitIdle();

            device.freeCommandBuffers(command_pool, cmd_buffers);
        }

        Logger::instance().info("Render loop finished");

        // Cleanup semaphores
        device.destroySemaphore(image_available_semaphore);
        device.destroySemaphore(render_finished_semaphore);

        // Cleanup ImGui
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        device.destroyDescriptorPool(imgui_descriptor_pool);

        // Cleanup
        Logger::instance().info("Cleaning up...");
        device.waitIdle();

        // Graphics cleanup
        device.destroyPipeline(graphics_pipeline);
        device.destroyPipelineLayout(graphics_pipeline_layout);
        device.destroyDescriptorSetLayout(graphics_descriptor_layout);
        device.destroyDescriptorPool(graphics_descriptor_pool);
        device.destroyBuffer(view_buffer);
        device.freeMemory(view_memory);
        device.destroyBuffer(color_buffer);
        device.freeMemory(color_memory);

        // Compute cleanup
        device.destroyPipeline(compute_pipeline);
        device.destroyPipelineLayout(compute_pipeline_layout);
        device.destroyDescriptorSetLayout(compute_descriptor_layout);
        device.destroyDescriptorPool(compute_descriptor_pool);
        device.destroyCommandPool(compute_command_pool);
        device.destroyBuffer(params_buffer);
        device.freeMemory(params_memory);

        // Common cleanup
        cleanup_swapchain(device, swapchain_data);
        device.destroyRenderPass(render_pass);
        context.instance().destroySurfaceKHR(surface);
        device.destroyCommandPool(command_pool);
        device.destroyBuffer(particle_buffer);
        device.freeMemory(particle_memory);

        Logger::instance().info("IFS Visualizer completed successfully");
        return 0;

    } catch (const std::exception& e) {
        std::println(std::cerr, "Error: {}", e.what());
        return 1;
    }
}
