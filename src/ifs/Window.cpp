#include <ifs/Window.hpp>
#include <ifs/Logger.hpp>
#include <algorithm>
#include <stdexcept>

namespace ifs {

// Static GLFW initialization
void Window::ensure_glfw_initialized() {
    static bool initialized = false;
    if (!initialized) {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
        initialized = true;
    }
}

std::expected<Window, std::string> Window::create(
    const VulkanContext& context,
    int width,
    int height,
    std::string_view title
) {
    try {
        Window window(context, width, height, title);

        if (auto result = window.initialize(); !result) {
            return std::unexpected(result.error());
        }

        return window;
    } catch (const std::exception& e) {
        return std::unexpected(std::string("Window creation failed: ") + e.what());
    }
}

Window::Window(
    const VulkanContext& context,
    int width,
    int height,
    std::string_view title
)
    : m_window_handle(nullptr)
    , m_width(width)
    , m_height(height)
    , m_context(&context)
    , m_device(context.device())
    , m_current_image_index(0)
    , m_needs_resize(false)
{
    ensure_glfw_initialized();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);  // Enable resizing

    m_window_handle = glfwCreateWindow(width, height, title.data(), nullptr, nullptr);
    if (!m_window_handle) {
        throw std::runtime_error("Failed to create GLFW window");
    }
}

Window::~Window() {
    cleanup();
    if (m_window_handle) {
        glfwDestroyWindow(m_window_handle);
    }
}

Window::Window(Window&& other) noexcept
    : m_window_handle(other.m_window_handle)
    , m_width(other.m_width)
    , m_height(other.m_height)
    , m_context(other.m_context)
    , m_device(other.m_device)
    , m_surface(other.m_surface)
    , m_surface_format(other.m_surface_format)
    , m_present_mode(other.m_present_mode)
    , m_swapchain(other.m_swapchain)
    , m_extent(other.m_extent)
    , m_swapchain_images(std::move(other.m_swapchain_images))
    , m_image_views(std::move(other.m_image_views))
    , m_framebuffers(std::move(other.m_framebuffers))
    , m_render_pass(other.m_render_pass)
    , m_depth_image(other.m_depth_image)
    , m_depth_memory(other.m_depth_memory)
    , m_depth_image_view(other.m_depth_image_view)
    , m_depth_format(other.m_depth_format)
    , m_current_image_index(other.m_current_image_index)
    , m_needs_resize(other.m_needs_resize)
{
    other.m_window_handle = nullptr;
    other.m_surface = nullptr;
    other.m_swapchain = nullptr;
    other.m_render_pass = nullptr;
    other.m_depth_image = nullptr;
    other.m_depth_memory = nullptr;
    other.m_depth_image_view = nullptr;
}

Window& Window::operator=(Window&& other) noexcept {
    if (this != &other) {
        cleanup();
        if (m_window_handle) {
            glfwDestroyWindow(m_window_handle);
        }

        m_window_handle = other.m_window_handle;
        m_width = other.m_width;
        m_height = other.m_height;
        m_context = other.m_context;
        m_device = other.m_device;
        m_surface = other.m_surface;
        m_surface_format = other.m_surface_format;
        m_present_mode = other.m_present_mode;
        m_swapchain = other.m_swapchain;
        m_extent = other.m_extent;
        m_swapchain_images = std::move(other.m_swapchain_images);
        m_image_views = std::move(other.m_image_views);
        m_framebuffers = std::move(other.m_framebuffers);
        m_render_pass = other.m_render_pass;
        m_depth_image = other.m_depth_image;
        m_depth_memory = other.m_depth_memory;
        m_depth_image_view = other.m_depth_image_view;
        m_depth_format = other.m_depth_format;
        m_current_image_index = other.m_current_image_index;
        m_needs_resize = other.m_needs_resize;

        other.m_window_handle = nullptr;
        other.m_surface = nullptr;
        other.m_swapchain = nullptr;
        other.m_render_pass = nullptr;
        other.m_depth_image = nullptr;
        other.m_depth_memory = nullptr;
        other.m_depth_image_view = nullptr;
    }
    return *this;
}

bool Window::should_close() const {
    return glfwWindowShouldClose(m_window_handle);
}

std::expected<void, std::string> Window::initialize() {
    if (auto result = create_surface(); !result) {
        return result;
    }

    if (auto result = create_swapchain(); !result) {
        return result;
    }

    if (auto result = create_depth_resources(); !result) {
        return result;
    }

    if (auto result = create_render_pass(); !result) {
        return result;
    }

    if (auto result = create_framebuffers(); !result) {
        return result;
    }

    return {};
}

std::expected<void, std::string> Window::create_surface() {
    VkSurfaceKHR surface_c;
    VkResult result = glfwCreateWindowSurface(
        static_cast<VkInstance>(m_context->instance()),
        m_window_handle,
        nullptr,
        &surface_c
    );

    if (result != VK_SUCCESS) {
        return std::unexpected("Failed to create window surface");
    }

    m_surface = vk::SurfaceKHR(surface_c);
    return {};
}

std::expected<void, std::string> Window::create_swapchain() {
    auto physical_device = m_context->physical_device();

    // Query swapchain support
    auto surface_capabilities_res = physical_device.getSurfaceCapabilitiesKHR(m_surface);
    auto surface_formats_res = physical_device.getSurfaceFormatsKHR(m_surface);
    auto present_modes_res = physical_device.getSurfacePresentModesKHR(m_surface);

    if (surface_formats_res.result != vk::Result::eSuccess or
    	present_modes_res.result != vk::Result::eSuccess or
    	surface_capabilities_res.result != vk::Result::eSuccess) {
        return std::unexpected("Inadequate swapchain support");
    }

    // Choose format, present mode, and extent

    m_surface_format = choose_surface_format(surface_formats_res.value);
    m_present_mode = choose_present_mode(present_modes_res.value);
    m_extent = choose_extent(surface_capabilities_res.value);

	auto surface_capabilities = surface_capabilities_res.value;
    // Determine image count
    uint32_t image_count = surface_capabilities.minImageCount + 1;
    if (surface_capabilities.maxImageCount > 0 &&
        image_count > surface_capabilities.maxImageCount) {
        image_count = surface_capabilities.maxImageCount;
    }

    // Create swapchain
    auto swapchain_info = vk::SwapchainCreateInfoKHR()
        .setSurface(m_surface)
        .setMinImageCount(image_count)
        .setImageFormat(m_surface_format.format)
        .setImageColorSpace(m_surface_format.colorSpace)
        .setImageExtent(m_extent)
        .setImageArrayLayers(1)
        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
        .setImageSharingMode(vk::SharingMode::eExclusive)
        .setPreTransform(surface_capabilities.currentTransform)
        .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
        .setPresentMode(m_present_mode)
        .setClipped(true);

	auto swapchain_res = m_device.createSwapchainKHR(swapchain_info);
	CHECK_VK_RESULT(swapchain_res, "Could not create swapchain {}");
    m_swapchain = swapchain_res.value;
	auto swapchain_imgs_res = m_device.getSwapchainImagesKHR(m_swapchain);
	CHECK_VK_RESULT(swapchain_imgs_res, "Could not get swapchain images {}");
    m_swapchain_images = swapchain_imgs_res.value;

    // Create image views
    m_image_views.clear();
    for (const auto& image : m_swapchain_images) {
        auto view_info = vk::ImageViewCreateInfo()
            .setImage(image)
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(m_surface_format.format)
            .setSubresourceRange(vk::ImageSubresourceRange()
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setBaseMipLevel(0)
                .setLevelCount(1)
                .setBaseArrayLayer(0)
                .setLayerCount(1));
    	auto img_view_res = m_device.createImageView(view_info);
    	CHECK_VK_RESULT(img_view_res, "Could not create image view {}");
        m_image_views.push_back(std::move(img_view_res.value));
    }

    return {};
}

std::expected<void, std::string> Window::create_render_pass() {
    auto color_attachment = vk::AttachmentDescription()
        .setFormat(m_surface_format.format)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

    auto depth_attachment = vk::AttachmentDescription()
        .setFormat(m_depth_format)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

    auto color_ref = vk::AttachmentReference()
        .setAttachment(0)
        .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

    auto depth_ref = vk::AttachmentReference()
        .setAttachment(1)
        .setLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

    auto subpass = vk::SubpassDescription()
        .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
        .setColorAttachments(color_ref)
        .setPDepthStencilAttachment(&depth_ref);

    auto dependency = vk::SubpassDependency()
        .setSrcSubpass(VK_SUBPASS_EXTERNAL)
        .setDstSubpass(0)
        .setSrcStageMask(
            vk::PipelineStageFlagBits::eColorAttachmentOutput |
            vk::PipelineStageFlagBits::eEarlyFragmentTests)
        .setDstStageMask(
            vk::PipelineStageFlagBits::eColorAttachmentOutput |
            vk::PipelineStageFlagBits::eEarlyFragmentTests)
        .setDstAccessMask(
            vk::AccessFlagBits::eColorAttachmentWrite |
            vk::AccessFlagBits::eDepthStencilAttachmentWrite);

    std::array<vk::AttachmentDescription, 2> attachments = {color_attachment, depth_attachment};

    auto render_pass_info = vk::RenderPassCreateInfo()
        .setAttachments(attachments)
        .setSubpasses(subpass)
        .setDependencies(dependency);

	auto render_pass_res = m_device.createRenderPass(render_pass_info);
	CHECK_VK_RESULT(render_pass_res, "Could not create render pass");
    m_render_pass = render_pass_res.value;
    return {};
}

std::expected<void, std::string> Window::create_framebuffers() {
    m_framebuffers.clear();
    for (const auto& view : m_image_views) {
        std::array<vk::ImageView, 2> attachments = {view, m_depth_image_view};

        auto framebuffer_info = vk::FramebufferCreateInfo()
            .setRenderPass(m_render_pass)
            .setAttachments(attachments)
            .setWidth(m_extent.width)
            .setHeight(m_extent.height)
            .setLayers(1);
    	auto frame_buffer_res = m_device.createFramebuffer(framebuffer_info);
    	CHECK_VK_RESULT(frame_buffer_res, "Could not create framebuffer {}");
        m_framebuffers.push_back(std::move(frame_buffer_res.value));
    }
    return {};
}

std::expected<void, std::string> Window::create_depth_resources() {
    m_depth_format = find_depth_format();

    // Create depth image
    auto image_info = vk::ImageCreateInfo()
        .setImageType(vk::ImageType::e2D)
        .setExtent(vk::Extent3D(m_extent.width, m_extent.height, 1))
        .setMipLevels(1)
        .setArrayLayers(1)
        .setFormat(m_depth_format)
        .setTiling(vk::ImageTiling::eOptimal)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setSamples(vk::SampleCountFlagBits::e1);

	auto img_res  = m_device.createImage(image_info);
	CHECK_VK_RESULT(img_res, "Could not create image {}");
    m_depth_image = img_res.value;

    // Allocate memory for depth image
    auto mem_requirements = m_device.getImageMemoryRequirements(m_depth_image);
    auto mem_properties = m_context->physical_device().getMemoryProperties();

    uint32_t memory_type_index = UINT32_MAX;
    vk::MemoryPropertyFlags required_props = vk::MemoryPropertyFlagBits::eDeviceLocal;

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((mem_requirements.memoryTypeBits & (1 << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & required_props) == required_props) {
            memory_type_index = i;
            break;
        }
    }

    if (memory_type_index == UINT32_MAX) {
        return std::unexpected("Failed to find suitable memory type for depth image");
    }

    auto alloc_info = vk::MemoryAllocateInfo()
        .setAllocationSize(mem_requirements.size)
        .setMemoryTypeIndex(memory_type_index);

	auto depth_mem_res = m_device.allocateMemory(alloc_info);
	CHECK_VK_RESULT(depth_mem_res, "Could not allocate depth memory {}");
    m_depth_memory = depth_mem_res.value;
    auto bind_res = m_device.bindImageMemory(m_depth_image, m_depth_memory, 0);
	CHECK_VK_RESULT_VOID(bind_res, "Could not bind depth image memory {}");
    // Create depth image view
    auto view_info = vk::ImageViewCreateInfo()
        .setImage(m_depth_image)
        .setViewType(vk::ImageViewType::e2D)
        .setFormat(m_depth_format)
        .setSubresourceRange(vk::ImageSubresourceRange()
            .setAspectMask(vk::ImageAspectFlagBits::eDepth)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(1));
	auto img_view_res = m_device.createImageView(view_info);
	CHECK_VK_RESULT(img_view_res, "Could not create image view {}");
    m_depth_image_view = img_view_res.value;
    return {};
}

vk::Format Window::find_depth_format() const {
    return find_supported_format(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment
    );
}

vk::Format Window::find_supported_format(
    const std::vector<vk::Format>& candidates,
    vk::ImageTiling tiling,
    vk::FormatFeatureFlags features
) const {
    for (vk::Format format : candidates) {
        auto props = m_context->physical_device().getFormatProperties(format);

        if (tiling == vk::ImageTiling::eLinear &&
            (props.linearTilingFeatures & features) == features) {
            return format;
        } else if (tiling == vk::ImageTiling::eOptimal &&
                   (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("Failed to find supported format");
}

std::optional<uint32_t> Window::acquire_next_image(
    vk::Semaphore signal_semaphore,
    uint64_t timeout
) {
    // Check if resize is needed
    if (m_needs_resize) {
        auto _ = m_device.waitIdle(); // Semantically makes no sense to error here

        if (auto result = recreate_swapchain(); !result) {
            Logger::instance().error("Failed to recreate swapchain: {}", result.error());
            return std::nullopt;
        }
        m_needs_resize = false;
        return std::nullopt;  // Signal caller to retry
    }

    uint32_t image_index;

	auto next_img_res = m_device.acquireNextImageKHR(m_swapchain, timeout, signal_semaphore, nullptr);
	vk::Result acquire_result = next_img_res.result;
	if (acquire_result == vk::Result::eSuccess ||
	    acquire_result == vk::Result::eErrorOutOfDateKHR ||
	    acquire_result == vk::Result::eSuboptimalKHR)
	{
		image_index = next_img_res.value;
	}
	else
	{
		Logger::instance().error("acquireNextImageKHR error: {}", to_string(acquire_result));
		return std::nullopt;
	}

    // Handle swapchain recreation
    if (acquire_result == vk::Result::eErrorOutOfDateKHR ||
        acquire_result == vk::Result::eSuboptimalKHR) {
        auto _ = m_device.waitIdle();
        if (auto result = recreate_swapchain(); !result) {
            Logger::instance().error("Failed to recreate swapchain: {}", result.error());
            return std::nullopt;
        }
        return std::nullopt;  // Signal caller to retry
    }

    if (acquire_result != vk::Result::eSuccess) {
        return std::nullopt;
    }

    m_current_image_index = image_index;
    return image_index;
}

bool Window::present(
    vk::Queue present_queue,
    vk::Semaphore wait_semaphore,
    uint32_t image_index
) {
    auto present_info = vk::PresentInfoKHR()
        .setWaitSemaphores(wait_semaphore)
        .setSwapchains(m_swapchain)
        .setImageIndices(image_index);

	VkResult result = vkQueuePresentKHR( // This hurts my soul but some genius at vk hpp decided that recreating a swapchain is a fatal error
	present_queue,
	reinterpret_cast<const VkPresentInfoKHR*>(&present_info)
	);

	vk::Result present_result = static_cast<vk::Result>(result);

	if (present_result == vk::Result::eErrorOutOfDateKHR)
	{
		// Swapchain is out of date, will be recreated on next acquire
		m_needs_resize = true;
		return false;
	}
	if (present_result != vk::Result::eSuccess && present_result != vk::Result::eSuboptimalKHR)
	{
		Logger::instance().error("presentKHR error: {}", to_string(present_result));
		return false;
	}
	// If suboptimal, we'll handle it on next acquire
	return true;
}

std::expected<void, std::string> Window::recreate_swapchain() {
    // Get new window size
    int width, height;
    glfwGetFramebufferSize(m_window_handle, &width, &height);

    // Handle minimization
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(m_window_handle, &width, &height);
        glfwWaitEvents();
    }

    m_width = width;
    m_height = height;

    // Cleanup old swapchain resources
    cleanup_swapchain();

    // Recreate swapchain, depth resources, and framebuffers
    if (auto result = create_swapchain(); !result) {
        return result;
    }

    if (auto result = create_depth_resources(); !result) {
        return result;
    }

    if (auto result = create_framebuffers(); !result) {
        return result;
    }

    Logger::instance().info("Swapchain recreated: {}x{}", width, height);
    return {};
}

void Window::cleanup_swapchain() {
    for (auto& fb : m_framebuffers) {
        m_device.destroyFramebuffer(fb);
    }
    m_framebuffers.clear();

    if (m_depth_image_view) {
        m_device.destroyImageView(m_depth_image_view);
        m_depth_image_view = nullptr;
    }

    if (m_depth_image) {
        m_device.destroyImage(m_depth_image);
        m_depth_image = nullptr;
    }

    if (m_depth_memory) {
        m_device.freeMemory(m_depth_memory);
        m_depth_memory = nullptr;
    }

    for (auto& view : m_image_views) {
        m_device.destroyImageView(view);
    }
    m_image_views.clear();

    if (m_swapchain) {
        m_device.destroySwapchainKHR(m_swapchain);
        m_swapchain = nullptr;
    }
}

void Window::cleanup() {
    if (m_device) {
        cleanup_swapchain();

        if (m_render_pass) {
            m_device.destroyRenderPass(m_render_pass);
            m_render_pass = nullptr;
        }

        if (m_surface) {
            m_context->instance().destroySurfaceKHR(m_surface);
            m_surface = nullptr;
        }
    }
}

vk::SurfaceFormatKHR Window::choose_surface_format(
    const std::vector<vk::SurfaceFormatKHR>& available_formats
) const {
    // Prefer SRGB if available
    for (const auto& format : available_formats) {
        if (format.format == vk::Format::eB8G8R8A8Srgb &&
            format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return format;
        }
    }
    return available_formats[0];
}

vk::PresentModeKHR Window::choose_present_mode(
    const std::vector<vk::PresentModeKHR>& available_modes
) const {
    // Prefer mailbox (triple buffering) if available
    for (const auto& mode : available_modes) {
        if (mode == vk::PresentModeKHR::eMailbox) {
            return mode;
        }
    }
    return vk::PresentModeKHR::eFifo;  // Always available
}

vk::Extent2D Window::choose_extent(const vk::SurfaceCapabilitiesKHR& capabilities) const {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    }

    // Window manager allows us to choose - clamp to valid range
    vk::Extent2D actual_extent = {
        static_cast<uint32_t>(m_width),
        static_cast<uint32_t>(m_height)
    };

    actual_extent.width = std::clamp(actual_extent.width,
        capabilities.minImageExtent.width,
        capabilities.maxImageExtent.width);
    actual_extent.height = std::clamp(actual_extent.height,
        capabilities.minImageExtent.height,
        capabilities.maxImageExtent.height);

    return actual_extent;
}

} // namespace ifs
