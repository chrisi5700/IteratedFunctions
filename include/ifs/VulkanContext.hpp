//
// Created by chris on 1/6/26.
//

#ifndef ITERATEDFUNCTIONS_VULKANCONTEXT_HPP
#define ITERATEDFUNCTIONS_VULKANCONTEXT_HPP

#include "Common.hpp"
#include <string_view>

struct QueueFamilyIndices
{
	uint32_t graphics;
	uint32_t compute;

	[[nodiscard]] bool has_dedicated_compute() const { return compute != graphics; }
};

class VulkanContext
{
public:
	explicit VulkanContext(std::string_view title);
	~VulkanContext();

	VulkanContext(const VulkanContext&) = delete;
	VulkanContext& operator=(const VulkanContext&) = delete;
	VulkanContext(VulkanContext&&) = delete;
	VulkanContext& operator=(VulkanContext&&) = delete;

	[[nodiscard]] vk::Instance instance() const { return m_instance; }
	[[nodiscard]] vk::PhysicalDevice physical_device() const { return m_physical_device; }
	[[nodiscard]] vk::Device device() const { return m_device; }
	[[nodiscard]] const QueueFamilyIndices& queue_indices() const { return m_queue_indices; }
	[[nodiscard]] vk::Queue graphics_queue() const { return m_graphics_queue; }
	[[nodiscard]] vk::Queue compute_queue() const { return m_compute_queue; }

private:
	vk::Instance m_instance;
	vk::DebugUtilsMessengerEXT m_debug_messenger;
	vk::PhysicalDevice m_physical_device;
	QueueFamilyIndices m_queue_indices;
	vk::Device m_device;
	vk::Queue m_graphics_queue;
	vk::Queue m_compute_queue;
};
#endif // ITERATEDFUNCTIONS_VULKANCONTEXT_HPP