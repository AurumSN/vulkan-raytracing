#pragma once

#include "window.h"

#include <string>
#include <vector>

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct QueueFamilyIndices {
    uint32_t graphicsFamily;
    uint32_t presentFamily;
    bool graphicsFamilyHasValue = false;
    bool presentFamilyHasValue = false;
    bool isComplete() {
        return graphicsFamilyHasValue && presentFamilyHasValue;
    }
};

class Device {
    VkInstance _instance;
    VkDebugUtilsMessengerEXT _debugMessenger;
    VkPhysicalDevice _physicalDevice = VK_NULL_HANDLE;
    Window &_window;
    VkCommandPool _commandPool;

    VkDevice _device;
    VkSurfaceKHR _surface;
    VkQueue _graphicsQueue;
    VkQueue _presentQueue;
    VkQueue _computeQueue;

    const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();

    bool isDeviceSuitable(VkPhysicalDevice device);
    std::vector<const char *>getRequiredExtensions();
    bool checkValidationLayerSupport();
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo);
    void hasGlfwRequiredInstanceExtensions();
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

public:
#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

    Device(Window &window);
    ~Device();

    Device(const Device &) = delete;
    void operator=(const Device &) = delete;

    VkCommandPool getCommandPool() {
        return _commandPool;
    }

    VkDevice device() {
        return _device;
    }

    VkSurfaceKHR surface() {
        return _surface;
    }

    VkQueue graphicsQueue() {
        return _graphicsQueue;
    }

    VkQueue computeQueue() {
        return _computeQueue;
    }

    VkQueue presentQueue() {
        return _presentQueue;
    }

    VkPhysicalDevice physicalDevice() {
        return _physicalDevice;
    }

    SwapChainSupportDetails getSwapChainSupport() {
        return querySwapChainSupport(_physicalDevice);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    QueueFamilyIndices findPhysicalQueueFamilies() {
        return findQueueFamilies(_physicalDevice);
    }

    VkFormat findSupportedFormat(const std::vector<VkFormat> &candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory);

    VkCommandBuffer beginSingleTimeCommands();

    void endSingleTimeCommands(VkCommandBuffer commandBuffer);

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t layerCount);

    void createImageWithInfo(const VkImageCreateInfo &imageInfo, VkMemoryPropertyFlags properties, VkImage &image, VkDeviceMemory &imageMemory);

    VkPhysicalDeviceProperties properties;
};