#pragma once

#include "device.h"

#include <string>
#include <vector>

class Pipeline {
    static std::vector<char> readFile(const std::string& filepath);
    
    Device &_device;
    VkPipeline _computePipeline;
    VkShaderModule _compShaderModule;

    Pipeline(const Pipeline&) = delete;
    void operator=(const Pipeline&) = delete;

    void createComputePipeline(const std::string& compFilepath, VkPipelineLayout layout);

    void createShaderModule(const std::vector<char>& code, VkShaderModule* shaderModule);
    
public:
    Pipeline(Device& device, const std::string& compFilepath, VkPipelineLayout layout);
    ~Pipeline();

    void bind(VkCommandBuffer commandBuffer);
};