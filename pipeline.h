#pragma once

#include "device.h"

#include <string>
#include <vector>

struct PipelineConfigInfo {
    VkViewport viewport;
    VkRect2D scissor;
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo;
    VkPipelineRasterizationStateCreateInfo rasterizationInfo;
    VkPipelineMultisampleStateCreateInfo multisampleInfo;
    VkPipelineColorBlendAttachmentState colorBlendAttachment;
    VkPipelineColorBlendStateCreateInfo colorBlendInfo;
    VkPipelineDepthStencilStateCreateInfo depthStencilInfo;
    VkPipelineLayout pipelineLayout = nullptr;
    VkRenderPass renderPass = nullptr;
    uint32_t subpass = 0;
};

class Pipeline {
    static std::vector<char> readFile(const std::string& filepath);
    
    Device &_device;
    VkPipeline _graphicsPipeline;
    VkShaderModule _vertShaderModule;
    VkShaderModule _fragShaderModule;
    VkPipeline _computePipeline;
    VkShaderModule _compShaderModule;

    Pipeline(const Pipeline&) = delete;
    void operator=(const Pipeline&) = delete;

    void createGraphicsPipeline(const std::string& vertFilepath, const std::string& fragFilepath, const PipelineConfigInfo& configInfo);

    void createComputePipeline(const std::string& compFilepath, VkPipelineLayout layout);

    void createShaderModule(const std::vector<char>& code, VkShaderModule* shaderModule);
    
public:
    static PipelineConfigInfo defaultPipelineConfigInfo(uint32_t width, uint32_t height);

    Pipeline(Device& device, const std::string& vertFilepath, const std::string& fragFilepath, const PipelineConfigInfo &configInfo, const std::string& compFilepath, VkPipelineLayout computeLayout);
    ~Pipeline();

    void bindGraphics(VkCommandBuffer commandBuffer);
    void bindCompute(VkCommandBuffer commandBuffer);
};