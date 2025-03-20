#include "pipeline.h"

#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cassert>

std::vector<char> Pipeline::readFile(const std::string &filepath) {
    std::ifstream file{ filepath, std::ios::ate | std::ios::binary };

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filepath);
    }

    size_t fileSize = static_cast<size_t>(file.tellg());

    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

void Pipeline::bind(VkCommandBuffer commandBuffer) {
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _computePipeline);
}

void Pipeline::createComputePipeline(const std::string &compFilepath, VkPipelineLayout layout) {
    auto compCode = readFile(compFilepath);

    createShaderModule(compCode, &_compShaderModule);

    VkPipelineShaderStageCreateInfo shaderStage{};
    shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStage.module = _compShaderModule;
    shaderStage.pName = "main";
    shaderStage.flags = 0;
    shaderStage.pNext = nullptr;
    shaderStage.pSpecializationInfo = nullptr;

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStage;
    pipelineInfo.layout = layout;
    pipelineInfo.flags = 0;
    pipelineInfo.pNext = nullptr;

    pipelineInfo.basePipelineIndex = -1;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    if (vkCreateComputePipelines(_device.device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &_computePipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create compute pipeline");
    }
}

void Pipeline::createShaderModule(const std::vector<char> &code, VkShaderModule *shaderModule) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    if (vkCreateShaderModule(_device.device(), &createInfo, nullptr, shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module");
    }
}

Pipeline::Pipeline(Device& device, const std::string& compFilepath, VkPipelineLayout layout) : _device{ device } {
    createComputePipeline(compFilepath, layout);
}

Pipeline::~Pipeline() {
    vkDestroyShaderModule(_device.device(), _compShaderModule, nullptr);
    vkDestroyPipeline(_device.device(), _computePipeline, nullptr);
}