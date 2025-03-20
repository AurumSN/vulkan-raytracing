#include "application.h"

#include <stdexcept>
#include <array>
#include <iostream>

Material Material::Rough(LiteMath::float3 color) {
    return {
        .color = LiteMath::float4(color.x, color.y, color.z, 1.0f),
        .reflective = 0.0f,
        .refractiv = 0.0f
    };
}

const Material Material::RoughWhite{
    .color = LiteMath::float4(1.0f, 1.0f, 1.0f, 1.0f),
    .reflective = 0.0f,
    .refractiv = 0.0f
};

const Material Material::PerfectMirror{
    .color = LiteMath::float4(0.0f, 0.0f, 0.0f, 1.0f),
    .reflective = 1.0f,
    .refractiv = 0.0f
};

Primitive Primitive::Plane(LiteMath::float3 normal, LiteMath::float3 position, Material material) {
    return {
        .data = LiteMath::float4(normal.x, normal.y, normal.z, -LiteMath::dot(normal, position)),
        .type = 0,
        .material = material
    };
}

Primitive Primitive::Sphere(LiteMath::float3 position, float radius, Material material) {
    return {
        .data = LiteMath::float4(position.x, position.y, position.z, radius),
        .type = 1,
        .material = material
    };
}

Light Light::Directional(LiteMath::float3 direction, LiteMath::float3 color, float intensity) {
    return {
        .position = LiteMath::float4(-direction.x, -direction.y, -direction.z, 0.0f),
        .color = LiteMath::float4(color.x * intensity, color.y * intensity, color.z * intensity, 1.0f)
    };
}

Light Light::Point(LiteMath::float3 position, LiteMath::float3 color, float intensity) {
    return {
        .position = LiteMath::float4(position.x, position.y, position.z, 1.0f),
        .color = LiteMath::float4(color.x * intensity, color.y * intensity, color.z * intensity, 1.0f)
    };
}

Application::Application() {
    _pixels.resize(WIDTH * HEIGHT);

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = static_cast<uint32_t>(WIDTH);
    imageInfo.extent.height = static_cast<uint32_t>(HEIGHT);
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.flags = 0;

    _device.createImageWithInfo(imageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _image, _imageMemory);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = _device.getCommandPool();
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(_device.device(), &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = _image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(_device.graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(_device.graphicsQueue());

    vkFreeCommandBuffers(_device.device(), _device.getCommandPool(), 1, &commandBuffer);

    createPipelineLayout();
    createPipeline();
    createCommandBuffers();
}

Application::~Application() {
    vkDestroyBuffer(_device.device(), _shaderStorageBufferLights, nullptr);
    vkFreeMemory(_device.device(), _shaderStorageBufferLightsMemory, nullptr);
    vkDestroyBuffer(_device.device(), _shaderStorageBufferBVHs, nullptr);
    vkFreeMemory(_device.device(), _shaderStorageBufferBVHsMemory, nullptr);
    vkDestroyBuffer(_device.device(), _shaderStorageBufferTriangles, nullptr);
    vkFreeMemory(_device.device(), _shaderStorageBufferTrianglesMemory, nullptr);
    vkDestroyBuffer(_device.device(), _shaderStorageBufferMeshes, nullptr);
    vkFreeMemory(_device.device(), _shaderStorageBufferMeshesMemory, nullptr);
    vkDestroyBuffer(_device.device(), _shaderStorageBufferIndices, nullptr);
    vkFreeMemory(_device.device(), _shaderStorageBufferIndicesMemory, nullptr);
    vkDestroyBuffer(_device.device(), _shaderStorageBufferVertices, nullptr);
    vkFreeMemory(_device.device(), _shaderStorageBufferVerticesMemory, nullptr);
    vkDestroyBuffer(_device.device(), _shaderStorageBufferPrimitves, nullptr);
    vkFreeMemory(_device.device(), _shaderStorageBufferPrimitvesMemory, nullptr);
    vkDestroyBuffer(_device.device(), _uniformBuffer, nullptr);
    vkFreeMemory(_device.device(), _uniformBufferMemory, nullptr);
    vkDestroySemaphore(_device.device(), _computeFinishedSemaphore, nullptr);
    vkDestroyFence(_device.device(), _computeInFlightFence, nullptr);
    vkDestroyImageView(_device.device(), _imageView, nullptr);
    vkDestroySampler(_device.device(), _imageSampler, nullptr);
    vkFreeMemory(_device.device(), _imageMemory, nullptr);
    vkDestroyImage(_device.device(), _image, nullptr);
    vkDestroyDescriptorPool(_device.device(), _descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(_device.device(), _computeDescriptorSetLayout, nullptr);
    vkDestroyPipelineLayout(_device.device(), _pipelineLayout, nullptr);
}

#include <chrono>

void Application::run() {
    SDL_Event ev;
    bool keyW = false;
    bool keyS = false;
    bool keyA = false;
    bool keyD = false;
    auto last = std::chrono::system_clock::now();
    int x = 0;
    int y = 0;
    bool skip = true;
    while (!_window.shouldClose()) {
        while (SDL_PollEvent(&ev) != 0) {
            switch (ev.type) {
            case SDL_QUIT:
                _window.stopWindow();
                break;
            case SDL_KEYDOWN:
                switch (ev.key.keysym.sym) {
                case SDLK_w:
                    keyW = true;
                    break;
                case SDLK_s:
                    keyS = true;
                    break;
                case SDLK_a:
                    keyA = true;
                    break;
                case SDLK_d:
                    keyD = true;
                    break;
                case SDLK_ESCAPE:
                    _window.stopWindow();
                    break;
                }
                break;
            case SDL_KEYUP:
                switch (ev.key.keysym.sym) {
                case SDLK_w:
                    keyW = false;
                    break;
                case SDLK_s:
                    keyS = false;
                    break;
                case SDLK_a:
                    keyA = false;
                    break;
                case SDLK_d:
                    keyD = false;
                    break;
                }
                break;
            case SDL_MOUSEMOTION:
                if (!skip) {
                    phi -= (float)(ev.motion.x - x) / WIDTH;
                    theta += (float)(ev.motion.y - y) / HEIGHT;
                }

                x = ev.motion.x;
                y = ev.motion.y;
                skip = false;
                break;
            case SDL_WINDOWEVENT:
                switch (ev.window.event) {
                case SDL_WINDOWEVENT_ENTER:
                    skip = true;
                    break;
                }
                break;
            }
        }
        auto now = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed_seconds = now - last;
        last = now;

        // int tmpX, tmpY;
        // SDL_GetMouseState(&tmpX, &tmpY);
        // phi += (float)(tmpX - x) / WIDTH;
        // x = tmpX;
        // y = tmpY;

        float deltaTime = elapsed_seconds.count();

        if (deltaTime > 0.001f) {
            std::cout << 1.0f / deltaTime << std::endl;
        }

        LiteMath::float3 dir{ 0.0f, 0.0f, 0.0f };
        bool b = false;

        if (keyW) {
            dir.z += 1.0f;
            b = true;
        }
        if (keyS) {
            dir.z -= 1.0f;
            b = true;
        }
        if (keyA) {
            dir.x += 1.0f;
            b = true;
        }
        if (keyD) {
            dir.x -= 1.0f;
            b = true;
        }

        if (b) {
            position += LiteMath::rotate4x4Y(phi) * LiteMath::normalize(dir) * deltaTime;
        }

        drawFrame();
        _window.draw(_pixels.data());
    }

    vkDeviceWaitIdle(_device.device());
}

void Application::createPipelineLayout() {
    std::array<VkDescriptorSetLayoutBinding, 9> layoutBindings{};
    layoutBindings[0].binding = 0;
    layoutBindings[0].descriptorCount = 1;
    layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBindings[0].pImmutableSamplers = nullptr;
    layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    layoutBindings[1].binding = 1;
    layoutBindings[1].descriptorCount = 1;
    layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    layoutBindings[1].pImmutableSamplers = nullptr;
    layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    layoutBindings[2].binding = 2;
    layoutBindings[2].descriptorCount = 1;
    layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings[2].pImmutableSamplers = nullptr;
    layoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    layoutBindings[3].binding = 3;
    layoutBindings[3].descriptorCount = 1;
    layoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings[3].pImmutableSamplers = nullptr;
    layoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    layoutBindings[4].binding = 4;
    layoutBindings[4].descriptorCount = 1;
    layoutBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings[4].pImmutableSamplers = nullptr;
    layoutBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    layoutBindings[5].binding = 5;
    layoutBindings[5].descriptorCount = 1;
    layoutBindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings[5].pImmutableSamplers = nullptr;
    layoutBindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    layoutBindings[6].binding = 6;
    layoutBindings[6].descriptorCount = 1;
    layoutBindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings[6].pImmutableSamplers = nullptr;
    layoutBindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    layoutBindings[7].binding = 7;
    layoutBindings[7].descriptorCount = 1;
    layoutBindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings[7].pImmutableSamplers = nullptr;
    layoutBindings[7].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    layoutBindings[8].binding = 8;
    layoutBindings[8].descriptorCount = 1;
    layoutBindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings[8].pImmutableSamplers = nullptr;
    layoutBindings[8].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // layoutBindings[1].binding = 1;
    // layoutBindings[1].descriptorCount = 1;
    // layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    // layoutBindings[1].pImmutableSamplers = nullptr;
    // layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // layoutBindings[2].binding = 2;
    // layoutBindings[2].descriptorCount = 1;
    // layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    // layoutBindings[2].pImmutableSamplers = nullptr;
    // layoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    layoutInfo.pBindings = layoutBindings.data();

    if (vkCreateDescriptorSetLayout(_device.device(), &layoutInfo, nullptr, &_computeDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create compute descriptor set layout");
    }

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &_computeDescriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    if (vkCreatePipelineLayout(_device.device(), &pipelineLayoutInfo, nullptr, &_pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout");
    }
}

void Application::createPipeline() {
    // auto pipelineConfig = Pipeline::defaultPipelineConfigInfo(_swapChain.width(), _swapChain.height());
    // pipelineConfig.renderPass = _swapChain.getRenderPass();
    // pipelineConfig.pipelineLayout = _pipelineLayout;
    _pipeline = std::make_unique<Pipeline>(
        _device,
        "shaders/compute.comp.spv", 
        _pipelineLayout
    );

    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 1;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[2].descriptorCount = 7;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(_device.device(), &poolInfo, nullptr, &_descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = _descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &_computeDescriptorSetLayout;

    if (vkAllocateDescriptorSets(_device.device(), &allocInfo, &_computeDescriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, _uniformBuffer, _uniformBufferMemory);

    
    vkMapMemory(_device.device(), _uniformBufferMemory, 0, bufferSize, 0, &_uniformBufferMapped);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = _image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(_device.device(), &viewInfo, nullptr, &_imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(_device.physicalDevice(), &properties);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(_device.device(), &samplerInfo, nullptr, &_imageSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }

    // // // Primitve SSBO

    bufferSize = sizeof(Primitive) * _Primitives.size() + sizeof(unsigned int) * 4;

    // Create a staging buffer used to upload data to the gpu
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void* data;
    
    vkMapMemory(_device.device(), stagingBufferMemory, 0, bufferSize, 0, &data);
    *(unsigned *)data = _Primitives.size();
    memcpy((void *)((unsigned *)data + 4), _Primitives.data(), sizeof(Primitive) * _Primitives.size());
    vkUnmapMemory(_device.device(), stagingBufferMemory);

    // Copy initial particle data to all storage buffers
    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _shaderStorageBufferPrimitves, _shaderStorageBufferPrimitvesMemory);
    _device.copyBuffer(stagingBuffer, _shaderStorageBufferPrimitves, bufferSize);

    vkDestroyBuffer(_device.device(), stagingBuffer, nullptr);
    vkFreeMemory(_device.device(), stagingBufferMemory, nullptr);

    std::vector<LiteMath::float4> vertices;
    std::vector<unsigned int> indices;
    std::vector<Mesh> meshes;
    std::vector<unsigned int> triangles;
    std::vector<BVH> bvhs;
    meshes.reserve(_Meshes.size() + 1);

    unsigned int countVert = 0;
    unsigned int countIndx = 0;
    unsigned int countTris = 0;
    for (const auto& m : _Meshes) {
        countVert += m.mesh.VerticesNum();
        countIndx += m.mesh.IndicesNum();
        countTris += m.mesh.TrianglesNum();
    }

    vertices.reserve(countVert);
    indices.reserve(countIndx);
    triangles.reserve(countTris);

    unsigned int offset = 0;
    unsigned int triOffset = 0;

    for (const auto& m : _Meshes) {
        for (const auto& v : m.mesh.vPos4f) {
            vertices.push_back(m.modelMatrix * v);
        }
        for (const auto& i : m.mesh.indices) {
            indices.push_back(i + offset);
        }
        meshes.push_back({
            .begin = (unsigned int)bvhs.size(),
            .material = m.material
        });
        std::vector<unsigned int> t;
        t.reserve(m.mesh.TrianglesNum());
        for (int i = 0; i < m.mesh.TrianglesNum(); i++) {
            t.push_back((triOffset + i) * 3);
        }
        buildBVH(vertices, indices, t.begin(), t.end(), triangles, bvhs);
        offset += m.mesh.VerticesNum();
        triOffset += m.mesh.TrianglesNum();
    }

    updateBVH(bvhs);

    bvhs.push_back({
        .index = triOffset
    });

    // // // Vertex SSBO

    bufferSize = sizeof(LiteMath::float4) * vertices.size();

    // Create a staging buffer used to upload data to the gpu
    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    
    vkMapMemory(_device.device(), stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), bufferSize);
    vkUnmapMemory(_device.device(), stagingBufferMemory);

    // Copy initial particle data to all storage buffers
    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _shaderStorageBufferVertices, _shaderStorageBufferVerticesMemory);
    _device.copyBuffer(stagingBuffer, _shaderStorageBufferVertices, bufferSize);

    vkDestroyBuffer(_device.device(), stagingBuffer, nullptr);
    vkFreeMemory(_device.device(), stagingBufferMemory, nullptr);

    // // // Indices SSBO

    bufferSize = sizeof(unsigned int) * indices.size();
    // Create a staging buffer used to upload data to the gpu
    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    
    vkMapMemory(_device.device(), stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), bufferSize);

    vkUnmapMemory(_device.device(), stagingBufferMemory);

    // Copy initial particle data to all storage buffers
    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _shaderStorageBufferIndices, _shaderStorageBufferIndicesMemory);
    _device.copyBuffer(stagingBuffer, _shaderStorageBufferIndices, bufferSize);

    vkDestroyBuffer(_device.device(), stagingBuffer, nullptr);
    vkFreeMemory(_device.device(), stagingBufferMemory, nullptr);

    // // // Meshes SSBO

    bufferSize = sizeof(Mesh) * meshes.size() + sizeof(unsigned int) * 4;
    // Create a staging buffer used to upload data to the gpu
    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    
    vkMapMemory(_device.device(), stagingBufferMemory, 0, bufferSize, 0, &data);
    *(unsigned *)data = meshes.size();
    memcpy((void *)((unsigned *)data + 4), meshes.data(), sizeof(Mesh) * meshes.size());

    vkUnmapMemory(_device.device(), stagingBufferMemory);

    // Copy initial particle data to all storage buffers
    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _shaderStorageBufferMeshes, _shaderStorageBufferMeshesMemory);
    _device.copyBuffer(stagingBuffer, _shaderStorageBufferMeshes, bufferSize);

    vkDestroyBuffer(_device.device(), stagingBuffer, nullptr);
    vkFreeMemory(_device.device(), stagingBufferMemory, nullptr);

    // // // Triangles SSBO

    bufferSize = sizeof(unsigned int) * triangles.size();
    // Create a staging buffer used to upload data to the gpu
    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    
    vkMapMemory(_device.device(), stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, triangles.data(), bufferSize);

    vkUnmapMemory(_device.device(), stagingBufferMemory);

    // Copy initial particle data to all storage buffers
    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _shaderStorageBufferTriangles, _shaderStorageBufferTrianglesMemory);
    _device.copyBuffer(stagingBuffer, _shaderStorageBufferTriangles, bufferSize);

    vkDestroyBuffer(_device.device(), stagingBuffer, nullptr);
    vkFreeMemory(_device.device(), stagingBufferMemory, nullptr);

    // // // BVHs SSBO

    bufferSize = sizeof(BVH) * bvhs.size();
    // Create a staging buffer used to upload data to the gpu
    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    
    vkMapMemory(_device.device(), stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, bvhs.data(), bufferSize);

    vkUnmapMemory(_device.device(), stagingBufferMemory);

    // Copy initial particle data to all storage buffers
    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _shaderStorageBufferBVHs, _shaderStorageBufferBVHsMemory);
    _device.copyBuffer(stagingBuffer, _shaderStorageBufferBVHs, bufferSize);

    vkDestroyBuffer(_device.device(), stagingBuffer, nullptr);
    vkFreeMemory(_device.device(), stagingBufferMemory, nullptr);

    // // // Lights SSBO

    bufferSize = sizeof(Light) * _Lights.size() + sizeof(unsigned int) * 4;
    // Create a staging buffer used to upload data to the gpu
    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    
    vkMapMemory(_device.device(), stagingBufferMemory, 0, bufferSize, 0, &data);
    *(unsigned *)data = _Lights.size();
    memcpy((void *)((unsigned int *)data + 4), _Lights.data(), bufferSize);

    vkUnmapMemory(_device.device(), stagingBufferMemory);

    // Copy initial particle data to all storage buffers
    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _shaderStorageBufferLights, _shaderStorageBufferLightsMemory);
    _device.copyBuffer(stagingBuffer, _shaderStorageBufferLights, bufferSize);

    vkDestroyBuffer(_device.device(), stagingBuffer, nullptr);
    vkFreeMemory(_device.device(), stagingBufferMemory, nullptr);

    VkDescriptorBufferInfo uniformBufferInfo{};
    uniformBufferInfo.buffer = _uniformBuffer;
    uniformBufferInfo.offset = 0;
    uniformBufferInfo.range = sizeof(UniformBufferObject);

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;// VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = _imageView;
    imageInfo.sampler = _imageSampler;

    VkDescriptorBufferInfo storageBufferPrimitives{};
    storageBufferPrimitives.buffer = _shaderStorageBufferPrimitves;
    storageBufferPrimitives.offset = 0;
    storageBufferPrimitives.range = sizeof(Primitive) * _Primitives.size() + sizeof(unsigned int) * 4;

    VkDescriptorBufferInfo storageBufferVertices{};
    storageBufferVertices.buffer = _shaderStorageBufferVertices;
    storageBufferVertices.offset = 0;
    storageBufferVertices.range = sizeof(LiteMath::float4) * vertices.size();

    VkDescriptorBufferInfo storageBufferIndices{};
    storageBufferIndices.buffer = _shaderStorageBufferIndices;
    storageBufferIndices.offset = 0;
    storageBufferIndices.range = sizeof(unsigned int) * indices.size();

    VkDescriptorBufferInfo storageBufferMeshes{};
    storageBufferMeshes.buffer = _shaderStorageBufferMeshes;
    storageBufferMeshes.offset = 0;
    storageBufferMeshes.range = sizeof(Mesh) * meshes.size() + sizeof(unsigned int) * 4;

    VkDescriptorBufferInfo storageBufferTriangles{};
    storageBufferTriangles.buffer = _shaderStorageBufferTriangles;
    storageBufferTriangles.offset = 0;
    storageBufferTriangles.range = sizeof(unsigned int) * triangles.size();

    VkDescriptorBufferInfo storageBufferBVHs{};
    storageBufferBVHs.buffer = _shaderStorageBufferBVHs;
    storageBufferBVHs.offset = 0;
    storageBufferBVHs.range = sizeof(BVH) * bvhs.size();

    VkDescriptorBufferInfo storageBufferLights{};
    storageBufferLights.buffer = _shaderStorageBufferLights;
    storageBufferLights.offset = 0;
    storageBufferLights.range = sizeof(Light) * _Lights.size() + sizeof(unsigned int) * 4;

    std::array<VkWriteDescriptorSet, 9> descriptorWrites{};
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = _computeDescriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &uniformBufferInfo;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = _computeDescriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pImageInfo = &imageInfo;

    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = _computeDescriptorSet;
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].dstArrayElement = 0;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &storageBufferPrimitives;

    descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[3].dstSet = _computeDescriptorSet;
    descriptorWrites[3].dstBinding = 3;
    descriptorWrites[3].dstArrayElement = 0;
    descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[3].descriptorCount = 1;
    descriptorWrites[3].pBufferInfo = &storageBufferVertices;

    descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[4].dstSet = _computeDescriptorSet;
    descriptorWrites[4].dstBinding = 4;
    descriptorWrites[4].dstArrayElement = 0;
    descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[4].descriptorCount = 1;
    descriptorWrites[4].pBufferInfo = &storageBufferIndices;

    descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[5].dstSet = _computeDescriptorSet;
    descriptorWrites[5].dstBinding = 5;
    descriptorWrites[5].dstArrayElement = 0;
    descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[5].descriptorCount = 1;
    descriptorWrites[5].pBufferInfo = &storageBufferMeshes;

    descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[6].dstSet = _computeDescriptorSet;
    descriptorWrites[6].dstBinding = 6;
    descriptorWrites[6].dstArrayElement = 0;
    descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[6].descriptorCount = 1;
    descriptorWrites[6].pBufferInfo = &storageBufferTriangles;

    descriptorWrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[7].dstSet = _computeDescriptorSet;
    descriptorWrites[7].dstBinding = 7;
    descriptorWrites[7].dstArrayElement = 0;
    descriptorWrites[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[7].descriptorCount = 1;
    descriptorWrites[7].pBufferInfo = &storageBufferBVHs;

    descriptorWrites[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[8].dstSet = _computeDescriptorSet;
    descriptorWrites[8].dstBinding = 8;
    descriptorWrites[8].dstArrayElement = 0;
    descriptorWrites[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[8].descriptorCount = 1;
    descriptorWrites[8].pBufferInfo = &storageBufferLights;

    vkUpdateDescriptorSets(_device.device(), descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateSemaphore(_device.device(), &semaphoreInfo, nullptr, &_computeFinishedSemaphore) != VK_SUCCESS ||
        vkCreateFence(_device.device(), &fenceInfo, nullptr, &_computeInFlightFence) != VK_SUCCESS) {
        throw std::runtime_error("failed to create compute synchronization objects for a frame!");
    }
}

void Application::createCommandBuffers() {
    // _commandBuffers.resize(_swapChain.imageCount());

    // VkCommandBufferAllocateInfo allocInfo{};
    // allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    // allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    // allocInfo.commandPool = _device.getCommandPool();
    // allocInfo.commandBufferCount = static_cast<uint32_t>(_commandBuffers.size());

    // if (vkAllocateCommandBuffers(_device.device(), &allocInfo, _commandBuffers.data()) != VK_SUCCESS) {
    //     throw std::runtime_error("failed to allocate command buffers");
    // }

    // for (int i = 0; i < _commandBuffers.size(); i++) {
    //     VkCommandBufferBeginInfo beginInfo{};
    //     beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    //     if (vkBeginCommandBuffer(_commandBuffers[i], &beginInfo) != VK_SUCCESS) {
    //         throw std::runtime_error("failed to begin recording command buffer");
    //     }

    //     VkRenderPassBeginInfo renderPassInfo{};
    //     renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    //     renderPassInfo.renderPass = _swapChain.getRenderPass();
    //     renderPassInfo.framebuffer = _swapChain.getFrameBuffer(i);

    //     renderPassInfo.renderArea.offset = { 0, 0 };
    //     renderPassInfo.renderArea.extent = _swapChain.getSwapChainExtent();

    //     std::array<VkClearValue, 2> clearValues{};
    //     clearValues[0].color = { 0.1f, 0.1f, 0.1f, 1.0f };
    //     clearValues[1].depthStencil = { 1.0f, 0 };

    //     renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    //     renderPassInfo.pClearValues = clearValues.data();

    //     vkCmdBeginRenderPass(_commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    //     _pipeline->bind(_commandBuffers[i]);
    //     vkCmdDraw(_commandBuffers[i], 3, 1, 0, 0];

    //     vkCmdEndRenderPass(_commandBuffers[i]);

    //     if (vkEndCommandBuffer(_commandBuffers[i]) != VK_SUCCESS) {
    //         throw std::runtime_error("failed to record command buffer");
    //     }
    // }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = _device.getCommandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(_device.device(), &allocInfo, &_commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate compute command buffers!");
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    vkWaitForFences(_device.device(), 1, &_computeInFlightFence, VK_TRUE, UINT64_MAX);

    UniformBufferObject ubo{};
    ubo.width = WIDTH;
    ubo.height = HEIGHT;
    ubo.proj = LiteMath::inverse4x4(LiteMath::perspectiveMatrix(90.0f, (float)WIDTH / HEIGHT, 0.01f, 100.0f));

    memcpy(_uniformBufferMapped, &ubo, sizeof(ubo));

    vkResetFences(_device.device(), 1, &_computeInFlightFence);

    if (vkBeginCommandBuffer(_commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    _pipeline->bind(_commandBuffer);

    vkCmdBindDescriptorSets(_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1, &_computeDescriptorSet, 0, 0);

    vkCmdDispatch(_commandBuffer, WIDTH, HEIGHT, 1);

    if (vkEndCommandBuffer(_commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &_commandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &_computeFinishedSemaphore;

    if (vkQueueSubmit(_device.graphicsQueue(), 1, &submitInfo, _computeInFlightFence) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }
}

#include <iostream>

void Application::drawFrame() {
    
    // uint32_t imageIndex;
    // auto result  = _swapChain.acquireNextImage(&imageIndex);

    // if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    //     throw std::runtime_error("failed to acquire swap chain image");
    // }

    // result = _swapChain.submitCommandBuffers(&_commandBuffers[imageIndex], &imageIndex);

    // if (result != VK_SUCCESS) {
    //     throw std::runtime_error("failed to present swap chain image");
    // }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    

    // vkWaitForFences(_device.device(), 1, &_computeInFlightFence, VK_TRUE, UINT64_MAX);
    

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    

    VkDeviceSize bufferSize = WIDTH * HEIGHT * 4;
    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
    VkCommandBuffer commandBuffer = _device.beginSingleTimeCommands();
    

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {0, 0, 0};
    region.imageExtent = {WIDTH, HEIGHT, 1};

    vkCmdCopyImageToBuffer(commandBuffer, _image, VK_IMAGE_LAYOUT_GENERAL, stagingBuffer, 1, &region);

    _device.endSingleTimeCommands(commandBuffer);

    // 
    void *data;
    vkMapMemory(_device.device(), stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(_pixels.data(), data, WIDTH * HEIGHT * 4);

    vkUnmapMemory(_device.device(), stagingBufferMemory);
    

    // Copy initial particle data to all storage buffers

    vkDestroyBuffer(_device.device(), stagingBuffer, nullptr);
    vkFreeMemory(_device.device(), stagingBufferMemory, nullptr);

    // void *data;
    // 
    // vkMapMemory(_device.device(), _imageMemory, 0, WIDTH * HEIGHT * 4, 0, &data);
    // 

    // memcpy(_pixels.data(), data, WIDTH * HEIGHT * 4);

    // vkUnmapMemory(_device.device(), _imageMemory);

    UniformBufferObject ubo{};
    ubo.width = WIDTH;
    ubo.height = HEIGHT;
    //ubo.view = LiteMath::inverse4x4(LiteMath::lookAt(LiteMath::float3(cosf(phi), 0.0f, sinf(phi)), LiteMath::float3(0.0f, 0.0f, 0.0f), LiteMath::float3(0.0f, 1.0f, 0.0f)));
        
    
    LiteMath::float3 lookAt{ 0.0f, 0.0f, 1.0f };
    lookAt = LiteMath::rotate4x4Y(phi) * LiteMath::rotate4x4X(theta) * lookAt;
    ubo.view = LiteMath::inverse4x4(LiteMath::lookAt(position, position + lookAt, LiteMath::float3(0.0f, 1.0f, 0.0f)));
    ubo.proj = LiteMath::inverse4x4(LiteMath::perspectiveMatrix(90.0f, (float)WIDTH / HEIGHT, 0.01f, 100.0f));
    

    memcpy(_uniformBufferMapped, &ubo, sizeof(ubo));

    vkResetFences(_device.device(), 1, &_computeInFlightFence);
    

    vkResetCommandBuffer(_commandBuffer, 0);
    
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(_commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording compute command buffer!");
    }
    

    _pipeline->bind(_commandBuffer);

    vkCmdBindDescriptorSets(_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1, &_computeDescriptorSet, 0, nullptr);
    

    vkCmdDispatch(_commandBuffer, WIDTH, HEIGHT, 1);

    if (vkEndCommandBuffer(_commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record compute command buffer!");
    }
    

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &_computeFinishedSemaphore;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &_commandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &_computeFinishedSemaphore;

    if (vkQueueSubmit(_device.graphicsQueue(), 1, &submitInfo, _computeInFlightFence) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }
    
}

#include <algorithm>

class Sorter {
    const std::vector<LiteMath::float4>& v;
    const std::vector<unsigned int>& i;
    const int dim;

public:
    Sorter(const std::vector<LiteMath::float4>& v, const std::vector<unsigned int>& i, int dim) : v{ v }, i{ i }, dim{ dim } {}

    bool operator()(unsigned int a, unsigned int b) {
        return std::max(std::max(v[i[a + 0]][dim], v[i[a + 1]][dim]), v[i[a + 2]][dim]) < std::max(std::max(v[i[b + 0]][dim], v[i[b + 1]][dim]), v[i[b + 2]][dim]);
    }
};

struct Box {
    LiteMath::float4 min;
    LiteMath::float4 max;

    Box(LiteMath::float4 min, LiteMath::float4 max) : min{min}, max{max} {}
    Box(LiteMath::float4 point) : min{point}, max{point} {}

    void include(LiteMath::float4 point) {
        for (int k = 0; k < 3; k++) {
            if (min[k] > point[k]) {
                min[k] = point[k];
            }
            if (max[k] < point[k]) {
                max[k] = point[k];
            }
        }
    }

    float surfaceArea() {
        return 2 * ((max.x - min.x) * ((max.y - min.y) + (max.z - min.z)) + (max.y - min.y) * (max.z - min.z));
    }
};

void Application::buildBVH(const std::vector<LiteMath::float4> &vertices, const std::vector<unsigned int> &indices,
    std::vector<unsigned int>::iterator triangles_begin, std::vector<unsigned int>::iterator triangles_end, std::vector<unsigned int> &triangleList,
    std::vector<BVH> &bvhs) {

    if (triangles_end == triangles_begin) {
        return;
    }

    BVH bvh{
        .index = (unsigned int)triangleList.size(),
        .escape = 0,
        .left = 0,
        .right = 0,
        .AABBmin = vertices[indices[*triangles_begin]],
        .AABBmax = vertices[indices[*triangles_begin]]
    };

    for (auto i = triangles_begin; i != triangles_end; ++i) {
        LiteMath::float4 pts[3]{vertices[indices[*i + 0]], vertices[indices[*i + 1]], vertices[indices[*i + 2]]};

        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                if (bvh.AABBmin[k] > pts[j][k]) {
                    bvh.AABBmin[k] = pts[j][k];
                }
                if (bvh.AABBmax[k] < pts[j][k]) {
                    bvh.AABBmax[k] = pts[j][k];
                }
            }
        }
    }

    int minDim = -1;
    float minSAH = 1.0f*Box(bvh.AABBmin, bvh.AABBmax).surfaceArea()*(triangles_end - triangles_begin);
    std::vector<unsigned int>::iterator leftEnd;

    for (int dim = 0; dim < 3; dim++) {
        std::sort(triangles_begin, triangles_end, Sorter(vertices, indices, dim));

        std::vector<Box> left;
        Box cur{vertices[indices[*triangles_begin]]};
        left.reserve(triangles_end - triangles_begin);
        for (auto i = triangles_begin; i != triangles_end; ++i) {
            cur.include(vertices[indices[*i + 0]]);
            cur.include(vertices[indices[*i + 1]]);
            cur.include(vertices[indices[*i + 2]]);
            left.push_back(cur);
        }

        cur = Box{vertices[indices[*(triangles_end - 1)]]};
        for (auto i = triangles_end - 1; i >= triangles_begin; --i) {
            cur.include(vertices[indices[*i + 0]]);
            cur.include(vertices[indices[*i + 1]]);
            cur.include(vertices[indices[*i + 2]]);
            float sah = 0.0f + cur.surfaceArea() * (triangles_end - i) + left[i - triangles_begin].surfaceArea() * (i - triangles_begin + 1);
            if (sah < minSAH) {
                minDim = dim;
                minSAH = sah;
                leftEnd = i;
            }
        }
    }

    if (minDim == -1) {
        triangleList.insert(triangleList.end(), triangles_begin, triangles_end);
        bvhs.push_back(bvh);
    } else {
        std::sort(triangles_begin, triangles_end, Sorter(vertices, indices, minDim));

        int index = bvhs.size() + 1;
        bvhs.push_back(bvh);
        bvhs[index - 1].left = bvhs.size() + 1;
        buildBVH(vertices, indices, triangles_begin, leftEnd, triangleList, bvhs);
        bvhs[index - 1].right = bvhs.size() + 1;
        buildBVH(vertices, indices, leftEnd, triangles_end, triangleList, bvhs);
    }
}

void Application::updateBVH(std::vector<BVH> &bvhs) {
    for (const auto& bvh : bvhs) {
        if (bvh.left != 0) {
            bvhs[bvh.left - 1].escape = bvh.right;
            bvhs[bvh.right - 1].escape = bvh.escape;
        }
    }
}
