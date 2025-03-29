#include "application.h"

#include <stdexcept>
#include <array>
#include <iostream>
#include <chrono>

#include "grid.h"

static size_t node_count = 0;

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

    Box() {}
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

void buildBVH(const std::vector<LiteMath::float4> &vertices, const std::vector<unsigned int> &indices,
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
    float minSAH = 1.0f * Box(bvh.AABBmin, bvh.AABBmax).surfaceArea() * (triangles_end - triangles_begin);
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
        for (auto i = triangles_end - 1; i >= triangles_begin + 1; --i) {
            cur.include(vertices[indices[*i + 0]]);
            cur.include(vertices[indices[*i + 1]]);
            cur.include(vertices[indices[*i + 2]]);
            float sah = 0.1f + cur.surfaceArea() * (triangles_end - i) + left[i - triangles_begin - 1].surfaceArea() * (i - triangles_begin);
            if (sah < minSAH) {
                minDim = dim;
                minSAH = sah;
                leftEnd = i;
            }
        }
    }
    node_count++;
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

void updateBVH(std::vector<BVH> &bvhs) {
    for (const auto& bvh : bvhs) {
        if (bvh.left != 0) {
            bvhs[bvh.left - 1].escape = bvh.right;
            bvhs[bvh.right - 1].escape = bvh.escape;
        }
    }
}


Application::Application(bool isSpheretracing) : Application{ { Light::Directional(LiteMath::float3(-1.0f, -1.0f, -1.0f)) }, isSpheretracing } {}
Application::Application(std::vector<PreMesh> meshes) : Application{ { }, meshes } {}
Application::Application(std::vector<Grid> grids) : Application{ { }, grids } {}
Application::Application(std::vector<Octree> octrees) : Application{ { }, octrees } {}
Application::Application(std::vector<Light> lights, bool isSphereTracing) : Application{ { Primitive{ .data = LiteMath::float4(0.0f, 0.0f, 0.0f, 1.0f), .type = 1 } }, lights, isSphereTracing } {}
Application::Application(std::vector<PreMesh> meshes, std::vector<Light> lights) : Application{ { }, meshes, lights } {}
Application::Application(std::vector<Grid> grids, std::vector<Light> lights) : Application{ { }, grids, lights } {}
Application::Application(std::vector<Octree> octrees, std::vector<Light> lights) : Application{ { }, octrees, lights } {}
Application::Application(std::vector<Primitive> primitives, bool isSpheretracing) : Application{ primitives, { Light::Directional(LiteMath::float3(-1.0f, -1.0f, -1.0f)) }, isSpheretracing } {}
Application::Application(std::vector<Primitive> primitives, std::vector<PreMesh> meshes) : Application{ primitives, meshes, { Light::Directional(LiteMath::float3(-1.0f, -1.0f, -1.0f)) } } {}
Application::Application(std::vector<Primitive> primitives, std::vector<Grid> grids) : Application{ primitives, grids, { Light::Directional(LiteMath::float3(-1.0f, -1.0f, -1.0f)) } } {}
Application::Application(std::vector<Primitive> primitives, std::vector<Octree> octrees) : Application{ primitives, octrees, { Light::Directional(LiteMath::float3(-1.0f, -1.0f, -1.0f)) } } {}

Application::Application(std::vector<Primitive> primitives, std::vector<Light> lights, bool isSphereTracing) : _Primitives{ primitives }, _Lights{ lights }, _isSphereTracing{ isSphereTracing } {
    _pixels.resize(WIDTH * HEIGHT);

    if (_isSphereTracing) {
        createSpheretracingPipelineLayout();
        createSpheretracingPipeline();
    } else {
        createRaytracingPipelineLayout();
        createRaytracingPipeline();
    }

    createCommandBuffers();
}

Application::Application(std::vector<Primitive> primitives, std::vector<PreMesh> meshes, std::vector<Light> lights) : _Primitives{ primitives }, _Meshes{ meshes }, _Lights{ lights }, _isSphereTracing{ false } {
    _pixels.resize(WIDTH * HEIGHT);

    createRaytracingPipelineLayout();
    createRaytracingPipeline();

    createCommandBuffers();
}
Application::Application(std::vector<Primitive> primitives, std::vector<Grid> grids, std::vector<Light> lights) : Application{ primitives, grids, { }, lights } {}
Application::Application(std::vector<Primitive> primitives, std::vector<Octree> octrees, std::vector<Light> lights) : Application{ primitives, { }, octrees, lights } {}

Application::Application(std::vector<Primitive> primitives, std::vector<Grid> grids, std::vector<Octree> octrees, std::vector<Light> lights) : _Primitives{ primitives }, _Grids{ grids }, _Octrees{ octrees }, _Lights{ lights }, _isSphereTracing{ true } {
    _pixels.resize(WIDTH * HEIGHT);

    createSpheretracingPipelineLayout();
    createSpheretracingPipeline();

    createCommandBuffers();
}

Application::~Application() {
    vkDestroyBuffer(_device.device(), _shaderStorageBufferLights, nullptr);
    vkFreeMemory(_device.device(), _shaderStorageBufferLightsMemory, nullptr);

    if (_isSphereTracing) {
        vkDestroyBuffer(_device.device(), _shaderStorageBufferGrids, nullptr);
        vkFreeMemory(_device.device(), _shaderStorageBufferGridsMemory, nullptr);
        vkDestroyBuffer(_device.device(), _shaderStorageBufferGridDistances, nullptr);
        vkFreeMemory(_device.device(), _shaderStorageBufferGridDistancesMemory, nullptr);
        vkDestroyBuffer(_device.device(), _shaderStorageBufferOctreeNodes, nullptr);
        vkFreeMemory(_device.device(), _shaderStorageBufferOctreeNodesMemory, nullptr);
        vkDestroyBuffer(_device.device(), _shaderStorageBufferOctrees, nullptr);
        vkFreeMemory(_device.device(), _shaderStorageBufferOctreesMemory, nullptr);
    } else {
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
    }

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

void Application::run() {
    SDL_Event ev;
    bool keyW = false;
    bool keyS = false;
    bool keyA = false;
    bool keyD = false;
    bool keyShift = false;
    bool keySpace = false;
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
                case SDLK_LSHIFT:
                    keyShift = true;
                    break;
                case SDLK_SPACE:
                    keySpace = true;
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
                case SDLK_LSHIFT:
                    keyShift = false;
                    break;
                case SDLK_SPACE:
                    keySpace = false;
                    break;
                }
                break;
            case SDL_MOUSEMOTION:
                if (!skip) {
                    phi += (float)(ev.motion.x - x) / WIDTH;
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

        float deltaTime = elapsed_seconds.count();

        if (deltaTime > 0.001f) {
            std::cout << 1.0f / deltaTime << std::endl;
        }

        LiteMath::float3 dir{ 0.0f, 0.0f, 0.0f };

        if (keyW) {
            dir.z += 1.0f;
        }
        if (keyS) {
            dir.z -= 1.0f;
        }
        if (keyD) {
            dir.x += 1.0f;
        }
        if (keyA) {
            dir.x -= 1.0f;
        }
        if (keySpace) {
            dir.y += 1.0f;
        }
        if (keyShift) {
            dir.y -= 1.0f;
        }

        if (LiteMath::length(dir) > 0.1f) {
            position += LiteMath::rotate4x4Y(phi) * LiteMath::normalize(dir) * deltaTime;
        }

        drawFrame();
        _window.draw(_pixels.data());
    }

    vkDeviceWaitIdle(_device.device());
}

VkDescriptorImageInfo Application::createOutputImageBuffer() {
    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.extent.width = static_cast<uint32_t>(WIDTH);
    imageCreateInfo.extent.height = static_cast<uint32_t>(HEIGHT);
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.flags = 0;

    _device.createImageWithInfo(imageCreateInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _image, _imageMemory);

    VkCommandBuffer commandBuffer = _device.beginSingleTimeCommands();

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

    _device.endSingleTimeCommands(commandBuffer);

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

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;// VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = _imageView;
    imageInfo.sampler = _imageSampler;

    return imageInfo;
}

VkDescriptorBufferInfo Application::createUniformBuffer() {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, _uniformBuffer, _uniformBufferMemory);

    vkMapMemory(_device.device(), _uniformBufferMemory, 0, bufferSize, 0, &_uniformBufferMapped);

    VkDescriptorBufferInfo uniformBufferInfo{};
    uniformBufferInfo.buffer = _uniformBuffer;
    uniformBufferInfo.offset = 0;
    uniformBufferInfo.range = sizeof(UniformBufferObject);

    return uniformBufferInfo;
}

template<typename T>
VkDescriptorBufferInfo Application::createBuffer(const std::vector<T>& data, VkBuffer& buffer, VkDeviceMemory& memory, bool withOffset) {
    if (data.size() == 0) {
        _device.createBuffer(1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, memory);

        VkDescriptorBufferInfo info{};
        info.buffer = buffer;
        info.offset = 0;
        info.range = 1;

        return info;
    }

    VkDeviceSize bufferSize = sizeof(T) * data.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void *mapped;

    vkMapMemory(_device.device(), stagingBufferMemory, 0, bufferSize, 0, &mapped);
    memcpy(mapped, data.data(), bufferSize);
    vkUnmapMemory(_device.device(), stagingBufferMemory);

    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, memory);
    _device.copyBuffer(stagingBuffer, buffer, bufferSize);

    vkDestroyBuffer(_device.device(), stagingBuffer, nullptr);
    vkFreeMemory(_device.device(), stagingBufferMemory, nullptr);

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = bufferSize;

    return bufferInfo;
}

template<typename T>
VkDescriptorBufferInfo Application::createCountedBuffer(const std::vector<T>& data, VkBuffer& buffer, VkDeviceMemory& memory, bool withOffset) {
    VkDeviceSize bufferSize = sizeof(T) * data.size() + sizeof(unsigned) * (withOffset ? 4 : 1);

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void* mapped;
    
    vkMapMemory(_device.device(), stagingBufferMemory, 0, bufferSize, 0, &mapped);
    *(unsigned *)mapped = data.size();
    memcpy((void *)((unsigned *)mapped + (withOffset ? 4 : 1)), data.data(), sizeof(T) * data.size());
    vkUnmapMemory(_device.device(), stagingBufferMemory);

    _device.createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, memory);
    _device.copyBuffer(stagingBuffer, buffer, bufferSize);

    vkDestroyBuffer(_device.device(), stagingBuffer, nullptr);
    vkFreeMemory(_device.device(), stagingBufferMemory, nullptr);

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = bufferSize;

    return bufferInfo;
}

void Application::createRaytracingPipelineLayout() {
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

void Application::createRaytracingPipeline() {
    _pipeline = std::make_unique<Pipeline>(
        _device,
        "shaders/raytracing.comp.spv", 
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

    std::array<VkWriteDescriptorSet, 9> descriptorWrites{};

    VkDescriptorBufferInfo uniformBufferInfo = createUniformBuffer();
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = _computeDescriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &uniformBufferInfo;

    VkDescriptorImageInfo imageInfo = createOutputImageBuffer();
    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = _computeDescriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pImageInfo = &imageInfo;

    VkDescriptorBufferInfo storageBufferPrimitives = createCountedBuffer<Primitive>(_Primitives, _shaderStorageBufferPrimitves, _shaderStorageBufferPrimitvesMemory);
    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = _computeDescriptorSet;
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].dstArrayElement = 0;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &storageBufferPrimitives;

    VkDescriptorBufferInfo storageBufferVertices = createBuffer<LiteMath::float4>(vertices, _shaderStorageBufferVertices, _shaderStorageBufferVerticesMemory);
    descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[3].dstSet = _computeDescriptorSet;
    descriptorWrites[3].dstBinding = 3;
    descriptorWrites[3].dstArrayElement = 0;
    descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[3].descriptorCount = 1;
    descriptorWrites[3].pBufferInfo = &storageBufferVertices;

    VkDescriptorBufferInfo storageBufferIndices = createBuffer<unsigned>(indices, _shaderStorageBufferIndices, _shaderStorageBufferIndicesMemory, false);
    descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[4].dstSet = _computeDescriptorSet;
    descriptorWrites[4].dstBinding = 4;
    descriptorWrites[4].dstArrayElement = 0;
    descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[4].descriptorCount = 1;
    descriptorWrites[4].pBufferInfo = &storageBufferIndices;

    VkDescriptorBufferInfo storageBufferMeshes = createCountedBuffer<Mesh>(meshes, _shaderStorageBufferMeshes, _shaderStorageBufferMeshesMemory);
    descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[5].dstSet = _computeDescriptorSet;
    descriptorWrites[5].dstBinding = 5;
    descriptorWrites[5].dstArrayElement = 0;
    descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[5].descriptorCount = 1;
    descriptorWrites[5].pBufferInfo = &storageBufferMeshes;

    VkDescriptorBufferInfo storageBufferTriangles = createBuffer<unsigned>(triangles, _shaderStorageBufferTriangles, _shaderStorageBufferTrianglesMemory, false);
    descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[6].dstSet = _computeDescriptorSet;
    descriptorWrites[6].dstBinding = 6;
    descriptorWrites[6].dstArrayElement = 0;
    descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[6].descriptorCount = 1;
    descriptorWrites[6].pBufferInfo = &storageBufferTriangles;

    VkDescriptorBufferInfo storageBufferBVHs = createBuffer<BVH>(bvhs, _shaderStorageBufferBVHs, _shaderStorageBufferBVHsMemory);
    descriptorWrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[7].dstSet = _computeDescriptorSet;
    descriptorWrites[7].dstBinding = 7;
    descriptorWrites[7].dstArrayElement = 0;
    descriptorWrites[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[7].descriptorCount = 1;
    descriptorWrites[7].pBufferInfo = &storageBufferBVHs;

    VkDescriptorBufferInfo storageBufferLights = createCountedBuffer<Light>(_Lights, _shaderStorageBufferLights, _shaderStorageBufferLightsMemory);
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

void Application::createSpheretracingPipelineLayout() {
    std::array<VkDescriptorSetLayoutBinding, 8> layoutBindings{};
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

void Application::createSpheretracingPipeline() {
    _pipeline = std::make_unique<Pipeline>(
        _device,
        "shaders/spheretracing.comp.spv", 
        _pipelineLayout
    );

    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 1;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[2].descriptorCount = 6;

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

    std::array<VkWriteDescriptorSet, 6> descriptorWrites{};

    VkDescriptorBufferInfo uniformBufferInfo = createUniformBuffer();
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = _computeDescriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &uniformBufferInfo;

    VkDescriptorImageInfo imageInfo = createOutputImageBuffer();
    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = _computeDescriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pImageInfo = &imageInfo;

    VkDescriptorBufferInfo storageBufferPrimitives = createCountedBuffer<Primitive>(_Primitives, _shaderStorageBufferPrimitves, _shaderStorageBufferPrimitvesMemory);
    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = _computeDescriptorSet;
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].dstArrayElement = 0;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &storageBufferPrimitives;

    VkDescriptorBufferInfo storageBufferLights = createCountedBuffer<Light>(_Lights, _shaderStorageBufferLights, _shaderStorageBufferLightsMemory);
    descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[3].dstSet = _computeDescriptorSet;
    descriptorWrites[3].dstBinding = 3;
    descriptorWrites[3].dstArrayElement = 0;
    descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[3].descriptorCount = 1;
    descriptorWrites[3].pBufferInfo = &storageBufferLights;

    std::vector<float> gridDistances;
    std::vector<GridData> grids;
    grids.reserve(_Grids.size());
    for (const auto& i : _Grids) {
        SdfGrid grid;
        load_sdf_grid(grid, i.name);
        grids.push_back({
            .index = static_cast<unsigned>(gridDistances.size()),
            .size = grid.size,
            .material = i.material,
            .inverseModel = i.inverseModel
        });
        gridDistances.insert(gridDistances.end(), grid.data.begin(), grid.data.end());
    }

    VkDescriptorBufferInfo storageBufferGridDistances = createBuffer<float>(gridDistances, _shaderStorageBufferGridDistances, _shaderStorageBufferGridDistancesMemory);
    descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[4].dstSet = _computeDescriptorSet;
    descriptorWrites[4].dstBinding = 4;
    descriptorWrites[4].dstArrayElement = 0;
    descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[4].descriptorCount = 1;
    descriptorWrites[4].pBufferInfo = &storageBufferGridDistances;

    VkDescriptorBufferInfo storageBufferGrids = createCountedBuffer<GridData>(grids, _shaderStorageBufferGrids, _shaderStorageBufferGridsMemory);
    descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[5].dstSet = _computeDescriptorSet;
    descriptorWrites[5].dstBinding = 5;
    descriptorWrites[5].dstArrayElement = 0;
    descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[5].descriptorCount = 1;
    descriptorWrites[5].pBufferInfo = &storageBufferGrids;

    std::vector<SdfOctreeNode> nodes;
    std::vector<OctreeData> octrees;
    octrees.reserve(_Octrees.size());
    for (const auto& i : _Octrees) {
        SdfOctree octree;
        load_sdf_octree(octree, i.name);
        octrees.push_back({
            .index = static_cast<unsigned>(nodes.size()),
            .material = i.material,
            .inverseModel = i.inverseModel
        });
        nodes.insert(nodes.end(), octree.nodes.begin(), octree.nodes.end());
    }

    VkDescriptorBufferInfo storageBufferOctreeNodes = createBuffer<SdfOctreeNode>(nodes, _shaderStorageBufferOctreeNodes, _shaderStorageBufferOctreeNodesMemory);
    descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[6].dstSet = _computeDescriptorSet;
    descriptorWrites[6].dstBinding = 6;
    descriptorWrites[6].dstArrayElement = 0;
    descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[6].descriptorCount = 1;
    descriptorWrites[6].pBufferInfo = &storageBufferOctreeNodes;

    VkDescriptorBufferInfo storageBufferOctrees = createCountedBuffer<OctreeData>(octrees, _shaderStorageBufferOctrees, _shaderStorageBufferOctreesMemory);
    descriptorWrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[7].dstSet = _computeDescriptorSet;
    descriptorWrites[7].dstBinding = 7;
    descriptorWrites[7].dstArrayElement = 0;
    descriptorWrites[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[7].descriptorCount = 1;
    descriptorWrites[7].pBufferInfo = &storageBufferOctrees;

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
    std::cout << node_count << std::endl;
}

void Application::updateUniformBuffer() {
    UniformBufferObject ubo{};
    ubo.width = WIDTH;
    ubo.height = HEIGHT;

    LiteMath::float3 lookAt{ 0.0f, 0.0f, 1.0f };
    lookAt = LiteMath::rotate4x4Y(phi) * LiteMath::rotate4x4X(theta) * lookAt;
    ubo.view = LiteMath::inverse4x4(LiteMath::lookAt(position, position + lookAt, LiteMath::float3(0.0f, 1.0f, 0.0f)));
    ubo.proj = LiteMath::inverse4x4(LiteMath::perspectiveMatrix(90.0f, (float)WIDTH / HEIGHT, 0.01f, 100.0f));

    memcpy(_uniformBufferMapped, &ubo, sizeof(ubo));
}

void Application::updateOutputImageData() {
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

    void *data;
    vkMapMemory(_device.device(), stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(_pixels.data(), data, WIDTH * HEIGHT * 4);

    vkUnmapMemory(_device.device(), stagingBufferMemory);

    vkDestroyBuffer(_device.device(), stagingBuffer, nullptr);
    vkFreeMemory(_device.device(), stagingBufferMemory, nullptr);
}

#include <iostream>

void Application::drawFrame() {
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    vkWaitForFences(_device.device(), 1, &_computeInFlightFence, VK_TRUE, UINT64_MAX);
    
    updateOutputImageData();

    updateUniformBuffer();

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