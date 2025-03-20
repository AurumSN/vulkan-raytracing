#pragma once

#include "window.h"
#include "pipeline.h"
#include "device.h"
#include "LiteMath/LiteMath.h"
#include "mesh.h"

#include <memory>
#include <vector>

struct UniformBufferObject {
    int width;
    int height;
    int _1;
    int _2;
    LiteMath::float4x4 view;
    LiteMath::float4x4 proj;
};

struct Material {
    LiteMath::float4 color;
    float reflective;
    float refractiv; // not used

    static Material Rough(LiteMath::float3 color);

    const static Material RoughWhite;
    const static Material PerfectMirror;
};

struct Primitive {
    LiteMath::float4 data;
    unsigned int type;
    float _1;
    float _2;
    float _3;
    Material material;
    float _4;
    float _5;

    static Primitive Plane(LiteMath::float3 normal, LiteMath::float3 position, Material material = Material::RoughWhite);

    static Primitive Sphere(LiteMath::float3 position, float radius, Material material = Material::RoughWhite);
};

struct PreMesh {
    cmesh4::SimpleMesh mesh;
    Material material;
    LiteMath::float4x4 modelMatrix;
};

struct Mesh {
    unsigned int begin;
    float _1;
    float _2;
    float _3;
    Material material;
    float _4;
    float _5;
};

struct BVH {
    unsigned int index;
    unsigned int escape;
    unsigned int left;
    unsigned int right;
    LiteMath::float4 AABBmin;
    LiteMath::float4 AABBmax;
};

struct Light {
    LiteMath::float4 position;
    LiteMath::float4 color;

    static Light Directional(LiteMath::float3 direction, LiteMath::float3 color = LiteMath::float3(1.0f, 1.0f, 1.0f), float intensity = 1.0f);

    static Light Point(LiteMath::float3 position, LiteMath::float3 color = LiteMath::float3(1.0f, 1.0f, 1.0f), float intensity = 1.0f);
};

class Application {
    std::vector<Primitive> _Primitives{ 
        {
            .data = LiteMath::float4(0.0f, 1.0f, 0.0f, 1.0f),
            .type = 0,
            .material = {LiteMath::float4(0.0f, 1.0f, 0.0f, 1.0f), 0.0f, 0.0f}
        },
        {
            .data = LiteMath::float4(1.0f, 0.0f, 0.0f, 1.0f),
            .type = 0,
            .material = {LiteMath::float4(1.0f, 0.0f, 0.0f, 1.0f), 0.9f, 0.0f}
        },
        {
            .data = LiteMath::float4(0.0f, 0.0f, 1.0f, 0.5f),
            .type = 1,
            .material = {LiteMath::float4(1.0f, 0.0f, 1.0f, 1.0f), 0.0f, 0.0f}
        }
    };

    std::vector<PreMesh> _Meshes{
        {
            .mesh = cmesh4::LoadMeshFromObj("./data/stanford-bunny.obj", false),
            .material = {LiteMath::float4(0.0f, 0.0f, 0.0f, 1.0f), 0.9f, 0.0f},
            .modelMatrix = LiteMath::scale4x4(LiteMath::float3(10.0f, 10.0f, 10.0f))
        },
        {
            .mesh = cmesh4::LoadMeshFromObj("./data/spot.obj", false),
            .material = {LiteMath::float4(0.0f, 1.0f, 1.0f, 1.0f), 0.0f, 0.0f},
            .modelMatrix = LiteMath::translate4x4(LiteMath::float3(2.0f, 0.0f, 0.0f))
        }
    };

    std::vector<Light> _Lights {
        Light::Directional(LiteMath::float3(0.0f, -1.0f, -1.0f)),
        Light::Point(LiteMath::float3(0.0f, 0.0f, -1.0f), LiteMath::float3(1.0f, 1.0f, 1.0f), 10.0f)
    };

    Window _window{ WIDTH, HEIGHT, "Hello Vulkan!" };
    Device _device{ _window };
    // SwapChain _swapChain{ _device, _window.getExtent() };
    //Pipeline _pipeline{ _device, "shaders/shader.vert.spv", "shaders/shader.frag.spv", Pipeline::defaultPipelineConfigInfo(WIDTH, HEIGHT) };
    std::unique_ptr<Pipeline> _pipeline;
    VkPipelineLayout _pipelineLayout;
    VkCommandBuffer _commandBuffer;
    VkDescriptorSetLayout _computeDescriptorSetLayout;
    VkImage _image;
    VkDeviceMemory _imageMemory;
    VkImageView _imageView;
    VkSampler _imageSampler;
    VkDescriptorSet _computeDescriptorSet;
    VkDescriptorPool _descriptorPool;
    VkFence _computeInFlightFence;
    VkSemaphore _computeFinishedSemaphore;
    std::vector<uint32_t> _pixels;

    VkBuffer _uniformBuffer;
    void *_uniformBufferMapped;
    VkDeviceMemory _uniformBufferMemory;

    VkBuffer _shaderStorageBufferPrimitves;
    VkDeviceMemory _shaderStorageBufferPrimitvesMemory;

    VkBuffer _shaderStorageBufferVertices;
    VkDeviceMemory _shaderStorageBufferVerticesMemory;

    VkBuffer _shaderStorageBufferIndices;
    VkDeviceMemory _shaderStorageBufferIndicesMemory;

    VkBuffer _shaderStorageBufferMeshes;
    VkDeviceMemory _shaderStorageBufferMeshesMemory;

    VkBuffer _shaderStorageBufferTriangles;
    VkDeviceMemory _shaderStorageBufferTrianglesMemory;

    VkBuffer _shaderStorageBufferBVHs;
    VkDeviceMemory _shaderStorageBufferBVHsMemory;

    VkBuffer _shaderStorageBufferLights;
    VkDeviceMemory _shaderStorageBufferLightsMemory;

    void createPipelineLayout();
    void createPipeline();
    void createCommandBuffers();
    void drawFrame();
    void buildBVH(const std::vector<LiteMath::float4>& vertices, const std::vector<unsigned int>& indices, std::vector<unsigned int>::iterator triangles_begin, std::vector<unsigned int>::iterator triangles_end, std::vector<unsigned int>& triangleList, std::vector<BVH>& bvhs);
    void updateBVH(std::vector<BVH>& bvhs);

    float phi = 0.0f;
    float theta = 0.0f;
    LiteMath::float3 position{0.0f, 0.0f, -10.0f};
    
public:
    static constexpr int WIDTH = 800;
    static constexpr int HEIGHT = 600;

    Application();
    ~Application();

    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;

    void run();
};