#pragma once

#include "window.h"
#include "pipeline.h"
#include "device.h"
#include "LiteMath/LiteMath.h"
#include "mesh.h"
#include "octree.h"
#include "swapchain.h"

#include <memory>
#include <vector>
#include <array>

struct UniformBufferObject {
    int width;
    int height;
    int _1;
    int _2;
    LiteMath::float4x4 view;
    LiteMath::float4x4 proj;
};

struct Material {
    LiteMath::float4 color = LiteMath::float4(1.0f, 1.0f, 1.0f, 1.0f);
    float reflective = 0.0f;
    float refractiv = 0.0f; // not used

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

struct Grid {
    std::string name;
    Material material;
    LiteMath::float4x4 inverseModel;
};

struct GridData {
    unsigned index;
    unsigned _1;
    unsigned _2;
    unsigned _3;
    LiteMath::uint3 size;
    unsigned _4;
    Material material;
    unsigned _5;
    unsigned _6;
    LiteMath::float4x4 inverseModel;
};

struct Octree {
    std::string name;
    Material material;
    LiteMath::float4x4 inverseModel;
};

struct OctreeNode {
    float values[8];
    unsigned offset;
    unsigned escape;
    unsigned _1;
    unsigned _2;
    LiteMath::float4 min;
    LiteMath::float4 max;
};

struct OctreeData {
    unsigned index;
    unsigned _1;
    unsigned _2;
    unsigned _3;
    Material material;
    unsigned _4;
    unsigned _5;
    LiteMath::float4x4 inverseModel;
};

void buildBVH(const std::vector<LiteMath::float4>& vertices, const std::vector<unsigned int>& indices, std::vector<unsigned int>::iterator triangles_begin, std::vector<unsigned int>::iterator triangles_end, std::vector<unsigned int>& triangleList, std::vector<BVH>& bvhs);
void updateBVH(std::vector<BVH>& bvhs);

class Application {

    std::vector<Primitive> _Primitives;
    std::vector<PreMesh> _Meshes;
    std::vector<Light> _Lights;
    std::vector<Grid> _Grids;
    std::vector<Octree> _Octrees;

    Window _window{ WIDTH, HEIGHT, "Hello Vulkan!" };
    Device _device{ _window };
    SwapChain _swapChain{ _device, _window.getExtent() };
    std::unique_ptr<Pipeline> _pipeline;
    VkPipelineLayout _graphicsPipelineLayout;
    VkPipelineLayout _computePipelineLayout;
    std::vector<VkCommandBuffer> _commandBuffers;
    std::array<VkCommandBuffer, SwapChain::MAX_FRAMES_IN_FLIGHT> _computeCommandBuffers;
    VkDescriptorSetLayout _graphicsDescriptorSetLayout;
    VkDescriptorSetLayout _computeDescriptorSetLayout;
    std::array<VkImage, SwapChain::MAX_FRAMES_IN_FLIGHT> _images;
    std::array<VkDeviceMemory, SwapChain::MAX_FRAMES_IN_FLIGHT> _imagesMemory;
    std::array<VkImageView, SwapChain::MAX_FRAMES_IN_FLIGHT> _imagesView;
    std::array<VkSampler, SwapChain::MAX_FRAMES_IN_FLIGHT> _imagesSampler;
    std::array<VkDescriptorSet, SwapChain::MAX_FRAMES_IN_FLIGHT> _graphicsDescriptorSets;
    std::array<VkDescriptorSet, SwapChain::MAX_FRAMES_IN_FLIGHT> _computeDescriptorSets;
    VkDescriptorPool _descriptorPool;
    VkDescriptorPool _graphicsDescriptorPool;
    std::array<VkFence, SwapChain::MAX_FRAMES_IN_FLIGHT> _computeInFlightFences;
    std::array<VkSemaphore, SwapChain::MAX_FRAMES_IN_FLIGHT> _computeFinishedSemaphores;

    std::array<VkBuffer, SwapChain::MAX_FRAMES_IN_FLIGHT> _uniformBuffers;
    std::array<void *, SwapChain::MAX_FRAMES_IN_FLIGHT> _uniformBuffersMapped;
    std::array<VkDeviceMemory, SwapChain::MAX_FRAMES_IN_FLIGHT> _uniformBuffersMemory;

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

    VkBuffer _shaderStorageBufferGridDistances;
    VkDeviceMemory _shaderStorageBufferGridDistancesMemory;

    VkBuffer _shaderStorageBufferGrids;
    VkDeviceMemory _shaderStorageBufferGridsMemory;

    VkBuffer _shaderStorageBufferOctreeNodes;
    VkDeviceMemory _shaderStorageBufferOctreeNodesMemory;

    VkBuffer _shaderStorageBufferOctrees;
    VkDeviceMemory _shaderStorageBufferOctreesMemory;

    VkBuffer _stagingImageBuffer;
    VkDeviceMemory _stagingImageBufferMemory;
    void* _stagingImageBufferMemoryMapped;

    VkDescriptorBufferInfo createUniformBuffer(size_t frame);
    VkDescriptorImageInfo createOutputImageBuffer(size_t frame);
    template<typename T>
    VkDescriptorBufferInfo createBuffer(const std::vector<T>& data, VkBuffer& buffer, VkDeviceMemory& memory, bool withOffset = true);
    template<typename T>
    VkDescriptorBufferInfo createCountedBuffer(const std::vector<T>& data, VkBuffer& buffer, VkDeviceMemory& memory, bool withOffset = true);
    void createGraphicsPipelineLayout();
    void createRaytracingPipelineLayout();
    void createRaytracingPipeline();
    void createCommandBuffers();
    void updateUniformBuffer(uint32_t currentImage);
    void drawFrame();

    float phi = 0.0f;
    float theta = 0.0f;
    LiteMath::float3 position{0.0f, 0.0f, -5.0f};

    bool _isSphereTracing = false;
    
public:
    static constexpr int WIDTH = 800;
    static constexpr int HEIGHT = 600;

    Application();
    Application(std::vector<PreMesh> meshes);
    Application(std::vector<Grid> grids);
    Application(std::vector<Octree> octrees);
    Application(std::vector<Light> lights);
    Application(std::vector<PreMesh> meshes, std::vector<Light> lights);
    Application(std::vector<Grid> grids, std::vector<Light> lights);
    Application(std::vector<Octree> octrees, std::vector<Light> lights);
    Application(std::vector<Primitive> primitives);
    Application(std::vector<Primitive> primitives, std::vector<PreMesh> meshes);
    Application(std::vector<Primitive> primitives, std::vector<Grid> grids);
    Application(std::vector<Primitive> primitives, std::vector<Octree> octrees);
    Application(std::vector<Primitive> primitives, std::vector<Light> lights);
    Application(std::vector<Primitive> primitives, std::vector<PreMesh> meshes, std::vector<Light> lights);
    Application(std::vector<Primitive> primitives, std::vector<Grid> grids, std::vector<Light> lights);
    Application(std::vector<Primitive> primitives, std::vector<Octree> octrees, std::vector<Light> lights);
    Application(std::vector<Primitive> primitives, std::vector<PreMesh> meshes, std::vector<Grid> grids, std::vector<Octree> octrees, std::vector<Light> lights);
    ~Application();

    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;

    void run();
};