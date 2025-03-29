#pragma once

#include <string>
#include <vector>

struct SdfOctreeNode {
    float values[8];
    unsigned offset;
};

struct SdfOctree {
    std::vector<SdfOctreeNode> nodes;
};

void save_sdf_octree(const SdfOctree &scene, const std::string &path);

void load_sdf_octree(SdfOctree &scene, const std::string &path);