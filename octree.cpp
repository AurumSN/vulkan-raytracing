#include "octree.h"

#include <fstream>

void save_sdf_octree(const SdfOctree &scene, const std::string &path) {
    std::ofstream fs(path, std::ios::binary);
    size_t size = scene.nodes.size();
    fs.write((const char *)&size, sizeof(unsigned));
    fs.write((const char *)scene.nodes.data(), size * sizeof(SdfOctreeNode));
    fs.flush();
    fs.close();
}

void load_sdf_octree(SdfOctree &scene, const std::string &path) {
    std::ifstream fs(path, std::ios::binary);
    unsigned sz = 0;
    fs.read((char *)&sz, sizeof(unsigned));
    scene.nodes.resize(sz);
    fs.read((char *)scene.nodes.data(), scene.nodes.size() * sizeof(SdfOctreeNode));
    fs.close();
}