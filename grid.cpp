#include "grid.h"

#include <fstream>

void save_sdf_grid(const SdfGrid &scene, const std::string &path) {
    std::ofstream fs(path, std::ios::binary);
    fs.write((const char *)&scene.size, 3 * sizeof(unsigned));
    fs.write((const char *)scene.data.data(), scene.size.x * scene.size.y * scene.size.z * sizeof(float));
    fs.flush();
    fs.close();
}

void load_sdf_grid(SdfGrid &scene, const std::string &path) {
    std::ifstream fs(path, std::ios::binary);
    fs.read((char *)&scene.size, 3 * sizeof(unsigned));
    scene.data.resize(scene.size.x * scene.size.y * scene.size.z);
    fs.read((char *)scene.data.data(), scene.size.x * scene.size.y * scene.size.z * sizeof(float));
    fs.close();
}