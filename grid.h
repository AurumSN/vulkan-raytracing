#pragma once

#include "LiteMath/LiteMath.h"

#include <string>
#include <vector>

struct SdfGrid {
    LiteMath::uint3 size;
    std::vector<float> data;
};

void save_sdf_grid(const SdfGrid &scene, const std::string &path);

void load_sdf_grid(SdfGrid &scene, const std::string &path);

void draw_sdf_grid_slice(const SdfGrid &grid, int z_level, int voxel_size, int width, int height, std::vector<uint32_t> &pixels);