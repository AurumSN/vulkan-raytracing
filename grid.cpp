#include "grid.h"

#include <fstream>

void save_sdf_grid(const SdfGrid &scene, const std::string &path)
{
  std::ofstream fs(path, std::ios::binary);
  fs.write((const char *)&scene.size, 3 * sizeof(unsigned));
  fs.write((const char *)scene.data.data(), scene.size.x * scene.size.y * scene.size.z * sizeof(float));
  fs.flush();
  fs.close();
}

void load_sdf_grid(SdfGrid &scene, const std::string &path)
{
  std::ifstream fs(path, std::ios::binary);
  fs.read((char *)&scene.size, 3 * sizeof(unsigned));
  scene.data.resize(scene.size.x * scene.size.y * scene.size.z);
  fs.read((char *)scene.data.data(), scene.size.x * scene.size.y * scene.size.z * sizeof(float));
  fs.close();
}

// void draw_sdf_grid_slice(const SdfGrid &grid, int z_level, int voxel_size, int width, int height, std::vector<uint32_t> &pixels)
// {
//   constexpr uint32_t COLOR_EMPTY = 0xFF333333;  // dark gray
//   constexpr uint32_t COLOR_FULL = 0xFFFFA500;   // orange
//   constexpr uint32_t COLOR_BORDER = 0xFF000000; // black

//   for (int y = 0; y < grid.size.y; y++)
//   {
//     for (int x = 0; x < grid.size.x; x++)
//     {
//       int index = x + y * grid.size.x + z_level * grid.size.x * grid.size.y;
//       uint32_t color = grid.data[index] < 0 ? COLOR_FULL : COLOR_EMPTY;
//       for (int i = 0; i <= voxel_size; i++)
//       {
//         for (int j = 0; j <= voxel_size; j++)
//         {
//           // flip the y axis
//           int pixel_idx = (x * voxel_size + i) + ((height - 1) - (y * voxel_size + j)) * width;
//           if (i == 0 || i == voxel_size || j == 0 || j == voxel_size)
//             pixels[pixel_idx] = COLOR_BORDER;
//           else
//             pixels[pixel_idx] = color;
//         }
//       }
//     }
//   }
// }