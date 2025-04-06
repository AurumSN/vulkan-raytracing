#include "application.h"

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <sstream>
#include <thread>

#include "grid.h"

std::string readString(std::istream& stream) {
    int ch;
    while ((ch = stream.peek()) != EOF) {
        if (std::isspace(ch)) {
            stream.get();
        } else {
            break;
        }
    }

    std::stringstream ss;
    char c;
    bool open = false;
    while (stream.get(c)) {
        if (c == '"') {
            if (open) {
                open = false;
                break;
            } else {
                open = true;
            }
        } else if (!open) {
            throw std::runtime_error("Could not parse string value: string doesn't begin with character '\"'");
        } else {
            ss << c;
        }
    }

    if (open) {
        throw std::runtime_error("Could not parse string value: the string doesn't end with character '\"'");
    }

    return ss.str();
}

float sdTriangle( LiteMath::float4 p, LiteMath::float4 a, LiteMath::float4 b, LiteMath::float4 c )
{
    LiteMath::float4 ba = b - a; 
    LiteMath::float4 pa = p - a;
    LiteMath::float4 cb = c - b; 
    LiteMath::float4 pb = p - b;
    LiteMath::float4 ac = a - c; 
    LiteMath::float4 pc = p - c;
    LiteMath::float4 nor = LiteMath::cross( ba, ac );
    float k = LiteMath::dot(nor,pa) <= 0.0f ? 1.0f : -1.0f;

    return 
        (LiteMath::sign(LiteMath::dot(LiteMath::cross(ba,nor),pa)) +
        LiteMath::sign(LiteMath::dot(LiteMath::cross(cb,nor),pb)) +
        LiteMath::sign(LiteMath::dot(LiteMath::cross(ac,nor),pc))<2.0)
        ?
        k * sqrtf(LiteMath::min( LiteMath::min(
                    LiteMath::dot(ba*LiteMath::clamp(LiteMath::dot(ba,pa)/LiteMath::dot(ba, ba),0.0f,1.0f)-pa,ba*LiteMath::clamp(LiteMath::dot(ba,pa)/LiteMath::dot(ba, ba),0.0f,1.0f)-pa),
                    LiteMath::dot(cb*LiteMath::clamp(LiteMath::dot(cb,pb)/LiteMath::dot(cb, cb),0.0f,1.0f)-pb,cb*LiteMath::clamp(LiteMath::dot(cb,pb)/LiteMath::dot(cb, cb),0.0f,1.0f)-pb)),
                    LiteMath::dot(ac*LiteMath::clamp(LiteMath::dot(ac,pc)/LiteMath::dot(ac, ac),0.0f,1.0f)-pc,ac*LiteMath::clamp(LiteMath::dot(ac,pc)/LiteMath::dot(ac, ac),0.0f,1.0f)-pc)))
        :
        -LiteMath::dot(nor,pa)/LiteMath::length(nor);
}

float distance(LiteMath::float4 point, LiteMath::float4 vertices[3]) {
    return sdTriangle(point, vertices[0], vertices[1], vertices[2]);
}

float sdSphere(LiteMath::float4 point, LiteMath::float4 sphere) {
    LiteMath::float3 p = LiteMath::float3(point.x - sphere.x, point.y - sphere.y, point.z - sphere.z);
    return LiteMath::length(p) - sphere.w;
}

float minabs(float a, float b) {
    if (std::abs(a) < std::abs(b)) {
        return a;
    }
    return b;
}

float maxabs(float a, float b) {
    if (std::abs(a) > std::abs(b)) {
        return a;
    }
    return b;
}

bool isInsideTriangle(LiteMath::float4 p, LiteMath::float4 a, LiteMath::float4 b, LiteMath::float4 c) {
    LiteMath::float4 nor = LiteMath::cross(b - a, a - c);
    return LiteMath::sign(LiteMath::dot(LiteMath::cross(b - a, nor), p - a)) + LiteMath::sign(LiteMath::dot(LiteMath::cross(c - b, nor), p - b)) + LiteMath::sign(LiteMath::dot(LiteMath::cross(a - c, nor), p - c)) >= 2.0;
}

void fastSweep(std::vector<float>& grid, const std::vector<bool>& frozen, LiteMath::int3 size) {
    const int NSweeps = 8;
    const int dirX[NSweeps][3] = { {size.x - 1, 0, -1}, {0, size.x - 1, 1}, {0, size.x - 1, 1}, {size.x - 1, 0, -1}, {size.x - 1, 0, -1}, {0, size.x - 1, 1}, {0, size.x - 1, 1}, {size.x - 1, 0, -1} };
    const int dirY[NSweeps][3] = { {size.y - 1, 0, -1}, {size.y - 1, 0, -1}, {0, size.y - 1, 1}, {0, size.y - 1, 1}, {size.y - 1, 0, -1}, {size.y - 1, 0, -1}, {0, size.y - 1, 1}, {0, size.y - 1, 1} };
    const int dirZ[NSweeps][3] = { {0, size.z - 1, 1}, {0, size.z - 1, 1}, {0, size.z - 1, 1}, {0, size.z - 1, 1}, {size.z - 1, 0, -1}, {size.z - 1, 0, -1}, {size.z - 1, 0, -1}, {size.z - 1, 0, -1} };
    float aa[3], eps = 1e-6f;
    float d_new, a, b;
    int s, ix, iy, iz, gridPos;
    const float h = 1.0f / (size.x - 1), f = 1.0f;
 
    for (s = 0; s < NSweeps; s++) {
        for (iz = dirZ[s][0]; dirZ[s][2] * iz <= dirZ[s][1]; iz += dirZ[s][2]) {
            for (iy = dirY[s][0]; dirY[s][2] * iy <= dirY[s][1]; iy += dirY[s][2]) {
                for (ix = dirX[s][0]; dirX[s][2] * ix <= dirX[s][1]; ix += dirX[s][2]) {
    
                    gridPos = ix + iy * size.x + iz * size.x * size.y;
    
                    if (!frozen[gridPos]) {
                        if (iz == 0 || iz == (size.z - 1)) {
                            if (iz == 0) {
                                aa[2] = minabs(grid[gridPos], grid[gridPos + size.x * size.y]);
                            }
                            if (iz == (size.z - 1)) {
                                aa[2] = minabs(grid[gridPos], grid[gridPos - size.x * size.y]);
                            }
                        }
                        else {
                            aa[2] = minabs(grid[gridPos - size.x * size.y], grid[gridPos + size.x * size.y]);
                        }

                        if (iy == 0 || iy == (size.y - 1)) {
                            if (iy == 0) {
                                aa[1] = minabs(grid[gridPos], grid[gridPos + size.x]);
                            }
                            if (iy == (size.y - 1)) {
                                aa[1] = minabs(grid[gridPos], grid[gridPos - size.x]);
                            }
                        }
                        else {
                            aa[1] = minabs(grid[gridPos - size.x], grid[gridPos + size.x]);
                        }
    
                        if (ix == 0 || ix == (size.x - 1)) {
                            if (ix == 0) {
                                aa[0] = minabs(grid[gridPos], grid[gridPos + 1]);
                            }
                            if (ix == (size.x - 1)) {
                                aa[0] = minabs(grid[gridPos], grid[gridPos - 1]);
                            }
                        }
                        else {
                            aa[0] = minabs(grid[gridPos - 1], grid[gridPos + 1]);
                        }

                        float tmp;
    
                        if (std::abs(aa[0]) > std::abs(aa[1])) { tmp = aa[0]; aa[0] = aa[1]; aa[1] = tmp; }
                        if (std::abs(aa[1]) > std::abs(aa[2])) { tmp = aa[1]; aa[1] = aa[2]; aa[2] = tmp; }
                        if (std::abs(aa[0]) > std::abs(aa[1])) { tmp = aa[0]; aa[0] = aa[1]; aa[1] = tmp; }
                
                        float d_curr = aa[0] + h * f * (aa[0] > 0.0f ? 1.0f : -1.0f);
                        if (std::abs(d_curr) <= std::abs(aa[1]) + eps || aa[0] * aa[1] < 0.0f) {
                            d_new = d_curr;
                        } else {
                            float a = 2.0f; 
                            float b = -2.0f * (aa[0] + aa[1]);
                            float c = aa[0] * aa[0] + aa[1] * aa[1] - h * h * f * f;
                            float D = sqrtf(b * b - 4.0f * a * c);
                            d_curr = maxabs(-b + D, -b - D) / (2.0f * a);
                
                            if (std::abs(d_curr) <= std::abs(aa[2]) + eps || aa[1] * aa[2] < 0.0f) {
                                d_new = d_curr;
                            } else {
                                a = 3.0f;
                                b = -2.0f * (aa[0] + aa[1] + aa[2]);
                                c = aa[0] * aa[0] + aa[1] * aa[1] + aa[2] * aa[2] - h * h * f * f;
                                D = sqrtf(b * b - 4.0f * a * c);
                                d_new = maxabs(-b + D, -b - D) / (2.0f * a);
                            }
                        }
                        grid[gridPos] = minabs(grid[gridPos], d_new);
                    }
                }
            }
        }
    }
}

bool intersectAABBTriangle(LiteMath::float4 min, LiteMath::float4 max, LiteMath::float4 vertices[3]) {
    LiteMath::float4 minTriangle = vertices[0];
    LiteMath::float4 maxTriangle = vertices[0];
    for (int i = 1; i < 3; i++) {
        minTriangle = LiteMath::min(minTriangle, vertices[i]);
        maxTriangle = LiteMath::max(maxTriangle, vertices[i]);
    }

    for (int i = 0; i < 3; i++) {
        if (minTriangle[i] > max[i] || maxTriangle[i] < min[i]) {
            return false;
        }
    }

    LiteMath::float4 directions[13]{
        LiteMath::float4(1.0f, 0.0f, 0.0f, 0.0f),
        LiteMath::float4(0.0f, 1.0f, 0.0f, 0.0f),
        LiteMath::float4(0.0f, 0.0f, 1.0f, 0.0f),
        LiteMath::cross(LiteMath::float4(1.0f, 0.0f, 0.0f, 0.0f), vertices[1] - vertices[0]),
        LiteMath::cross(LiteMath::float4(1.0f, 0.0f, 0.0f, 0.0f), vertices[2] - vertices[1]),
        LiteMath::cross(LiteMath::float4(1.0f, 0.0f, 0.0f, 0.0f), vertices[0] - vertices[2]),
        LiteMath::cross(LiteMath::float4(0.0f, 1.0f, 0.0f, 0.0f), vertices[1] - vertices[0]),
        LiteMath::cross(LiteMath::float4(0.0f, 1.0f, 0.0f, 0.0f), vertices[2] - vertices[1]),
        LiteMath::cross(LiteMath::float4(0.0f, 1.0f, 0.0f, 0.0f), vertices[0] - vertices[2]),
        LiteMath::cross(LiteMath::float4(0.0f, 0.0f, 1.0f, 0.0f), vertices[1] - vertices[0]),
        LiteMath::cross(LiteMath::float4(0.0f, 0.0f, 1.0f, 0.0f), vertices[2] - vertices[1]),
        LiteMath::cross(LiteMath::float4(0.0f, 0.0f, 1.0f, 0.0f), vertices[0] - vertices[2]),
        LiteMath::cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    };

    LiteMath::float4 c = (min + max) * 0.5f;
    LiteMath::float4 e = max - min;

    for (int i = 0; i < 13; i++) {
        float p0 = LiteMath::dot(vertices[0] - c, directions[i]);
        float p1 = LiteMath::dot(vertices[1] - c, directions[i]);
        float p2 = LiteMath::dot(vertices[2] - c, directions[i]);

        float r = e.x * std::abs(LiteMath::dot(LiteMath::float4(1.0f, 0.0f, 0.0f, 0.0f), directions[i])) + e.y * std::abs(LiteMath::dot(LiteMath::float4(0.0f, 1.0f, 0.0f, 0.0f), directions[i])) + e.z * std::abs(LiteMath::dot(LiteMath::float4(0.0f, 0.0f, 1.0f, 0.0f), directions[i]));

        if (std::max(-std::max(p0, std::max(p1, p2)), std::min(p0, std::min(p1, p2))) > r) {
            return false;
        }
    }

    return true;
}

void splitOctree(std::vector<SdfOctreeNode>& nodes, const std::vector<LiteMath::float4>& vertices, const std::vector<unsigned>& indices, const std::vector<unsigned>& triangles, unsigned offset, LiteMath::float4 min, LiteMath::float4 max, unsigned depth) {
    constexpr float epsilon = 0.001f;
    
    LiteMath::float4 points[8]{
        min,
        LiteMath::float4(max.x, min.y, min.z, min.w),
        LiteMath::float4(min.x, max.y, min.z, min.w),
        LiteMath::float4(max.x, max.y, min.z, min.w),
        LiteMath::float4(min.x, min.y, max.z, min.w),
        LiteMath::float4(max.x, min.y, max.z, min.w),
        LiteMath::float4(min.x, max.y, max.z, min.w),
        max
    };
    LiteMath::float4 minimums[8];
    LiteMath::float4 maximums[8];
    for (int i = 0; i < 8; i++) {
        int arr[3]{ i % 2, (i >> 1) % 2, i >> 2 };
        for (int j = 0; j < 3; j++) {
            if (arr[j]) {
                minimums[i][j] = (min[j] + max[j]) * 0.5f;
                maximums[i][j] = max[j];
            } else {
                minimums[i][j] = min[j];
                maximums[i][j] = (min[j] + max[j]) * 0.5f;
            }
        }
    }

    if (depth == 0 || triangles.size() == 0) {
        nodes[offset].offset = 0;
        for (int i = 0; i < 8; i++) {
            nodes[offset].values[i] = 1000.0f;
        }

        for (const auto& i : triangles) {
            LiteMath::float4 vert[3]{
                vertices[indices[i + 0]],
                vertices[indices[i + 1]],
                vertices[indices[i + 2]]
            };
            for (int j = 0; j < 8; j++) {
                float da = distance(points[j], vert);
                if (nodes[offset].values[j] < 0.0f && da >= 0.0f) {
                    if (std::abs(nodes[offset].values[j]) + epsilon > std::abs(da)) {
                        nodes[offset].values[j] = da;
                    }
                } else if (nodes[offset].values[j] >= 0.0f && da < 0.0f) {
                    if (std::abs(nodes[offset].values[j]) > std::abs(da) + epsilon) {
                        nodes[offset].values[j] = da;
                    }
                } else if (std::abs(nodes[offset].values[j]) > std::abs(da)) {
                    nodes[offset].values[j] = da;
                }
            }
        }
    } else {
        std::array<std::vector<unsigned>, 8> tris;
        for (const auto& i : triangles) {
            LiteMath::float4 vert[3]{
                vertices[indices[i + 0]],
                vertices[indices[i + 1]],
                vertices[indices[i + 2]]
            };
            for (int j = 0; j < 8; j++) {
                if (intersectAABBTriangle(minimums[j], maximums[j], vert)) {
                    tris[j].push_back(i);
                }
            }
        }

        nodes[offset].offset = nodes.size();
        nodes.resize(nodes.size() + 8);
        for (int i = 0; i < 8; i++) {
            splitOctree(nodes, vertices, indices, tris[i], nodes[offset].offset + i, minimums[i], maximums[i], depth - 1);
        }
    }
}

bool intersectAABBSphere(LiteMath::float4 min, LiteMath::float4 max, LiteMath::float4 sphere) {
    float dmin = 0;
    for (int i = 0; i < 3; i++) {
        if (sphere[i] < min[i]) {
            dmin += std::sqrt(sphere[i] - min[i]);
        } else if (sphere[i] > max[i]) {
            dmin += std::sqrt(sphere[i] - max[i]);  
        }
    }

    if (dmin <= sphere.w * sphere.w) {
        return true;
    }

    return false;
}

void sphereCutter(const SdfOctree& octree, SdfOctree& output, LiteMath::float4 sphere, unsigned offset, unsigned offsetOutput, LiteMath::float4 min, LiteMath::float4 max) {
    LiteMath::float4 points[8]{
        min,
        LiteMath::float4(max.x, min.y, min.z, min.w),
        LiteMath::float4(min.x, max.y, min.z, min.w),
        LiteMath::float4(max.x, max.y, min.z, min.w),
        LiteMath::float4(min.x, min.y, max.z, min.w),
        LiteMath::float4(max.x, min.y, max.z, min.w),
        LiteMath::float4(min.x, max.y, max.z, min.w),
        max
    };

    LiteMath::float4 minimums[8];
    LiteMath::float4 maximums[8];
    for (int i = 0; i < 8; i++) {
        int arr[3]{ i % 2, (i >> 1) % 2, i >> 2 };
        for (int j = 0; j < 3; j++) {
            if (arr[j]) {
                minimums[i][j] = (min[j] + max[j]) * 0.5f;
                maximums[i][j] = max[j];
            } else {
                minimums[i][j] = min[j];
                maximums[i][j] = (min[j] + max[j]) * 0.5f;
            }
        }
    }

    if (true || intersectAABBSphere(min, max, sphere)) {
        bool b = true;
        for (int i = 0; i < 8; i++) {
            if (LiteMath::dot(points[i] - LiteMath::float4(sphere.x, sphere.y, sphere.z, 1.0f), points[i] - LiteMath::float4(sphere.x, sphere.y, sphere.z, 1.0f)) > sphere.w * sphere.w) {
                b = false;
                break;
            }
        }

        if (b) {
            output.nodes[offsetOutput].offset = 0;
            for (int j = 0; j < 8; j++) {
                output.nodes[offsetOutput].values[j] = 1000.0f;
            }
        } else if (octree.nodes[offset].offset == 0) {
            bool b2 = false;
            
            for (int j = 0; j < 8; j++) {
                if (octree.nodes[offset].values[j] != 0 && octree.nodes[offset].values[j] < 10.0f) {
                    b2 = true;
                    break;
                }
            }

            for (int j = 0; j < 8; j++) {
                output.nodes[offsetOutput].values[j] = 1000.0f;
            }

            if (b2) {
                for (int j = 0; j < 8; j++) {
                    output.nodes[offsetOutput].values[j] = std::max(octree.nodes[offset].values[j], -sdSphere(points[j], sphere));
                }
            }
        } else {
            output.nodes[offsetOutput].offset = output.nodes.size();
            output.nodes.resize(output.nodes.size() + 8);
            for (int j = 0; j < 8; j++) {
                sphereCutter(octree, output, sphere, octree.nodes[offset].offset + j, output.nodes[offsetOutput].offset + j, minimums[j], maximums[j]);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    const char OBJ_FILE[] = ".obj";
    const char GRID_FILE[] = ".grid";
    const char OCTREE_FILE[] = ".octree";
    const char SCENE_FILE[] = ".scene";
    const char OUTPUT[] = "-o";
    const char SIZE[] = "-s";
    const char SPHERE[] = "-sph";
    const char DEPTH[] = "-d";

    if (argc == 1) {
        std::cout << "Please, enter file name" << std::endl;

        return EXIT_SUCCESS;
    }

    const char* input_path = "";
    const char* output_path = nullptr;
    unsigned size = 256;
    unsigned depth = 5;
    std::vector<LiteMath::float4> spheres;

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], OUTPUT, sizeof(OUTPUT)) == 0) {
            output_path = argv[i + 1];
            i++;
        } else if (strncmp(argv[i], SIZE, sizeof(SIZE)) == 0) {
            size = std::atoi(argv[i + 1]);
            i++;
        } else if (strncmp(argv[i], DEPTH, sizeof(DEPTH)) == 0) {
            depth = std::atoi(argv[i + 1]);
            i++;
        } else if (strncmp(argv[i], SPHERE, sizeof(SPHERE)) == 0) {
            LiteMath::float4 sphere;
            sphere.x = std::atof(argv[i + 1]);
            sphere.y = std::atof(argv[i + 2]);
            sphere.z = std::atof(argv[i + 3]);
            sphere.w = std::atof(argv[i + 4]);
            spheres.push_back(sphere);
            i += 4;
        } else {
            input_path = argv[i];
        }
    }

    size_t len = strnlen(input_path, 255);

    if (output_path != nullptr) {
        if (strncmp(&input_path[len - sizeof(OBJ_FILE) + 1], OBJ_FILE, sizeof(OBJ_FILE)) == 0) {
            len = strnlen(output_path, 255);
            if (strncmp(&output_path[len - sizeof(GRID_FILE) + 1], GRID_FILE, sizeof(GRID_FILE)) == 0) {
                cmesh4::SimpleMesh mesh = cmesh4::LoadMeshFromObj(input_path, false);

                LiteMath::float4 max = mesh.vPos4f[0];
                LiteMath::float4 min = mesh.vPos4f[0];

                for (const auto& v : mesh.vPos4f) {
                    max = LiteMath::max(max, v);
                    min = LiteMath::min(min, v);
                }

                LiteMath::float4 center = (max + min) * 0.5f;
                float k = std::max(max.x - min.x, std::max(max.y - min.y, max.z - min.z)) * 0.75f;

                std::vector<float> grid(size * size * size, -10000.0f);
                std::vector<bool> frozen(size * size * size, false);

                std::vector<int> inside(size * size * size, false);
                std::vector<bool> par(size * size * size, false);

                for (unsigned i = 0; i < mesh.IndicesNum(); i += 3) {
                    LiteMath::float4 a = (mesh.vPos4f[mesh.indices[i + 0]] - center) / k;
                    LiteMath::float4 b = (mesh.vPos4f[mesh.indices[i + 1]] - center) / k;
                    LiteMath::float4 c = (mesh.vPos4f[mesh.indices[i + 2]] - center) / k;
                    a.w = 1.0f;
                    b.w = 1.0f;
                    c.w = 1.0f;
                    LiteMath::float4 arr[]{ a, b, c };

                    LiteMath::float4 pa = (a + LiteMath::float4(1.0f, 1.0f, 1.0f, 0.0f)) * 0.5f * (size - 1);
                    LiteMath::float4 pb = (b + LiteMath::float4(1.0f, 1.0f, 1.0f, 0.0f)) * 0.5f * (size - 1);
                    LiteMath::float4 pc = (c + LiteMath::float4(1.0f, 1.0f, 1.0f, 0.0f)) * 0.5f * (size - 1);

                    LiteMath::uint3 ua = LiteMath::uint3(pa.x, pa.y, pa.z);
                    LiteMath::uint3 ub = LiteMath::uint3(pb.x, pb.y, pb.z);
                    LiteMath::uint3 uc = LiteMath::uint3(pc.x, pc.y, pc.z);

                    LiteMath::uint3 min = LiteMath::min(ua, LiteMath::min(ub, uc)) - LiteMath::uint3(1, 1, 1) * 1;
                    LiteMath::uint3 max = LiteMath::max(ua, LiteMath::max(ub, uc)) + LiteMath::uint3(1, 1, 1) * 2;

                    LiteMath::float4 nor = LiteMath::cross(pb - pa, pc - pa);
                    float delta = 2.0f * 2.0001f / (size - 1);
                    float epsilon = 0.0001f / (size - 1);

                    
                    for (unsigned x = min.x; x < max.x; x++) {
                        for (unsigned y = min.y; y < max.y; y++) {
                            for (unsigned z = min.z; z < max.z; z++) {
                                LiteMath::float4 p = LiteMath::float4(x, y, z, 1.0f);
                                inside[x + y * size + z * size * size] = LiteMath::dot(p - pa, nor) < 0.0f ? -1 : (LiteMath::dot(p - pa, nor) > 0.0f ? 1 : 0);
                                par[x + y * size + z * size * size] = isInsideTriangle(p, pa, pb, pc);
                            }
                        }
                    }

                    for (unsigned x = min.x; x < max.x; x++) {
                        for (unsigned y = min.y; y < max.y; y++) {
                            for (unsigned z = min.z; z < max.z; z++) {
                                LiteMath::float4 p = 2.0f * LiteMath::float4(x, y, z, 0.0f) / (size - 1) - LiteMath::float4(1.0f, 1.0f, 1.0, 0.0f);
                                p.w = 1.0f;
                                unsigned index = x + y * size + z * size * size;
                                float da = distance(p, arr);
                                
                                if (std::abs(da) <= delta) {
                                    frozen[index] = true;
                                    if (grid[index] < 0.0f && da >= 0.0f) {
                                        if (std::abs(grid[index]) + epsilon > std::abs(da)) {
                                            grid[index] = da;
                                        }
                                    } else if (grid[index] >= 0.0f && da < 0.0f) {
                                        if (std::abs(grid[index]) > std::abs(da) + epsilon) {
                                            grid[index] = da;
                                        }
                                    } else if (std::abs(grid[index]) > std::abs(da)) {
                                        grid[index] = da;
                                    }
                                }
                            }
                        }
                    }
                }

                fastSweep(grid, frozen, LiteMath::int3(size, size, size));

                SdfGrid sdfgrid{
                    .size = LiteMath::uint3(size, size, size),
                    .data = grid
                };

                save_sdf_grid(sdfgrid, output_path);
            } else {
                cmesh4::SimpleMesh mesh = cmesh4::LoadMeshFromObj(input_path, false);

                LiteMath::float4 max = mesh.vPos4f[0];
                LiteMath::float4 min = mesh.vPos4f[0];

                for (const auto& v : mesh.vPos4f) {
                    max = LiteMath::max(max, v);
                    min = LiteMath::min(min, v);
                }

                LiteMath::float4 center = (max + min) * 0.5f;
                float k = std::max(max.x - min.x, std::max(max.y - min.y, max.z - min.z)) * 0.75f;

                std::vector<LiteMath::float4> vertices(mesh.VerticesNum());
                for (size_t i = 0; i < mesh.VerticesNum(); i++) {
                    vertices[i] = (mesh.vPos4f[i] - center) / k;
                }

                std::vector<SdfOctreeNode> nodes;
                nodes.resize(1);

                std::vector<unsigned> triangles(mesh.TrianglesNum());
                for (size_t i = 0; i < mesh.TrianglesNum(); i++) {
                    triangles[i] = i * 3;
                }

                splitOctree(nodes, vertices, mesh.indices, triangles, 0, LiteMath::float4(-1.0f, -1.0f, -1.0f, 1.0f), LiteMath::float4(1.0f, 1.0f, 1.0f, 1.0f), depth);

                std::cout << nodes.size() << std::endl;

                SdfOctree sdfoctree{
                    .nodes = nodes
                };

                save_sdf_octree(sdfoctree, output_path);
            }
        } else if (strncmp(&input_path[len - sizeof(GRID_FILE) + 1], GRID_FILE, sizeof(GRID_FILE)) == 0) {
            SdfGrid grid;

            load_sdf_grid(grid, input_path);

            for (unsigned x = 0; x < grid.size.x; x++) {
                for (unsigned y = 0; y < grid.size.y; y++) {
                    for (unsigned z = 0; z < grid.size.z; z++) {
                        LiteMath::float4 position{ static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), 0.0f };
                        position.x /= grid.size.x - 1;
                        position.y /= grid.size.y - 1;
                        position.z /= grid.size.z - 1;
                        position = position * 2.0f - LiteMath::float4(1.0f, 1.0f, 1.0f, 0.0f);
                        position.w = 1.0f;

                        for (const auto& s : spheres) {
                            float d = -sdSphere(position, s);
                            if (d > grid.data[x + y * grid.size.x + z * grid.size.x * grid.size.y]) {
                                grid.data[x + y * grid.size.x + z * grid.size.x * grid.size.y] = d;
                            }
                        }
                    }
                }
            }

            SdfGrid sdfgrid{
                .size = grid.size,
                .data = grid.data
            };

            save_sdf_grid(sdfgrid, output_path);
        } else if (strncmp(&input_path[len - sizeof(OCTREE_FILE) + 1], OCTREE_FILE, sizeof(OCTREE_FILE)) == 0) {
            SdfOctree octree;

            load_sdf_octree(octree, input_path);

            SdfOctree sdfoctree;

            sdfoctree.nodes.resize(1);

            sphereCutter(octree, sdfoctree, spheres[0], 0, 0, LiteMath::float4(-1.0f, -1.0f, -1.0f, 1.0f), LiteMath::float4(1.0f, 1.0f, 1.0f, 1.0f));

            save_sdf_octree(sdfoctree, output_path);
        }
    } else if (strncmp(&input_path[len - sizeof(OBJ_FILE) + 1], OBJ_FILE, sizeof(OBJ_FILE)) == 0) {
        PreMesh mesh{
            .mesh = cmesh4::LoadMeshFromObj(input_path, false),
        };

        Application app{ { mesh } };

        try {
            app.run();
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
            return EXIT_FAILURE;
        }
    } else if (strncmp(&input_path[len - sizeof(GRID_FILE) + 1], GRID_FILE, sizeof(GRID_FILE)) == 0) {
        Grid grid{
            .name = input_path
        };

        Application app{ { grid } };
        try {
            app.run();
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
            return EXIT_FAILURE;
        }
    } else if (strncmp(&input_path[len - sizeof(OCTREE_FILE) + 1], OCTREE_FILE, sizeof(OCTREE_FILE)) == 0) {
        Octree octree{
            .name = input_path
        };

        Application app{ { octree } };

        try {
            app.run();
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
            return EXIT_FAILURE;
        }
    } else if (strncmp(&input_path[len - sizeof(SCENE_FILE) + 1], SCENE_FILE, sizeof(SCENE_FILE)) == 0) {
        std::ifstream file(input_path, std::ios::in);

        std::vector<Primitive> primitives;
        std::vector<PreMesh> meshes;
        std::vector<Light> lights;
        std::vector<Grid> grids;
        std::vector<Octree> octrees;

        std::string str;

        constexpr unsigned NONE = 0;
        constexpr unsigned PRIMITIVE = 1;
        constexpr unsigned MESH = 2;
        constexpr unsigned GRID = 3;
        constexpr unsigned OCTREE = 4;
        constexpr unsigned LIGHT = 5;

        unsigned lastType = NONE;

        while (file >> str) {
            if (str.compare("PLANE") == 0) {
                float a, b, c, d;
                file >> a >> b >> c >> d;
                primitives.push_back({
                    .data = LiteMath::float4(a, b, c, d),
                    .type = 0
                });

                lastType = PRIMITIVE;
            } else if (str.compare("SPHERE") == 0) {
                float a, b, c, d;
                file >> a >> b >> c >> d;
                primitives.push_back({
                    .data = LiteMath::float4(a, b, c, d),
                    .type = 1
                });

                lastType = PRIMITIVE;
            } else if (str.compare("CUBE") == 0) {
                float a, b, c, d;
                file >> a >> b >> c >> d;
                primitives.push_back({
                    .data = LiteMath::float4(a, b, c, d),
                    .type = 2
                });

                lastType = PRIMITIVE;
            } else if (str.compare("MESH") == 0) {
                str = readString(file);
                meshes.push_back({
                    .mesh = cmesh4::LoadMeshFromObj(str.c_str(), false)
                });

                lastType = MESH;
            } else if (str.compare("DIRECTIONAL") == 0) {
                file >> str;
                float x, y, z;
                file >> x >> y >> z;

                lights.push_back(Light::Directional(LiteMath::float3(x, y, z)));

                lastType = LIGHT;
            } else if (str.compare("POINT") == 0) {
                file >> str;
                float x, y, z;
                file >> x >> y >> z;

                lights.push_back(Light::Point(LiteMath::float3(x, y, z)));

                lastType = LIGHT;
            } else if (str.compare("COLOR") == 0) {
                float r, g, b;
                file >> r >> g >> b;

                if (lastType == PRIMITIVE) {
                    primitives.back().material.color = LiteMath::float4(r, g, b, 1.0f);
                } else if (lastType == MESH) {
                    meshes.back().material.color = LiteMath::float4(r, g, b, 1.0f);
                } else if (lastType == GRID) {
                    grids.back().material.color = LiteMath::float4(r, g, b, 1.0f);
                } else if (lastType == OCTREE) {
                    octrees.back().material.color = LiteMath::float4(r, g, b, 1.0f);
                } else if (lastType == LIGHT) {
                    lights.back().color = LiteMath::float4(r, g, b, 1.0f);
                }
            } else if (str.compare("REFLECTIVE") == 0) {
                float r;
                file >> r;
                if (lastType == PRIMITIVE) {
                    primitives.back().material.reflective = r;
                }else if (lastType == MESH) {
                    meshes.back().material.reflective = r;
                } else if (lastType == GRID) {
                    grids.back().material.reflective = r;
                } else if (lastType == OCTREE) {
                    octrees.back().material.reflective = r;
                }
            } else if (str.compare("REFRACTIV") == 0) {
                float r;
                file >> r;
                if (lastType == PRIMITIVE) {
                    primitives.back().material.refractiv = r;
                }else if (lastType == MESH) {
                    meshes.back().material.refractiv = r;
                } else if (lastType == GRID) {
                    grids.back().material.refractiv = r;
                } else if (lastType == OCTREE) {
                    octrees.back().material.refractiv = r;
                }
            } else if (str.compare("INTENSITY") == 0) {
                float i;
                file >> i;
                lights.back().color *= i;
            } else if (str.compare("SCALE") == 0) {
                float x, y, z;
                file >> x >> y >> z;
                if (lastType == MESH) {
                    meshes.back().modelMatrix = meshes.back().modelMatrix * LiteMath::scale4x4(LiteMath::float3(x, y, z));
                } else if (lastType == GRID) {
                    grids.back().inverseModel = LiteMath::scale4x4(LiteMath::float3(1.0f / x, 1.0f / y, 1.0f / z)) * grids.back().inverseModel;
                } else if (lastType == OCTREE) {
                    octrees.back().inverseModel = LiteMath::scale4x4(LiteMath::float3(1.0f / x, 1.0f / y, 1.0f / z)) * octrees.back().inverseModel;
                }
            } else if (str.compare("POSITION") == 0) {
                float x, y, z;
                file >> x >> y >> z;
                if (lastType == MESH) {
                    meshes.back().modelMatrix = meshes.back().modelMatrix * LiteMath::translate4x4(LiteMath::float3(x, y, z));
                } else if (lastType == GRID) {
                    grids.back().inverseModel = LiteMath::translate4x4(LiteMath::float3(-x, -y, -z)) * grids.back().inverseModel;
                } else if (lastType == OCTREE) {
                    octrees.back().inverseModel = LiteMath::translate4x4(LiteMath::float3(-x, -y, -z)) * octrees.back().inverseModel;
                }
            } else if (str.compare("ROTATION") == 0) {
                float x, y, z;
                file >> x >> y >> z;
                if (lastType == MESH) {
                    meshes.back().modelMatrix = meshes.back().modelMatrix * LiteMath::rotate4x4X(x) * LiteMath::rotate4x4Y(y) * LiteMath::rotate4x4Z(z);
                } else if (lastType == GRID) {
                    grids.back().inverseModel = LiteMath::rotate4x4Z(-x) * LiteMath::rotate4x4Y(-y) * LiteMath::rotate4x4X(-z) * grids.back().inverseModel;
                } else if (lastType == OCTREE) {
                    octrees.back().inverseModel = LiteMath::rotate4x4Z(-x) * LiteMath::rotate4x4Y(-y) * LiteMath::rotate4x4X(-z) * octrees.back().inverseModel;
                }
            } else if (str.compare("GRID") == 0) {
                str = readString(file);
                grids.push_back({
                    .name = str
                });

                lastType = GRID;
            } else if (str.compare("OCTREE") == 0) {
                str = readString(file);
                lastType = OCTREE;
            }
        }

        Application app{ primitives, meshes, grids, octrees, lights };

        try {
            app.run();
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}