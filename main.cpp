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

float udTriangle( LiteMath::float4 p, LiteMath::float4 a, LiteMath::float4 b, LiteMath::float4 c )
{
    LiteMath::float4 ba = b - a; 
    LiteMath::float4 pa = p - a;
    LiteMath::float4 cb = c - b; 
    LiteMath::float4 pb = p - b;
    LiteMath::float4 ac = a - c; 
    LiteMath::float4 pc = p - c;
    LiteMath::float4 nor = LiteMath::cross( ba, ac );

//   return sqrt(
//     (LiteMath::sign(LiteMath::dot(LiteMath::cross(ba,nor),pa)) +
//     LiteMath::sign(LiteMath::dot(LiteMath::cross(cb,nor),pb)) +
//     LiteMath::sign(LiteMath::dot(LiteMath::cross(ac,nor),pc))<2.0)
//      ?
//      LiteMath::min( LiteMath::min(
//         LiteMath::dot(ba*LiteMath::clamp(LiteMath::dot(ba,pa)/LiteMath::dot(ba, ba),0.0f,1.0f)-pa,ba*LiteMath::clamp(LiteMath::dot(ba,pa)/LiteMath::dot(ba, ba),0.0f,1.0f)-pa),
//         LiteMath::dot(cb*LiteMath::clamp(LiteMath::dot(cb,pb)/LiteMath::dot(cb, cb),0.0f,1.0f)-pb,cb*LiteMath::clamp(LiteMath::dot(cb,pb)/LiteMath::dot(cb, cb),0.0f,1.0f)-pb)),
//         LiteMath::dot(ac*LiteMath::clamp(LiteMath::dot(ac,pc)/LiteMath::dot(ac, ac),0.0f,1.0f)-pc,ac*LiteMath::clamp(LiteMath::dot(ac,pc)/LiteMath::dot(ac, ac),0.0f,1.0f)-pc))
//      :
//      LiteMath::dot(nor,pa)*LiteMath::dot(nor,pa)/LiteMath::dot(nor, nor));
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
    // LiteMath::float4 pt = vertices[1];
    // float minDist = LiteMath::length(point - pt);
    // for (int i = 1; i < 3; i++) {
    //     float dist = LiteMath::length(point - vertices[i]);
    //     if (dist < minDist) {
    //         pt = vertices[i];
    //         minDist = dist;
    //     }
    // }
    // return LiteMath::sign(LiteMath::dot(LiteMath::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]), point - pt)) * minDist;
    // return LiteMath::sign(LiteMath::dot(LiteMath::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]), point - pt)) * udTriangle(point, vertices[0], vertices[1], vertices[2]);
    return udTriangle(point, vertices[0], vertices[1], vertices[2]);
}

float sdSphere(LiteMath::float4 point, LiteMath::float4 sphere) {
    LiteMath::float3 p = LiteMath::float3(point.x - sphere.x, point.y - sphere.y, point.z - sphere.z);
    return LiteMath::length(p) - sphere.w;
}

// void calculate_block(LiteMath::uint3 start, LiteMath::uint3 end, LiteMath::uint3 size, LiteMath::float4 center, float k, const cmesh4::SimpleMesh& mesh, std::vector<float>& grid) {
//     for (unsigned x = start.x; x < end.x && x < size.x; x++) {
//         for (unsigned y = start.y; y < end.y && y < size.y; y++) {
//             for (unsigned z = start.z; z < end.z && z < size.z; z++) {
//                 LiteMath::float4 position{ static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), 0.0f };
//                 position.x /= size.x - 1;
//                 position.y /= size.y - 1;
//                 position.z /= size.z - 1;
//                 position = position * 2.0f - LiteMath::float4(1.0f, 1.0f, 1.0f, 0.0f);
//                 position.w = 1.0f;
//                 float minDist = 1000000000000000.0f;
//                 for (unsigned i = 0; i < mesh.IndicesNum(); i += 3) {
//                     LiteMath::float4 a = mesh.vPos4f[mesh.indices[i + 0]];
//                     LiteMath::float4 b = mesh.vPos4f[mesh.indices[i + 1]];
//                     LiteMath::float4 c = mesh.vPos4f[mesh.indices[i + 2]];
//                     a -= center;
//                     b -= center;
//                     c -= center;
//                     a /= k;
//                     b /= k;
//                     c /= k;
//                     a.w = 1.0f;
//                     b.w = 1.0f;
//                     c.w = 1.0f;
//                     LiteMath::float4 arr[] = { a, b, c };
//                     float d = distance(position, arr);
//                     if (std::abs(d) < std::abs(minDist)) {
//                         minDist = d;
//                     }
//                 }
//                 grid[z + y * size.z + x * size.z * size.y] = minDist;
//             }
//         }
//     }
// }

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
    // sweep directions { start, end, step }
    const int dirX[NSweeps][3] = { {size.x - 1, 0, -1}, {0, size.x - 1, 1}, {0, size.x - 1, 1}, {size.x - 1, 0, -1}, {size.x - 1, 0, -1}, {0, size.x - 1, 1}, {0, size.x - 1, 1}, {size.x - 1, 0, -1} };
    const int dirY[NSweeps][3] = { {size.y - 1, 0, -1}, {size.y - 1, 0, -1}, {0, size.y - 1, 1}, {0, size.y - 1, 1}, {size.y - 1, 0, -1}, {size.y - 1, 0, -1}, {0, size.y - 1, 1}, {0, size.y - 1, 1} };
    const int dirZ[NSweeps][3] = { {0, size.z - 1, 1}, {0, size.z - 1, 1}, {0, size.z - 1, 1}, {0, size.z - 1, 1}, {size.z - 1, 0, -1}, {size.z - 1, 0, -1}, {size.z - 1, 0, -1}, {size.z - 1, 0, -1} };
    float aa[3], eps = 1e-6f;
    float d_new, a, b;
    int s, ix, iy, iz, gridPos;
    const float h = 1.0f / (size.x - 1), f = 1.0f;
 
    for (s = 0; s < NSweeps; s++) { // <--------- NSweeps
        for (iz = dirZ[s][0]; dirZ[s][2] * iz <= dirZ[s][1]; iz += dirZ[s][2]) {
            for (iy = dirY[s][0]; dirY[s][2] * iy <= dirY[s][1]; iy += dirY[s][2]) {
                for (ix = dirX[s][0]; dirX[s][2] * ix <= dirX[s][1]; ix += dirX[s][2]) {
    
                    gridPos = ix + iy * size.x + iz * size.x * size.y;
    
                    if (!frozen[gridPos]) {
    
                        // === neighboring cells (Upwind Godunov) ===
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
                
                        float d_curr = aa[0] + h * f * (aa[0] > 0.0f ? 1.0f : -1.0f); // just a linear equation with the first neighbor value
                        if (std::abs(d_curr) <= std::abs(aa[1]) + eps || aa[0] * aa[1] < 0.0f) {
                            d_new = d_curr; // accept the solution
                        } else {
                            // quadratic equation with coefficients involving 2 neighbor values aa
                            float a = 2.0f; 
                            float b = -2.0f * (aa[0] + aa[1]);
                            float c = aa[0] * aa[0] + aa[1] * aa[1] - h * h * f * f;
                            float D = sqrtf(b * b - 4.0f * a * c);
                            // choose the minimal root
                            d_curr = maxabs(-b + D, -b - D) / (2.0f * a);
                
                            if (std::abs(d_curr) <= std::abs(aa[2]) + eps || aa[1] * aa[2] < 0.0f) {
                                d_new = d_curr; // accept the solution
                            } else {
                                // quadratic equation with coefficients involving all 3 neighbor values aa
                                a = 3.0f;
                                b = -2.0f * (aa[0] + aa[1] + aa[2]);
                                c = aa[0] * aa[0] + aa[1] * aa[1] + aa[2] * aa[2] - h * h * f * f;
                                D = sqrtf(b * b - 4.0f * a * c);
                                // choose the minimal root
                                d_new = maxabs(-b + D, -b - D) / (2.0f * a);
                            }
                        }
                        // update if d_new is smaller
                        grid[gridPos] = minabs(grid[gridPos], d_new);
                    }
                }
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

    if (argc == 1) {
        std::cout << "Please, enter file name" << std::endl;

        return EXIT_SUCCESS;
    }

    const char* input_path = "";
    const char* output_path = nullptr;
    unsigned size = 256;
    std::vector<LiteMath::float4> spheres;

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], OUTPUT, sizeof(OUTPUT)) == 0) {
            output_path = argv[i + 1];
            i++;
        } else if (strncmp(argv[i], SIZE, sizeof(SIZE)) == 0) {
            size = std::atoi(argv[i + 1]);
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

            // std::vector<std::thread> th;
            // constexpr unsigned step = 2;

            // for (unsigned x = 0; x < size; x += step) {
            //     for (unsigned y = 0; y < size; y += step) {
            //         for (unsigned z = 0; z < size; z += step) {
            //             th.push_back(std::thread{ calculate_block, LiteMath::uint3(x, y, z), LiteMath::uint3(x + step, y + step, z + step), LiteMath::uint3(size, size, size), center, k, std::ref(mesh), std::ref(grid) });
            //         }
            //     }
            // }

            // for (auto& t : th) {
            //     t.join();
            // }

            // for (unsigned x = 0; x < size; x++) {
            //     for (unsigned y = 0; y < size; y++) {
            //         for (unsigned z = 0; z < size; z++) {
            //             LiteMath::float4 position{ static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), 0.0f };
            //             position.x /= size - 1;
            //             position.y /= size - 1;
            //             position.z /= size - 1;
            //             position = position * 2.0f - LiteMath::float4(1.0f, 1.0f, 1.0f, 0.0f);
            //             position.w = 1.0f;
            //             float minDist = 1000000000000000.0f;
            //             for (unsigned i = 0; i < mesh.IndicesNum(); i += 3) {
            //                 LiteMath::float4 a = mesh.vPos4f[mesh.indices[i + 0]];
            //                 LiteMath::float4 b = mesh.vPos4f[mesh.indices[i + 1]];
            //                 LiteMath::float4 c = mesh.vPos4f[mesh.indices[i + 2]];
            //                 a -= center;
            //                 b -= center;
            //                 c -= center;
            //                 a /= k;
            //                 b /= k;
            //                 c /= k;
            //                 a.w = 1.0f;
            //                 b.w = 1.0f;
            //                 c.w = 1.0f;
            //                 LiteMath::float4 arr[] = { a, b, c };
            //                 float d = distance(position, arr);
            //                 if (std::abs(d) < std::abs(minDist)) {
            //                     minDist = d;
            //                 }
            //             }
            //             grid[z + y * size + x * size * size] = minDist;
            //         }
            //     }
            // }

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

                // for (unsigned j = 0; j < 8; j++) {
                //     LiteMath::uint3 delta = LiteMath::uint3(j % 2, (j >> 1) % 2, j >> 2);
                //     LiteMath::float4 offset = LiteMath::float4(j % 2, (j >> 1) % 2, j >> 2, 0.0f) / (size - 1);
                //     LiteMath::uint3 ca = ua + delta;
                //     LiteMath::uint3 cb = ub + delta;
                //     LiteMath::uint3 cc = uc + delta;
                //     LiteMath::float4 oa = 2 * LiteMath::float4(ca.x, ca.y, ca.z, 0.0f) / (size - 1) - LiteMath::float4(1.0f, 1.0f, 1.0, 0.0f);
                //     LiteMath::float4 ob = 2 * LiteMath::float4(cb.x, cb.y, cb.z, 0.0f) / (size - 1) - LiteMath::float4(1.0f, 1.0f, 1.0, 0.0f);
                //     LiteMath::float4 oc = 2 * LiteMath::float4(cc.x, cc.y, cc.z, 0.0f) / (size - 1) - LiteMath::float4(1.0f, 1.0f, 1.0, 0.0f);
                //     oa.w = 1.0f;
                //     ob.w = 1.0f;
                //     oc.w = 1.0f;
                //     float da = distance(oa, arr);
                //     float db = distance(ob, arr);
                //     float dc = distance(oc, arr);
                //     frozen[ca.z + ca.y * size + ca.x * size * size] = true;
                //     if (std::abs(grid[ca.z + ca.y * size + ca.x * size * size]) > std::abs(da)) {
                //         grid[ca.z + ca.y * size + ca.x * size * size] = da;
                //     }
                //     frozen[cb.z + cb.y * size + cb.x * size * size] = true;
                //     if (std::abs(grid[cb.z + cb.y * size + cb.x * size * size]) > std::abs(db)) {
                //         grid[cb.z + cb.y * size + cb.x * size * size] = db;
                //     }
                //     frozen[cc.z + cc.y * size + cc.x * size * size] = true;
                //     if (std::abs(grid[cc.z + cc.y * size + cc.x * size * size]) > std::abs(dc)) {
                //         grid[cc.z + cc.y * size + cc.x * size * size] = dc;
                //     }
                // }

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
                            bool flg = false;
                            // if (x > min.x && LiteMath::dot(p - a, nor) * LiteMath::dot(p - a - LiteMath::float4(delta, 0.0f, 0.0f, 0.0f), nor) <= epsilon && (isInsideTriangle(p, arr) || isInsideTriangle(p - LiteMath::float4(delta, 0.0f, 0.0f, 0.0f), arr))) {
                            //     flg = true;
                            // } else if (x < max.x - 1 && LiteMath::dot(p - a, nor) * LiteMath::dot(p - a + LiteMath::float4(delta, 0.0f, 0.0f, 0.0f), nor) <= epsilon && (isInsideTriangle(p, arr) || isInsideTriangle(p + LiteMath::float4(delta, 0.0f, 0.0f, 0.0f), arr))) {
                            //     flg = true;
                            // } else if (y > min.y && LiteMath::dot(p - a, nor) * LiteMath::dot(p - a - LiteMath::float4(0.0f, delta, 0.0f, 0.0f), nor) <= epsilon && (isInsideTriangle(p, arr) || isInsideTriangle(p - LiteMath::float4(0.0f, delta, 0.0f, 0.0f), arr))) {
                            //     flg = true;
                            // } else if (y < max.y - 1 && LiteMath::dot(p - a, nor) * LiteMath::dot(p - a + LiteMath::float4(0.0f, delta, 0.0f, 0.0f), nor) <= epsilon && (isInsideTriangle(p, arr) || isInsideTriangle(p + LiteMath::float4(0.0f, delta, 0.0f, 0.0f), arr))) {
                            //     flg = true;
                            // } else if (z > min.z && LiteMath::dot(p - a, nor) * LiteMath::dot(p - a - LiteMath::float4(0.0f, 0.0f, delta, 0.0f), nor) <= epsilon && (isInsideTriangle(p, arr) || isInsideTriangle(p - LiteMath::float4(0.0f, 0.0f, delta, 0.0f), arr))) {
                            //     flg = true;
                            // } else if (z < max.z - 1 && LiteMath::dot(p - a, nor) * LiteMath::dot(p - a + LiteMath::float4(0.0f, 0.0f, delta, 0.0f), nor) <= epsilon && (isInsideTriangle(p, arr) || isInsideTriangle(p + LiteMath::float4(0.0f, 0.0f, delta, 0.0f), arr))) {
                            //     flg = true;
                            // }
                            unsigned index = x + y * size + z * size * size;
                            // if (x == max.x - 1) {
                            //     if (inside[index] != inside[index - 1] && (par[index] || par[index - 1])) {
                            //         flg = true;
                            //     }
                            // } else if (x == min.x) {
                            //     if (inside[index] != inside[index + 1] && (par[index] || par[index + 1])) {
                            //         flg = true;
                            //     }
                            // } else {
                            //     if (inside[index + 1] != inside[index - 1] && (par[index + 1] || par[index - 1])) {
                            //         flg = true;
                            //     }
                            // }

                            // if (y == max.y - 1) {
                            //     if (inside[index] != inside[index - size] && (par[index] || par[index - size])) {
                            //         flg = true;
                            //     }
                            // } else if (y == min.y) {
                            //     if (inside[index] != inside[index + size] && (par[index] || par[index + size])) {
                            //         flg = true;
                            //     }
                            // } else {
                            //     if (inside[index + size] != inside[index - size] && (par[index + size] || par[index - size])) {
                            //         flg = true;
                            //     }
                            // }

                            // if (z == max.z - 1) {
                            //     if (inside[index] != inside[index - size * size] && (par[index] || par[index - size * size])) {
                            //         flg = true;
                            //     }
                            // } else if (z == min.z) {
                            //     if (inside[index] != inside[index + size * size] && (par[index] || par[index + size * size])) {
                            //         flg = true;
                            //     }
                            // } else {
                            //     if (inside[index + size * size] != inside[index - size * size] && (par[index + size * size] || par[index - size * size])) {
                            //         flg = true;
                            //     }
                            // }

                            // if (x > min.x && inside[index] != inside[index - 1] && (par[index] || par[index - 1])) {
                            //     //grid[index - 1] = -1.0f;
                            //     flg = true;
                            // }
                            // if (x < max.x - 1 && inside[index] != inside[index + 1] && (par[index] || par[index + 1])) {
                            //     //grid[index + 1] = -1.0f;
                            //     flg = true;
                            // }

                            // if (y > min.y && inside[index] != inside[index - size] && (par[index] || par[index - size])) {
                            //     //grid[index - size] = -1.0f;
                            //     flg = true;
                            // }
                            // if (y < max.y - 1 && inside[index] != inside[index + size] && (par[index] || par[index + size])) {
                            //     //grid[index + size] = -1.0f;
                            //     flg = true;
                            // }

                            // if (z > min.z && inside[index] != inside[index - size * size] && (par[index] || par[index - size * size])) {
                            //     //grid[index - size * size] = -1.0f;
                            //     flg = true;
                            // }
                            // if (z < max.z - 1 && inside[index] != inside[index + size * size] && (par[index] || par[index + size * size])) {
                            //     //grid[index + size * size] = -1.0f;
                            //     flg = true;
                            // }

                            // if (x > min.x + 1 && inside[index] != inside[index - 2] && (par[index] || par[index - 2])) {
                            //     //grid[index - 1] = -1.0f;
                            //     flg = true;
                            // }
                            // if (x < max.x - 2 && inside[index] != inside[index + 2] && (par[index] || par[index + 2])) {
                            //     //grid[index + 1] = -1.0f;
                            //     flg = true;
                            // }

                            // if (y > min.y + 1 && inside[index] != inside[index - 2 * size] && (par[index] || par[index - 2 * size])) {
                            //     //grid[index - size] = -1.0f;
                            //     flg = true;
                            // }
                            // if (y < max.y - 2 && inside[index] != inside[index + 2 * size] && (par[index] || par[index + 2 * size])) {
                            //     //grid[index + size] = -1.0f;
                            //     flg = true;
                            // }

                            // if (z > min.z + 1 && inside[index] != inside[index - 2 * size * size] && (par[index] || par[index - 2 * size * size])) {
                            //     //grid[index - size * size] = -1.0f;
                            //     flg = true;
                            // }
                            // if (z < max.z - 2 && inside[index] != inside[index + 2 * size * size] && (par[index] || par[index + 2 * size * size])) {
                            //     //grid[index + size * size] = -1.0f;
                            //     flg = true;
                            // }

                            // if (y == max.y - 1 && inside[index] != inside[index - size] && (par[index] || par[index - size])) {
                            //     flg = true;
                            // } else if (y == min.y && inside[index] != inside[index + size] && (par[index] || par[index + size])) {
                            //     flg = true;
                            // }
                            
                            // if (z == max.z - 1 && inside[index] != inside[index - size * size] && (par[index] || par[index - size * size])) {
                            //     flg = true;
                            // } else if (z == min.z && inside[index] != inside[index + size * size] && (par[index] || par[index + size * size])) {
                            //     flg = true;
                            // }

                            // if (x > min.x && isInsideTriangle(p, arr) != isInsideTriangle(p - LiteMath::float4(delta, 0.0f, 0.0f, 0.0f), arr)) {
                            //     flg = true;
                            // } else if (x < max.x - 1 && isInsideTriangle(p, arr) != isInsideTriangle(p + LiteMath::float4(delta, 0.0f, 0.0f, 0.0f), arr)) {
                            //     flg = true;
                            // } else if (y > min.y && isInsideTriangle(p, arr) != isInsideTriangle(p - LiteMath::float4(0.0f, delta, 0.0f, 0.0f), arr)) {
                            //     flg = true;
                            // } else if (y < max.y - 1 && isInsideTriangle(p, arr) != isInsideTriangle(p + LiteMath::float4(0.0f, delta, 0.0f, 0.0f), arr)) {
                            //     flg = true;
                            // } else if (z > min.z && isInsideTriangle(p, arr) != isInsideTriangle(p - LiteMath::float4(0.0f, 0.0f, delta, 0.0f), arr)) {
                            //     flg = true;
                            // } else if (z < max.z - 1 && isInsideTriangle(p, arr) != isInsideTriangle(p + LiteMath::float4(0.0f, 0.0f, delta, 0.0f), arr)) {
                            //     flg = true;
                            // }

                            //if (flg) {
                                //grid[x + y * size + z * size * size] = -1.0f;
                                float da = distance(p, arr);
                                // if (grid[index] < 0.0f && da > 0.0f && std::abs(grid[index]) >= std::abs(da)) {
                                //     grid[index] = da;
                                // } else if (std::abs(grid[index]) > std::abs(da)) {
                                //     grid[index] = da;
                                // }
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
                                    // if (grid[index] < da) {
                                        grid[index] = da;
                                    }
                                }
                            //}
                        }
                    }
                }

                // for (unsigned x = min.x; x < max.x; x++) {
                //     for (unsigned y = min.y; y < max.y; y++) {
                //         for (unsigned z = min.z; z < max.z; z++) {
                //             bool b = false;
                //             unsigned gridCell = x + y * size + z * size * size;
                //             if (x > min.x && grid[gridCell] * grid[gridCell - 1] <= 0.0f) {
                //                 b = true;
                //             } else if (x < max.x - 1 && grid[gridCell] * grid[gridCell + 1] <= 0.0f) {
                //                 b = true;
                //             } else if (y > min.y && grid[gridCell] * grid[gridCell - size] <= 0.0f) {
                //                 b = true;
                //             } else if (y < max.y - 1 && grid[gridCell] * grid[gridCell + size] <= 0.0f) {
                //                 b = true;
                //             } else if (z > min.z && grid[gridCell] * grid[gridCell - size * size] <= 0.0f) {
                //                 b = true;
                //             } else if (z < max.z - 1 && grid[gridCell] * grid[gridCell + size * size] <= 0.0f) {
                //                 b = true;
                //             }
                //             frozen[gridCell] = b;
                //         }
                //     }
                // }
            }

            fastSweep(grid, frozen, LiteMath::int3(size, size, size));

            SdfGrid sdfgrid{
                .size = LiteMath::uint3(size, size, size),
                .data = grid
            };

            save_sdf_grid(sdfgrid, output_path);
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
            /// ...
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

        bool isSphere = false;
        std::string str;
        file >> str;
        if (str.compare("SPHERETRACING") == 0) {
            isSphere = true;
        }

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

        if (isSphere) {
            Application app{ primitives, grids, octrees, lights };

            try {
                app.run();
            } catch (const std::exception &e) {
                std::cerr << e.what() << std::endl;
                return EXIT_FAILURE;
            }
        } else {
            Application app{ primitives, meshes, lights };

            try {
                app.run();
            } catch (const std::exception &e) {
                std::cerr << e.what() << std::endl;
                return EXIT_FAILURE;
            }
        }
    }

    // std::vector<Primitive> primitives{ 
    //     {
    //         .data = LiteMath::float4(0.0f, 1.0f, 0.0f, 1.0f),
    //         .type = 0,
    //         .material = {LiteMath::float4(0.0f, 1.0f, 0.0f, 1.0f), 0.0f, 0.0f}
    //     },
    //     {
    //         .data = LiteMath::float4(-1.0f, 0.0f, 0.0f, 1.0f),
    //         .type = 0,
    //         .material = {LiteMath::float4(1.0f, 0.0f, 0.0f, 1.0f), 0.9f, 0.0f}
    //     },
    //     {
    //         .data = LiteMath::float4(0.0f, 0.0f, 1.0f, 0.5f),
    //         .type = 2,
    //         .material = {LiteMath::float4(1.0f, 0.0f, 1.0f, 1.0f), 0.0f, 0.0f}
    //     }
    // };

    // std::vector<PreMesh> meshes{
    //     {
    //         .mesh = cmesh4::LoadMeshFromObj("./data/stanford-bunny.obj", false),
    //         .material = {LiteMath::float4(0.0f, 0.0f, 0.0f, 1.0f), 0.9f, 0.0f},
    //         .modelMatrix = LiteMath::scale4x4(LiteMath::float3(10.0f, 10.0f, 10.0f))
    //     }//,
    //     // {
    //     //     .mesh = cmesh4::LoadMeshFromObj("./data/spot.obj", false),
    //     //     .material = {LiteMath::float4(0.0f, 1.0f, 1.0f, 1.0f), 0.0f, 0.0f},
    //     //     .modelMatrix = LiteMath::translate4x4(LiteMath::float3(2.0f, 0.0f, 0.0f))
    //     // }
    // };

    // std::vector<Light> lights {
    //     Light::Directional(LiteMath::float3(0.0f, -1.0f, -1.0f)),
    //     Light::Point(LiteMath::float3(0.0f, 0.0f, -1.0f), LiteMath::float3(1.0f, 1.0f, 1.0f), 10.0f)
    // };

    // std::vector<Grid> grids {
    //     {
    //         .name = "data/example_grid_large.grid",
    //         .material = {LiteMath::float4(1.0f, 0.0f, 0.0f, 1.0f), 0.0f, 0.0f},
    //         .inverseModel = LiteMath::float4x4() // Should be inversed
    //     }
    // };

    // if (flags & SPHERE) {
    //     Application app{ primitives, grids, lights };

    //     try {
    //         app.run();
    //     } catch (const std::exception &e) {
    //         std::cerr << e.what() << std::endl;
    //         return EXIT_FAILURE;
    //     }
    // } else {
    //     Application app{ primitives, meshes, lights };

    //     try {
    //         app.run();
    //     } catch (const std::exception &e) {
    //         std::cerr << e.what() << std::endl;
    //         return EXIT_FAILURE;
    //     }
    // }

    return EXIT_SUCCESS;
}