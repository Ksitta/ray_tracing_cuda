#ifndef MESH_H
#define MESH_H

#include "common.cuh"
#include "vec3.cuh"
#include "hitable.cuh"
#include "material.cuh"
#include "triangle.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

class Mesh : public Hitable {

public:
    Triangle *triangles;
    int num_triangles;
    Material *mat_ptr;

    __device__ Mesh(Triangle *trians, int n, Material *m){
        triangles = new Triangle[n];
        num_triangles = n;
        mat_ptr = m;
        for (int i = 0; i < num_triangles; i++){
            triangles[i].set(trians[i], mat_ptr);
        }
    }

    __device__ bool hit(const ray& r, float t_min, float t_max, HitRecord& rec) const {
        HitRecord temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;

        for(int i = 0; i < num_triangles; i++){
            if(triangles[i].hit(r, t_min, closest_so_far, temp_rec)){
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
                // rec.set_face_normal(r, vec3(0, 1, 0));
            }
        }

        return hit_anything;
    }

private:

};

inline void read_mesh(std::string filename, Triangle **d_triangles, int *num_triangles){
    struct TriangleIndex {
        TriangleIndex() {
            x[0] = 0; x[1] = 0; x[2] = 0;
        }
        int &operator[](const int i) { return x[i]; }
        // By Computer Graphics convention, counterclockwise winding is front face
        int x[3]{};
    };

    std::vector<Vec3> v;
    std::vector<TriangleIndex> t;
    std::vector<Vec3> n;
    std::vector<Triangle> triangles;

    std::ifstream f;
    f.open(filename);
    if (!f.is_open()) {
        std::cout << "Cannot open " << filename << "\n";
        return;
    }
    std::string line;
    std::string vTok("v");
    std::string fTok("f");
    std::string texTok("vt");
    char bslash = '/', space = ' ';
    std::string tok;
    int texID;
    while (true) {
        std::getline(f, line);
        if (f.eof()) {
            break;
        }
        if (line.size() < 3) {
            continue;
        }
        if (line.at(0) == '#') {
            continue;
        }
        std::stringstream ss(line);
        ss >> tok;
        if (tok == vTok) {
            Vec3 vec;
            ss >> vec[0] >> vec[1] >> vec[2];
            v.push_back(vec);
        } else if (tok == fTok) {
            if (line.find(bslash) != std::string::npos) {
                std::replace(line.begin(), line.end(), bslash, space);
                std::stringstream facess(line);
                TriangleIndex trig;
                facess >> tok;
                for (int ii = 0; ii < 3; ii++) {
                    facess >> trig[ii] >> texID;
                    trig[ii]--;
                }
                t.push_back(trig);
            } else {
                TriangleIndex trig;
                for (int ii = 0; ii < 3; ii++) {
                    ss >> trig[ii];
                    trig[ii]--;
                }
                t.push_back(trig);
            }
        } else if (tok == texTok) {
            float texcoord[2];
            ss >> texcoord[0];
            ss >> texcoord[1];
        }
    }
    for (int triId = 0; triId < (int) t.size(); ++triId) {
        TriangleIndex triIndex = t[triId];
        triangles.emplace_back(v[triIndex[0]],v[triIndex[1]], v[triIndex[2]], nullptr);
    }
    f.close();
    checkCudaErrors(cudaMalloc((void**)d_triangles, triangles.size() * sizeof(Triangle)));
    checkCudaErrors(cudaMemcpy(*d_triangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    *num_triangles = triangles.size();
}

#endif
