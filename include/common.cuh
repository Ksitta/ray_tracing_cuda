
#ifndef __COMMON_CUH__
#define __COMMON_CUH__

#include <cuda_runtime.h>
#include "vec3.cuh"

const float PI = 3.14159265358979f;
const float eps = 1e-6f;
const float inf = 1e20f;
const float REACTOR = 1.5f;

__device__ __host__ inline float clamp(float x, float a = 0, float b = 1) {
    if(x != x) return 0;
    return x < a ? a : (x > b ? b : x);
}

__device__ __host__ inline Vec3 rotateY(float theta, const Vec3& p){
    float sin_theta = sin(theta);
    float cos_theta = cos(theta);
    return Vec3(cos_theta * p.x() - sin_theta * p.z(), p.y(), sin_theta * p.x() + cos_theta * p.z());
}

__device__ inline Vec3 tangent_normal_disk(float theta, float R){
    return R * Vec3(-sin(theta), 0, cos(theta));
}

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

#endif