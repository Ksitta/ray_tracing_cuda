
#ifndef __COMMON_CUH__
#define __COMMON_CUH__

#include <cuda_runtime.h>
#include "vec3.cuh"

const float PI = 3.14159265358979f;
const float eps = 1e-6f;
const float inf = 1e20f;
const float REACTOR = 1.5f;

__device__ __host__ inline float clamp(float x, float a = 0, float b = 1) {
    return x < a ? a : (x > b ? b : x);
}

__device__ __host__ inline vec3 rotate(const vec3& pole, float theta, const vec3& p){
    float dz = p.z() - pole.z();
    float dx = p.x() - pole.x();
    float d = sqrt(dz * dz + dx * dx);
    float newz = pole.z() + d * cos(theta);
    float newx = pole.x() + d * sin(theta);
    return vec3(newx, p.y(), newz);
}


__device__ inline vec3 tangent_at_disk(float theta, float R){
    return R * vec3(cos(theta), 0, -sin(theta));
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