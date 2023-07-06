#ifndef RAYH
#define RAYH
#include "vec3.cuh"

class ray
{
    public:
        __device__ ray() {}
        __device__ ray(const Vec3& a, const Vec3& b, float time = 0.0f) { 
            orig = a; 
            dir = b;
            tm = time;
        }
        __device__ Vec3 origin() const       { return orig; }
        __device__ Vec3 direction() const    { return dir; }
        __device__ Vec3 point_at_parameter(float t) const { return orig + t*dir; }
        __device__ float time() const    { return tm; }

        Vec3 orig;
        Vec3 dir;
        float tm;
};

#endif
