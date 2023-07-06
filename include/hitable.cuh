#ifndef HITABLEH
#define HITABLEH

#include "ray.cuh"
#include "common.cuh"

class Material;

struct HitRecord
{
    float t;
    float u;
    float v;
    Vec3 p;
    Vec3 normal;
    Material *mat_ptr;
    bool front_face;
    
    __device__ inline void set_face_normal(const ray& r, const Vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
    }
};

class Hitable  {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
};

#endif
