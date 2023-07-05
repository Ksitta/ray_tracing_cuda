#ifndef __CIRCLE_CUH__
#define __CIRCLE_CUH__

#include "hitable.cuh"

class xz_circle : public hitable {
    public:
        __device__ xz_circle() {}

        __device__ xz_circle(float x, float y, float rad, float _k,
            material* mat)
            : x(x), z(z), k(_k), mp(mat), rad(rad) { rad2 = rad * rad; };

        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override{
            float t = (k-r.origin().y()) / r.direction().y();
            if (t < t_min || t > t_max)
                return false;
            float x = r.origin().x() + t*r.direction().x();
            float z = r.origin().z() + t*r.direction().z();
            float dx = x - this->x;
            float dz = z - this->z;
            if (dx * dx + dz * dz > rad2)
                return false;
            rec.t = t;
            vec3 outward_normal = vec3(0, 1, 0);
            rec.set_face_normal(r, outward_normal);
            rec.mat_ptr = mp;
            rec.p = r.point_at_parameter(t);
            return true;
        }


    public:
        material* mp;
        float x;
        float z;
        float rad;
        float rad2;
        float k;
};

#endif