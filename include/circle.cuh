#ifndef __CIRCLE_CUH__
#define __CIRCLE_CUH__

#include "hitable.cuh"

class XZCircle : public Hitable {
    public:
        __device__ XZCircle() {}

        __device__ XZCircle(float x, float y, float rad, float _k,
            Material* mat)
            : x(x), z(z), k(_k), mp(mat), rad(rad) { rad2 = rad * rad; };

        __device__ virtual bool hit(const ray& r, float t_min, float t_max, HitRecord& rec) const override{
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
            Vec3 outward_normal = Vec3(0, 1, 0);
            rec.set_face_normal(r, outward_normal);
            rec.mat_ptr = mp;
            rec.p = r.point_at_parameter(t);
            return true;
        }


    public:
        Material* mp;
        float x;
        float z;
        float rad;
        float rad2;
        float k;
};

#endif