#ifndef __CYLINDER_H__
#define __CYLINDER_H__

#include "common.cuh"
#include "hitable.cuh"
#include "vec3.cuh"
#include "material.cuh"

class Cylinder : public hitable {
public:
    __device__ Cylinder(float r, float h,const vec3& _pos, material* m) {
        this->pos = _pos;
        this->radius = r;
        this->height = h;
        this->mat = m;
    }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& h) const override {
        float rx = r.direction().x();
        float rz = r.direction().z();
        float len = sqrt(rx * rx + rz * rz);
        rx /= len;
        rz /= len;

        float ox = pos.x() - r.origin().x();
        float oz = pos.z() - r.origin().z();

        float a = rx * ox + rz * oz;

        if(a > eps){
            float rds = ox * ox + oz * oz - a * a;
            float rads = this->radius * this->radius;
            if(rds < rads){
                float inner = sqrt(rads - rds);
                float dis = a - inner;
                float t = dis / len;
                vec3 p = r.point_at_parameter(t);
                if(t > t_min && t < t_max && p.y() > pos.y() && p.y() < pos.y() + height){
                    vec3 n = vec3(p.x() - pos.x(), 0, p.z() - pos.z());
                    n.make_unit_vector();
                    h.set_face_normal(r, n);
                    h.t = t;
                    h.p = p;
                    h.mat_ptr = mat;
                    return true;
                }
            }
        }

        return false;
    }

    float radius;
    float height;
    vec3 pos;
    material *mat;
};

#endif
