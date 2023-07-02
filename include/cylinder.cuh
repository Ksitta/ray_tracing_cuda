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
        // 求解 (z-pos.z))^2 + (x-pos.x))^2 = R^2
        // 其中 z = r.origin().z + r.direction().z * t, x = r.origin().x + r.direction().x * t
        // 代入展开得，(r.direction().z^2 + r.direction().x^2) * t^2 +
        // 2*(r.direction().z*(r.origin().z-pos.z) + r.direction().x*(r.origin().x-pos.x)) * t +
        // ((r.origin().z-pos.z)^2 + (r.origin().x-pos.x)^2 - R^2) = 0
        float dz = r.origin().z() - pos.z();
        float dx = r.origin().x() - pos.x();
        // if(dz * dz + dx * dx < radius * radius){
        //     h.t = (pos + vec3(0, height / 3, 0) - r.o).length();
        //     h.x = r.at(h.t);
        //     return true;
        // }
        float a = r.direction().z() * r.direction().z() + r.direction().x() * r.direction().x();
        float b = 2 * (r.direction().z() * dz + r.direction().x() * dx);
        float c = dz * dz + dx * dx - radius * radius;
        float delta = b * b - 4 * a * c;
        if (delta > eps) {
            float t1 = (-b - sqrt(delta)) / (2 * a);           
            if (t1 > eps){
                vec3 p = r.point_at_parameter(t1);
                if (p.y() > pos.y() && p.y() < pos.y() + height){
                    vec3 n = vec3(p.x() - pos.x(), 0, p.z() - pos.z());
                    n.make_unit_vector();
                    h.normal = n;
                    h.t = t1;
                    h.p = p;
                    // h.set(t1, n, p, mat);
                    return true;
                }
            }
            float t2 = (-b + sqrt(delta)) / (2 * a);
            if (t2 >  eps) {
                vec3 p = r.point_at_parameter(t2);
                if (p.y() > pos.y() && p.y() < pos.y() + height) {
                    vec3 n = vec3(p.x() - pos.x(), 0, p.z() - pos.z());
                    n.make_unit_vector();
                    h.normal = n;
                    h.t = t2;
                    h.p = p;
                    // h.set(t2, n, p, mat);
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
