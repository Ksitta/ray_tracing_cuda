#ifndef __CYLINDER_H__
#define __CYLINDER_H__

#include "common.cuh"
#include "hitable.cuh"
#include "vec3.cuh"
#include "material.cuh"
#include "circle.cuh"

class Cylinder : public Hitable {
public:
    __device__ Cylinder(float r, float h,const Vec3& _pos, Material* m) {
        this->pos = _pos;
        this->radius = r;
        this->height = h;
        this->mat = m;
        this->c1 = new XZCircle(_pos.x(), pos.z(), r, pos.y(), m);
        this->c2 = new XZCircle(_pos.x(), pos.z(), r, pos.y() + h, m);
        this->radius2 = r * r;
    }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, HitRecord& h) const override {
        float rx = r.direction().x();
        float rz = r.direction().z();
        float len = sqrt(rx * rx + rz * rz);
        rx /= len;
        rz /= len;

        float ox = pos.x() - r.origin().x();
        float oz = pos.z() - r.origin().z();

        float a = rx * ox + rz * oz;

        bool hit = false;
        // if(c1->hit(r, t_min, t_max, h)){
        //     t_max = h.t;
        //     hit = true;
        // }
        // if(c2->hit(r, t_min, t_max, h)){
        //     t_max = h.t;
        //     hit = true;
        // }

        if(a > eps){
            float rds = ox * ox + oz * oz - a * a;
            float rads = this->radius * this->radius;
            if(rds < rads){
                float inner = sqrt(rads - rds);
                float dis = a - inner;
                float t = dis / len;
                Vec3 p = r.point_at_parameter(t);
                if(t > t_min && t < t_max && p.y() > pos.y() && p.y() < pos.y() + height){
                    Vec3 n = Vec3(p.x() - pos.x(), 0, p.z() - pos.z());
                    n.make_unit_vector();
                    h.set_face_normal(r, n);
                    h.t = t;
                    h.p = p;
                    h.mat_ptr = mat;
                    return true;
                }
            }
        }
        float len2 = ox * ox + oz * oz;
        if(len2 < this->radius2){
            float h2 = len2 - a * a;
            float w2 = this->radius2 - h2;
            float w = sqrt(w2);
            float dis = w + a;
            float t = dis / len;
            Vec3 p = r.point_at_parameter(t);
            if(t > t_min && t < t_max && p.y() > pos.y() && p.y() < pos.y() + height){
                Vec3 n = Vec3(p.x() - pos.x(), 0, p.z() - pos.z());
                n.make_unit_vector();
                h.set_face_normal(r, n);
                h.t = t;
                h.p = p;
                h.mat_ptr = mat;
                return true;
            }
        }

        return hit;
    }

    float radius;
    float radius2;
    float height;
    Vec3 pos;
    Material *mat;
    XZCircle *c1, *c2;
};

#endif
