#ifndef REVSURFACE_HPP
#define REVSURFACE_HPP

#include "hitable.cuh"
#include "curve.cuh"
#include "cylinder.cuh"
#include "common.cuh"

__device__ inline void getUVOfRevsurface(const vec3& p, float& u, float& v, const vec3& pole, float height){
    float dx = p.x() - pole.x();
    float dy = p.y() - pole.y();
    float dz = p.z() - pole.z();
    float phi = atan2(dx, dz) + PI;
    u = phi / (PI * 2);
    v = dy / height;
}

class RevSurface : public hitable {
    Curve *curve;
    float height;
    float maxR;
    vec3 pos;
    Cylinder *outer;
    material *mat;
public:
    __device__ RevSurface(const vec3& position, Curve *curve, material* matl) 
    : curve(curve), pos(position) {
        height = curve->controls[curve->controls_num - 1].y() - curve->controls[0].y();
        maxR = curve->max_r;
        outer = new Cylinder(maxR, height, pos, matl);
        this->mat = matl;
    }

    __device__ virtual bool intersect(const ray& r, hit_record& h, float t, float u, float v, float t_min, float t_max) const {
        float eps = 1e-4f;
        for (int i = 0; i < 10; i++) {
            t = clamp(t, eps, inf);
            u = clamp(u, 0, 1);
            CurvePoint eval = curve->evaluate(u);
            vec3 point = eval.v;
            vec3 tangent = eval.t;
            vec3 rot_point = rotateY(v, point) + pos;
            vec3 new_point = r.point_at_parameter(t);
            vec3 diff = new_point - rot_point;
            float d = sqrt(point.z() * point.z() + point.x() * point.x());
            if (diff.squared_length() < eps){
                if (t < t_max && t > t_min && t > 0.1f){ 
                    vec3 du = rotateY(v, tangent);
                    vec3 dv = tangent_normal_disk(v, d);
                    vec3 n = cross(dv, du);
                    n.make_unit_vector();
                    // getUVOfRevsurface(rot_point, h.u, h.v, pos, height);
                    h.t = t;
                    h.set_face_normal(r, n);
                    h.p = rot_point;
                    h.mat_ptr = mat;
                    // h.set(t, n, rot_point, mat);
                    return true;
                }else{
                    return false;
                }
            }
            vec3 dir = r.direction();
            vec3 du = rotateY(v, tangent);
            vec3 dv = tangent_normal_disk(v, d);
            vec3 dF = -diff;
            float D = dot(dir, cross(du, dv));
            u += dot(dir, cross(dv, dF)) / D;
            v -= dot(dir, cross(du, dF)) / D;
            if(v < -PI) v += 2 * PI;
            if(v > PI) v -= 2 * PI;
            t += dot(du, cross(dv, dF)) / D;
        }
        return false;
    }

    __device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &h) const override {
        hit_record hc;
        if (outer->hit(r, t_min, t_max, hc)){
            float t = hc.t;
            float u = (hc.p.y() - pos.y()) / height;
            if(hc.p.y() == pos.y() || hc.p.y() == pos.y() + height){
                u = curve->solve(hc.p.x() - pos.x());
            }
            float v = atan2(hc.p.z() - pos.z(), hc.p.x() - pos.x());
            return intersect(r, h, t, u, v, t_min, t_max);
        }
        return false;
    }


};

#endif
