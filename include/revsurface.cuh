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
        maxR = getMaxRadius();
        for (int i = 0; i < curve->controls_num; i++){
            curve->controls[i] += pos;
        }
        outer = new Cylinder(maxR, height, pos, matl);
        this->mat = matl;
    }

    __device__ float getMaxRadius() {
        float max = 0;
        for(int i = 0; i < curve->controls_num; i++){
            if (curve->controls[i].x() > max){
                max = curve->controls[i].x();
            }
        }
        return max;
    }

    __device__ virtual bool intersect(const ray& r, hit_record& h, float t, float u, float v, float t_min, float t_max) const {
        float eps = 1e-4f;
        for (int i = 0; i < 10; i++) {
            t = clamp(t, eps, inf);
            u = clamp(u, 0, 1);
            v = clamp(v, -PI, PI);
            CurvePoint eval = curve->evaluate(u);
            vec3 point = eval.v + pos;
            vec3 tangent = eval.t;
            vec3 rot_point = rotate(pos, v, point);
            vec3 new_point = r.point_at_parameter(t);
            vec3 diff = new_point - rot_point;
            if (diff.squared_length() < eps){
                if (t < t_max && t > t_min && 0 < u && u < 1){ 
                    float d = sqrt(pow(point.z() - pos.z(), 2) + pow(point.x() - pos.x(), 2));
                    vec3 du = rotate(vec3(0, 0, 0), v, tangent);
                    vec3 dv = tangent_at_disk(v, d);
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
            float d = vec3(point.x() - pos.x(), 0, point.z() - pos.z()).length();
            vec3 dir = r.direction();
            vec3 du = rotate(vec3(0,0,0), v, tangent);
            vec3 dv = tangent_at_disk(v, d);
            vec3 dF = -diff;
            float D = dot(dir, cross(du, dv));
            u += dot(dir, cross(dv, dF)) / D;
            v -= dot(dir, cross(du, dF)) / D;
            t += dot(du, cross(dv, dF)) / D;
        }
        return false;
    }

    __device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &h) const override {
        hit_record hc;
        float d_pole = sqrt(pow(r.origin().x() - pos.x(), 2) + pow(r.origin().z() - pos.z(), 2));
        if (outer->hit(r, t_min, t_max, hc)){
            float t = hc.t;
            float u = clamp(curve->solve(hc.p.y() - pos.y()));
            float v = atan2(hc.p.z(), hc.p.x());
            // printf("hit outer\n");
            return intersect(r, h, t, u, v, t_min, t_max);
        }
        return false;
    }


};

#endif
