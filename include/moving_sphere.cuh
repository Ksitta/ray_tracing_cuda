#ifndef MOVING_SPHERE_H
#define MOVING_SPHERE_H

#include "hitable.cuh"

class moving_sphere : public hitable {
    public:
        __device__ moving_sphere() {}
        __device__ moving_sphere(
            vec3 cen0, vec3 cen1, float _time0, float _time1, float r, material* m)
            : center0(cen0), center1(cen1), time0(_time0), time1(_time1), radius(r), mat_ptr(m)
        {};

        __device__ virtual bool hit(
            const ray& r, float t_min, float t_max, hit_record& rec) const override;

        __device__ vec3 center(float time) const;

    public:
        vec3 center0, center1;
        float time0, time1;
        float radius;
        material *mat_ptr;
};

__device__ vec3 moving_sphere::center(float time) const {
    return center0 + ((time - time0) / (time1 - time0))*(center1 - center0);
}

__device__ bool moving_sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center(r.time());
    float a = r.direction().squared_length();
    float half_b = dot(oc, r.direction());
    float c = oc.squared_length() - radius*radius;

    float discriminant = half_b*half_b - a*c;
    if (discriminant < 0) return false;
    float sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.point_at_parameter(rec.t);
    vec3 outward_normal = (rec.p - center(r.time())) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

#endif