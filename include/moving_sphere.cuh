#ifndef MOVING_SPHERE_H
#define MOVING_SPHERE_H

#include "hitable.cuh"

class MovingSphere : public Hitable {
    public:
        __device__ MovingSphere() {}
        __device__ MovingSphere(
            Vec3 cen0, Vec3 cen1, float _time0, float _time1, float r, Material* m)
            : center0(cen0), center1(cen1), time0(_time0), time1(_time1), radius(r), mat_ptr(m)
        {};

        __device__ virtual bool hit(
            const ray& r, float t_min, float t_max, HitRecord& rec) const override;

        __device__ Vec3 center(float time) const;

    public:
        Vec3 center0, center1;
        float time0, time1;
        float radius;
        Material *mat_ptr;
};

__device__ Vec3 MovingSphere::center(float time) const {
    return center0 + ((time - time0) / (time1 - time0))*(center1 - center0);
}

__device__ bool MovingSphere::hit(const ray& r, float t_min, float t_max, HitRecord& rec) const {
    Vec3 oc = r.origin() - center(r.time());
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
    Vec3 outward_normal = (rec.p - center(r.time())) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

#endif