#ifndef SPHEREH
#define SPHEREH

#include "hitable.cuh"

class Sphere: public Hitable  {
    public:
        __device__ Sphere() {}
        __device__ Sphere(Vec3 cen, float r, Material *m) : center(cen), radius(r), mat_ptr(m)  {};
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, HitRecord& rec) const;
        __device__ static void get_sphere_uv(const Vec3& p, float& u, float& v) {
            // p: a given point on the sphere of radius one, centered at the origin.
            // u: returned value [0,1] of angle around the Y axis from X=-1.
            // v: returned value [0,1] of angle from Y=-1 to Y=+1.
            //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
            //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
            //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

            float theta = acos(-p.y());
            float phi = atan2(-p.z(), p.x()) + PI;

            u = phi / (2*PI);
            v = theta / PI;
        }
        Vec3 center;
        float radius;
        Material *mat_ptr;
};

__device__ bool Sphere::hit(const ray& r, float t_min, float t_max, HitRecord& rec) const {
    Vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.set_face_normal(r, (rec.p - center) / radius);
            rec.mat_ptr = mat_ptr;
            get_sphere_uv(rec.normal, rec.u, rec.v);
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.set_face_normal(r, (rec.p - center) / radius);
            rec.mat_ptr = mat_ptr;
            get_sphere_uv(rec.normal, rec.u, rec.v);
            return true;
        }
    }
    return false;
}


#endif
