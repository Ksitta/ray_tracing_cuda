#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>
#include "ray.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ Vec3 random_in_unit_disk(curandState *local_rand_state) {
    Vec3 p;
    do {
        p = 2.0f*Vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),0) - Vec3(1,1,0);
    } while (dot(p,p) >= 1.0f);
    return p;
}

class Camera {
public:
    __device__ Camera(Vec3 lookfrom, 
    Vec3 lookat, 
    Vec3 vup, 
    float vfov, 
    float aspect, 
    float aperture, 
    float focus_dist,
    float _time0 = 0,
    float _time1 = 0) { // vfov is top to bottom in degrees
        lens_radius = aperture / 2.0f;
        float theta = vfov*((float)M_PI)/180.0f;
        float half_height = tan(theta/2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin  - half_width*focus_dist*u -half_height*focus_dist*v - focus_dist*w;
        horizontal = 2.0f*half_width*focus_dist*u;
        vertical = 2.0f*half_height*focus_dist*v;            
        time0 = _time0;
        time1 = _time1;
    }
    __device__ ray get_ray(float s, float t, curandState *local_rand_state) {
        Vec3 rd = lens_radius*random_in_unit_disk(local_rand_state);
        Vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset, 
        lower_left_corner + s*horizontal + t*vertical - origin - offset,
        curand_uniform(local_rand_state)
        );
    }

    Vec3 origin;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 u, v, w;
    float lens_radius;
    float time0, time1;
};

#endif
