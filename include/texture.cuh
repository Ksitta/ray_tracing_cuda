#ifndef TEXTURE_H
#define TEXTURE_H

#include "common.cuh"

class texture {
    public:
        __device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class solid_color : public texture {
    public:
        __device__ solid_color() {}
        __device__ solid_color(vec3 c) : color_value(c) {}

        __device__ solid_color(float red, float green, float blue)
          : solid_color(vec3(red,green,blue)) {}

        __device__ virtual vec3 value(float u, float v, const vec3& p) const override {
            return color_value;
        }

    private:
        vec3 color_value;
};

class checker_texture : public texture {
    public:
        __device__ checker_texture() {}

        __device__ checker_texture(texture* _even, texture* _odd)
            : even(_even), odd(_odd) {}

        __device__ checker_texture(vec3 c1, vec3 c2)
            : even(new solid_color(c1)) , odd(new solid_color(c2)) {}

        __device__ virtual vec3 value(float u, float v, const vec3& p) const override {
            float sines = sin(10*p.x())*sin(10*p.y())*sin(10*p.z());
            if (sines < 0)
                return odd->value(u, v, p);
            else
                return even->value(u, v, p);
        }

    public:
        texture* odd;
        texture* even;
};

#endif