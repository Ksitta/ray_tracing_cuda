#ifndef TEXTURE_H
#define TEXTURE_H

#include "common.cuh"
#include "stb_image.h"

class texture {
    public:
        __device__ virtual Vec3 value(float u, float v, const Vec3& p) const = 0;
};

class solid_color : public texture {
    public:
        __device__ solid_color() {}
        __device__ solid_color(Vec3 c) : color_value(c) {}

        __device__ solid_color(float red, float green, float blue)
          : solid_color(Vec3(red,green,blue)) {}

        __device__ virtual Vec3 value(float u, float v, const Vec3& p) const override {
            return color_value;
        }

    private:
        Vec3 color_value;
};

class checker_texture : public texture {
    public:
        __device__ checker_texture() {}

        __device__ checker_texture(texture* _even, texture* _odd)
            : even(_even), odd(_odd) {}

        __device__ checker_texture(Vec3 c1, Vec3 c2)
            : even(new solid_color(c1)) , odd(new solid_color(c2)) {}

        __device__ virtual Vec3 value(float u, float v, const Vec3& p) const override {
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


class ImageTexture : public texture {
    public:
        const static int bytes_per_pixel = 3;

        ImageTexture()
          : data(nullptr), width(0), height(0), bytes_per_scanline(0) {}

        __device__ ImageTexture(unsigned char* pixels, int width, int height)
          : data(pixels), width(width), height(height), bytes_per_scanline(width * bytes_per_pixel) {}

        ImageTexture(const char* filename) {
            auto components_per_pixel = bytes_per_pixel;

            data = stbi_load(
                filename, &width, &height, &components_per_pixel, components_per_pixel);

            if (!data) {
                std::cerr << "ERROR: Could not load Texture image file '" << filename << "'.\n";
                width = height = 0;
            }

            bytes_per_scanline = bytes_per_pixel * width;
        }

        __device__ virtual Vec3 value(float u, float v, const Vec3& p) const override {

            if (data == nullptr){
                return Vec3(0,1,1);
            }

            u = clamp(u, 0.0, 1.0);
            v = 1.0 - clamp(v, 0.0, 1.0); 

            auto i = int(u * width);
            auto j = int(v * height);

            if (i >= width)  i = width - 1;
            if (j >= height) j = height - 1;

            double color_scale = 1.0 / 255.0;
            auto pixel = data + j * bytes_per_scanline + i * bytes_per_pixel;

            return Vec3(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
        }

    private:
        unsigned char *data;
        int width, height;
        int bytes_per_scanline;
};

#endif