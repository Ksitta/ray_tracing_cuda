#include <iostream>
#include <time.h>
#include <float.h>
#include <fstream>
#include <sstream>
#include <string>
#include <curand_kernel.h>
#include "vec3.cuh"
#include "ray.cuh"
#include "sphere.cuh"
#include "hitable_list.cuh"
#include "camera.cuh"
#include "material.cuh"
#include "common.cuh"
#include "moving_sphere.cuh"
#include "common.cuh"
#include "rect.cuh"
#include "revsurface.cuh"
#include "mesh.cuh"

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ Vec3 color(const ray& r, Hitable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    Vec3 cur_attenuation = Vec3(1.0,1.0,1.0);
    Vec3 cur_color = Vec3(0, 0, 0);
    for(int i = 0; i < 50; i++) {
        HitRecord rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            Vec3 attenuation;
            Vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
            cur_color += cur_attenuation * emitted;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return cur_color;
            }
        }
        else {
            cur_color += cur_attenuation * Vec3(0,0,0);
            return cur_color;
        }
    }
    return Vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(Vec3 *fb, int max_x, int max_y, int ns, Camera **cam, Hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    Vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))
__device__ int object_num = 0;
const int max_num_hitables = 1000;
Hitable **d_list;
__device__ int texture_num = 0;
const int max_num_texture = 1000;
texture **d_texturelist;

__device__ void add_object(Hitable **d_list, Hitable *object) {
    d_list[object_num++] = object;
}

// __global__ void create_world(hitable **d_list, hitable **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         curandState local_rand_state = *rand_state;
//         auto red   = new Lambertian(vec3(.65, .05, .05));
//         auto white = new Lambertian(vec3(.73, .73, .73));
//         auto green = new Lambertian(vec3(.12, .45, .15));
//         auto light = new DiffuseLight(vec3(15, 15, 15));

//         add_object(d_list, new YZRect(0, 555, 0, 555, 555, green));
//         add_object(d_list, new YZRect(0, 555, 0, 555, 0, red));
//         add_object(d_list, new XZRect(213, 343, 227, 332, 554, light));
//         add_object(d_list, new XZRect(0, 555, 0, 555, 0, white));
//         add_object(d_list, new XZRect(0, 555, 0, 555, 555, white));
//         add_object(d_list, new XYRect(0, 555, 0, 555, 555, white));

//         float sf = 300;
//         vec3 vase[] = {
//             vec3(0.27, 0, 0) * sf,
//             vec3(0.29, 0.1, 0) * sf,
//             vec3(0.33, 0.2, 0) * sf,
//             vec3(0.40, 0.4, 0) * sf,
//             vec3(0.36, 0.6, 0) * sf,
//             vec3(0.21, 0.72, 0) * sf,
//             vec3(0.3, 1, 0) * sf,
//         };
//         add_object(d_list,
//                     // new Sphere(vec3(70, 0, 55), 19.8, new Lambertian(vec3(117 / 255.f,190 / 255.f, 204 / 255.f)))
//                     new RevSurface(vec3(278, 0, 400), new Curve(vase, 7), new Lambertian(vec3(117 / 255.f,190 / 255.f, 204 / 255.f)))
//                     // new Cylinder(24, 60, vec3(70, -5, 55), new Lambertian(vec3(117 / 255.f,190 / 255.f, 204 / 255.f)))
//         );


//         *rand_state = local_rand_state;
//         *d_world  = new HittableList(d_list, object_num);

//         // float aspect_ratio = 1.0;
//         // int image_width = 600;
//         // int samples_per_pixel = 200;
//         vec3 background = vec3(0,0,0);
//         vec3 lookfrom = vec3(278, 278, -800);
//         vec3 lookat = vec3(278, 278, 0);
//         float vfov = 40.0;
//         float dist_to_focus = 1;
//         float aperture = 0;

//         *d_camera   = new Camera(lookfrom,
//                                  lookat,
//                                  vec3(0,1,0),
//                                  vfov,
//                                  float(nx)/float(ny),
//                                  aperture,
//                                  dist_to_focus);
//     }
// }

__global__ void add_mesh(Hitable **d_list, Triangle *triangles, int triangles_cnt){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // add_object(d_list, new Mesh(triangles, triangles_cnt, new Metal(vec3(0.999, 0.999, 0.999), 0)));
        add_object(d_list, new Mesh(triangles, triangles_cnt, new Lambertian(Vec3(220 / 255.f, 174 / 255.f, 185 / 255.f))));
    }
}

__global__ void create_world(Hitable **d_list, Hitable **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state, texture **d_tex) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;

        add_object(d_list, new YZRect(0, 1000, 0, 1000, 1, new Lambertian(Vec3(117 / 255.f,190 / 255.f, 204 / 255.f))));
        add_object(d_list, new XYRect(0, 1000, 0, 1000, 0, new Lambertian(Vec3(220 / 255.f, 174 / 255.f, 185 / 255.f))));
        add_object(d_list, new XZRect(0, 1000, 0, 1000, 0, new Lambertian(Vec3(200 / 255.f, 150 / 255.f, 123 / 255.f))));
        add_object(d_list, new XZRect(0, 1000, 0, 1000, 81.6, new Lambertian(Vec3(200 / 255.f, 150 / 255.f, 123 / 255.f))));
        add_object(d_list, new Sphere(Vec3(80, 681.6 - 0.285, 115), 600, new DiffuseLight(Vec3(10, 10, 10))));

        add_object(d_list, new Sphere(Vec3(60, 13, 100.6), 13, new Metal(Vec3(0.999, 0.999, 0.999), 0)));
        add_object(d_list, new Sphere(Vec3(128, 15, 110), 15, new Dielectric(1.5)));

        float sf = 60;

        Vec3 vase[] = {
            Vec3(0.27, 0, 0) * sf,
            Vec3(0.29, 0.1, 0) * sf,
            Vec3(0.33, 0.2, 0) * sf,
            Vec3(0.40, 0.4, 0) * sf,
            Vec3(0.36, 0.6, 0) * sf,
            Vec3(0.21, 0.72, 0) * sf,
            Vec3(0.3, 1, 0) * sf,
        };
        add_object(d_list,
                    // new Sphere(vec3(70, 0, 55), 19.8, new Lambertian(vec3(117 / 255.f,190 / 255.f, 204 / 255.f)))
                    new RevSurface(Vec3(70, -5, 55), new BezierCurve(vase, 7), new Lambertian(d_tex[0]))
                    // new Cylinder(24, 60, vec3(70, -5, 55), new Lambertian(vec3(117 / 255.f,190 / 255.f, 204 / 255.f)))
        );

        // add_object(d_list, new Triangle(vec3(0, 21.213, 121.213), vec3(30, 21.213, 121.213), vec3(30, 42.426, 100), new Lambertian(vec3(220 / 255.f, 174 / 255.f, 185 / 255.f))));
        *rand_state = local_rand_state;
        *d_world  = new HittableList(d_list, object_num);

        Vec3 lookfrom(140, 52, 180.6f);
        Vec3 lookat(50, 30, 50);
        float dist_to_focus = 1;
        float aperture = 0;
        *d_camera   = new Camera(lookfrom,
                                 lookat,
                                 Vec3(0,1,0),
                                 45,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void free_world(Hitable **d_list, Hitable **d_world, Camera **d_camera) {
    for(int i=0; i < object_num; i++) {
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

__global__ void add_texture_to_list(texture **d_list, unsigned char *data, int width, int height) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[texture_num++] = new ImageTexture(data, width, height);
    }
}

void add_texture(const char* filename, texture **d_list){
    int components_per_pixel = 3;
    int height, width;
    unsigned char *data = stbi_load(
                filename, &width, &height, &components_per_pixel, components_per_pixel);
    unsigned char *d_data;
    if (!data) {
        std::cerr << "ERROR: Could not load Texture image file '" << filename << "'.\n";
        width = height = 0;
        exit(1);
    }

    checkCudaErrors(cudaMalloc((void**)&d_data, width * height * components_per_pixel * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(d_data, data, width * height * components_per_pixel * sizeof(unsigned char), cudaMemcpyHostToDevice));
    free(data);
    add_texture_to_list<<<1, 1>>>(d_list, d_data, width, height);
}

int main() {
    int nx = 1920;
    int ny = 1080;
    int ns = 10000;
    int tx = 16;
    int ty = 16;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(Vec3);

    // allocate FB
    Vec3 *fb;

    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the Camera

    checkCudaErrors(cudaMalloc((void **)&d_list, max_num_hitables * sizeof(Hitable *)));
    Hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Hitable *)));
    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));

    checkCudaErrors(cudaMalloc((void **)&d_texturelist, max_num_texture * sizeof(texture *)));

    // Add object here
    Triangle *triangles;
    int triangle_num;
    read_mesh("../mesh/cube.obj", &triangles, &triangle_num);
    add_mesh<<<1, 1>>>(d_list, triangles, triangle_num);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    add_texture("../imgs/2.jpg", d_texturelist);
    create_world<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2, d_texturelist);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state);
    // checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    float timer_seconds = ((float)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    FILE *f = fopen("image.ppm", "w");

    fprintf(f, "P3\n%d %d\n%d\n", nx, ny, 255);

    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99f*fb[pixel_index].r());
            int ig = int(255.99f*fb[pixel_index].g());
            int ib = int(255.99f*fb[pixel_index].b());
            if(ir > 255){
                ir = 255;
            }
            if(ig > 255){
                ig = 255;
            }
            if(ib > 255){
                ib = 255;
            }
            fprintf(f, "%d %d %d ", ir, ig, ib);
        }
    }

    fclose(f);

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}
