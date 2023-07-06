#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "hitable.cuh"
#include "common.cuh"

class Triangle: public hitable {
public:
	material *mat_ptr;
	vec3 normal;
	vec3 vertices[3];
    vec3 e1, e2;
    // a b c are three vertex positions of the triangle
	__host__ __device__ Triangle( const vec3 & a, const vec3 & b, const vec3 & c, material* m) : mat_ptr(m) {
		this->vertices[0] = a;
		this->vertices[1] = b;
		this->vertices[2] = c;
        this->e1 = b - a;
        this->e2 = c - a;
		this->normal = cross(e1, e2);
		this->normal.make_unit_vector();
	}

	__device__ Triangle() {};

	__device__ void set(const Triangle & a, material *mat) {
		this->vertices[0] = a.vertices[0];
		this->vertices[1] = a.vertices[1];
		this->vertices[2] = a.vertices[2];
        this->e1 = a.e1;
        this->e2 = a.e2;
		this->normal = a.normal;
		this->mat_ptr = mat;
	}

	__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
		vec3 s = r.origin() - this->vertices[0];
	    vec3 s1 = cross(r.dir, e2);
    	vec3 s2 = cross(s, e1);

		float s1e1 = dot(s1, e1);
		float t = dot(s2, e2) / s1e1;
		float b1 = dot(s1, s) / s1e1;
		float b2 = dot(s2, r.dir) / s1e1;

		if (b1 >= 0.f && b2 >= 0.f && (b1 + b2) <= 1.f && t >= t_min && t <= t_max) {
			rec.t = t;
			rec.mat_ptr = this->mat_ptr;
			rec.set_face_normal(r, normal);
			rec.p = r.point_at_parameter(t);
			return true;
		}

		return false;
    }

	// virtual bool bounding_box(AABB& output_box){
	// 	AABB k;
	// 	for(int i = 0; i < 3; i++){
	// 		k.fit(vertices[i]);
	// 	}
	// 	output_box = k;
	// 	return true;
	// }
protected:

};

#endif //TRIANGLE_H
