#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "hitable.cuh"
#include "common.cuh"
#include "mat3.cuh"

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
        this->e1 = a - b;
        this->e2 = a - c;
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
		float t, b_pos, r_pos;
		vec3 dir = r.dir;
		dir.make_unit_vector();
		vec3 s = this->vertices[0] - r.origin();
		mat3 det0(dir, this->e1, this->e2);
		mat3 det1(s, this->e1, this->e2);
		mat3 det2(dir, s, this->e2);
		mat3 det3(dir, this->e1, s);
		float tem = det0.determinant();
		if(tem == 0){
			return false;
		}
		t = det1.determinant() / tem;
		b_pos = det2.determinant() / tem;
		r_pos = det3.determinant() / tem;
		if(t < t_min || b_pos < 0 || b_pos > 1 || r_pos < 0 || r_pos > 1 || b_pos + r_pos > 1 || t > t_max){
			return false;
		}
		rec.t = t;
		rec.mat_ptr = this->mat_ptr;
		rec.set_face_normal(r, normal);
		rec.p = r.point_at_parameter(t);
		return true;
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
