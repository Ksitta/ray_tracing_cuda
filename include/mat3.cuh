#ifndef MATRIX3F_H
#define MATRIX3F_H

#include "vec3.cuh"

// 3x3 Matrix, stored in column major order (OpenGL style)
class mat3
{
public:

	__device__ mat3( const vec3& v0, const vec3& v1, const vec3& v2){
        m_elements[0] = v0.x(); m_elements[3] = v1.x(); m_elements[6] = v2.x();
        m_elements[1] = v0.y(); m_elements[4] = v1.y(); m_elements[7] = v2.y();
        m_elements[2] = v0.z(); m_elements[5] = v1.z(); m_elements[8] = v2.z();
    }

	__device__ float determinant() const {
        return m_elements[0] * (m_elements[4] * m_elements[8] - m_elements[5] * m_elements[7]) -
               m_elements[1] * (m_elements[3] * m_elements[8] - m_elements[5] * m_elements[6]) +
               m_elements[2] * (m_elements[3] * m_elements[7] - m_elements[4] * m_elements[6]);
    }
	
private:

	float m_elements[ 9 ];

};

#endif // MATRIX3F_H
