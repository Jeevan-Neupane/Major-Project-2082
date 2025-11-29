#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Vector3 {
    float x, y, z;
    
    Vector3() : x(0), y(0), z(0) {}
    Vector3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    Vector3 operator+(const Vector3& v) const { return Vector3(x+v.x, y+v.y, z+v.z); }
    Vector3 operator-(const Vector3& v) const { return Vector3(x-v.x, y-v.y, z-v.z); }
    Vector3 operator*(float s) const { return Vector3(x*s, y*s, z*s); }
    Vector3 operator/(float s) const { return Vector3(x/s, y/s, z/s); }
    
    float dot(const Vector3& v) const { return x*v.x + y*v.y + z*v.z; }
    Vector3 cross(const Vector3& v) const {
        return Vector3(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
    }
    float length() const { return std::sqrt(x*x + y*y + z*z); }
    Vector3 normalize() const {
        float len = length();
        return len > 1e-8f ? (*this / len) : Vector3();
    }
};

struct Matrix3x3 {
    float m[9];
    
    Matrix3x3() {
        for (int i = 0; i < 9; i++) m[i] = 0.0f;
    }
    
    static Matrix3x3 identity() {
        Matrix3x3 mat;
        mat.m[0] = mat.m[4] = mat.m[8] = 1.0f;
        return mat;
    }
    
    Vector3 operator*(const Vector3& v) const {
        return Vector3(
            m[0]*v.x + m[1]*v.y + m[2]*v.z,
            m[3]*v.x + m[4]*v.y + m[5]*v.z,
            m[6]*v.x + m[7]*v.y + m[8]*v.z
        );
    }
    
    Matrix3x3 inverse() const {
        Matrix3x3 inv;
        float det = m[0]*(m[4]*m[8]-m[5]*m[7]) - m[1]*(m[3]*m[8]-m[5]*m[6]) + m[2]*(m[3]*m[7]-m[4]*m[6]);
        if (std::abs(det) < 1e-9f) return identity();
        
        float invDet = 1.0f / det;
        inv.m[0] = (m[4]*m[8] - m[5]*m[7]) * invDet;
        inv.m[1] = (m[2]*m[7] - m[1]*m[8]) * invDet;
        inv.m[2] = (m[1]*m[5] - m[2]*m[4]) * invDet;
        inv.m[3] = (m[5]*m[6] - m[3]*m[8]) * invDet;
        inv.m[4] = (m[0]*m[8] - m[2]*m[6]) * invDet;
        inv.m[5] = (m[2]*m[3] - m[0]*m[5]) * invDet;
        inv.m[6] = (m[3]*m[7] - m[4]*m[6]) * invDet;
        inv.m[7] = (m[1]*m[6] - m[0]*m[7]) * invDet;
        inv.m[8] = (m[0]*m[4] - m[1]*m[3]) * invDet;
        
        return inv;
    }
};

#endif // MATH_UTILS_H
