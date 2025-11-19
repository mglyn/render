#pragma once
#include <glm/glm.hpp>
// 兼容非 CUDA 编译单元：在普通 C++ 编译时 __host__ / __device__ 为空宏
#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

class Ray {
public:
    __host__ __device__ Ray() = default;
    __host__ __device__ Ray(const glm::vec3& origin, const glm::vec3& direction) : orig(origin), dir(direction) {}
    __host__ __device__ glm::vec3 origin() const { return orig; }
    __host__ __device__ glm::vec3 direction() const { return dir; }
    __host__ __device__ glm::vec3 at(float t) const { return orig + t * dir; }
private:
    glm::vec3 orig;
    glm::vec3 dir;
};

// 命中信息结构体
struct HitRecord {
    float t;
    glm::vec3 point;
    glm::vec3 normal;
    glm::vec3 albedo;
    float metallic;
    glm::vec3 emission; // 新增：物体自发光（可为0）
};
