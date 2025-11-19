#pragma once

#include <glm/glm.hpp>

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


