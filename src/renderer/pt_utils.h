#pragma once
#include <glm/glm.hpp>
#include <math.h>

__device__ inline uint32_t pcg_hash(uint32_t& seed) {
    seed = seed * 747796405u + 2891336453u;
    uint32_t word = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ inline float rand01(uint32_t& seed) { 
    return static_cast<float>(pcg_hash(seed)) / 4294967295.0f; 
}

// 重要性采样
__device__ inline glm::vec3 cosineSampleHemisphere(const glm::vec3& normal, uint32_t& seed) {
    float u1 = rand01(seed);
    float u2 = rand01(seed);
    float r = sqrtf(u1);
    float theta = 2.0f * 3.1415926535f * u2;
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u1));
    glm::vec3 tangent = glm::normalize(glm::abs(normal.x) > 0.1f ? glm::cross(normal, glm::vec3(0.0f, 1.0f, 0.0f)) : glm::cross(normal, glm::vec3(1.0f, 0.0f, 0.0f)));
    glm::vec3 bitangent = glm::cross(normal, tangent);
    glm::vec3 sample = x * tangent + y * bitangent + z * normal;
    return glm::normalize(sample);
}

// 均匀半球采样
__device__ inline glm::vec3 uniformSampleHemisphere(const glm::vec3& normal, uint32_t& seed) {
    float u1 = rand01(seed);
    float u2 = rand01(seed);
    float z = u1; // [0,1]
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = 2.0f * 3.1415926535f * u2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);

    glm::vec3 tangent = glm::normalize(glm::abs(normal.x) > 0.1f ? glm::cross(normal, glm::vec3(0.0f, 1.0f, 0.0f)) : glm::cross(normal, glm::vec3(1.0f, 0.0f, 0.0f)));
    glm::vec3 bitangent = glm::cross(normal, tangent);
    glm::vec3 sample = x * tangent + y * bitangent + z * normal;
    return glm::normalize(sample);
}
