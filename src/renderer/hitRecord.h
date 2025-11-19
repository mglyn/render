#pragma once

#include <glm/glm.hpp>

// 命中信息结构体
struct HitRecord {
    float t = 0.0f;
    glm::vec3 point = glm::vec3(0.0f);
    glm::vec3 normal = glm::vec3(0.0f);
    glm::vec3 albedo = glm::vec3(0.0f);
    float metallic = 0.0f;
    glm::vec3 emission = glm::vec3(0.0f);
    int materialIndex = -1;
    int primitiveIndex = -1;
    int objectIndex = -1;
    int fromIlluminant = 0;
};