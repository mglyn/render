#pragma once

#include <glm/glm.hpp>

// 命中信息结构体
struct HitRecord {
    float t;
    glm::vec3 point;
    glm::vec3 normal;
    glm::vec3 albedo;
    float metallic;
    glm::vec3 emission;
};