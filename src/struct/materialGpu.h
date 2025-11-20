#pragma once

#include <glm/glm.hpp>

struct MaterialGpu {
    glm::vec3 albedo;
    float metallic;
    glm::vec3 emission;
    float roughness;
};
