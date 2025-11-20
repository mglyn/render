#ifndef MATERIAL_GPU_H
#define MATERIAL_GPU_H

#include <glm/glm.hpp>

struct MaterialGpu {
    glm::vec3 albedo;
    glm::vec3 emission;
    float metallic;
    float roughness;
};

#endif // MATERIAL_GPU_H
