#pragma once

#include <glm/glm.hpp>

struct Material {
	glm::vec3 albedo;    // 漫反射/基础颜色
	float metallic;      // 金属度 [0,1]
	glm::vec3 emission;  // 自发光 (可为0)
};