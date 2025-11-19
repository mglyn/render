#pragma once

#include <glm/glm.hpp>

enum ShapeType {
	SHAPE_SPHERE = 0,
	SHAPE_TRIANGLE = 1,
	SHAPE_PLANE = 2
};

struct Material {
	glm::vec3 albedo;    // 漫反射/基础颜色
	float metallic;      // 金属度 [0,1]
};

struct Triangle {
	glm::vec3 v0;
	glm::vec3 v1;
	glm::vec3 v2;
	Material mat;
};

