#pragma once

#include <glm/glm.hpp>

enum ShapeTypeGPU {
	SHAPE_SPHERE_GPU = 0,
	SHAPE_TRIANGLE_GPU = 1,
	SHAPE_PLANE_GPU = 2
};

struct TriangleGPU {
	glm::vec3 v0;
	glm::vec3 v1;
	glm::vec3 v2;
	glm::vec3 n0;
	glm::vec3 n1;
	glm::vec3 n2;
	glm::vec2 t0;
	glm::vec2 t1;
	glm::vec2 t2;
	int materialIndex;
};

