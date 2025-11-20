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

	bool operator<(const Material& other) const {
		if (albedo.r != other.albedo.r) return albedo.r < other.albedo.r;
		if (albedo.g != other.albedo.g) return albedo.g < other.albedo.g;
		if (albedo.b != other.albedo.b) return albedo.b < other.albedo.b;
		return metallic < other.metallic;
	}
};

struct Triangle {
	glm::vec3 v0;
	glm::vec3 v1;
	glm::vec3 v2;
	glm::vec3 n0;
	glm::vec3 n1;
	glm::vec3 n2;
	glm::vec2 t0;
	glm::vec2 t1;
	glm::vec2 t2;
	Material mat;
};

