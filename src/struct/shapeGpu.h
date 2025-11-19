#pragma once

#include <glm/glm.hpp>

// GPU 侧材质（与 CPU 侧 Material 保持布局兼容）
struct MaterialGPU {
	glm::vec3 albedo;
	float metallic;
	glm::vec3 emission;
	float pad; // 对齐占位
};

enum ShapeTypeGPU {
	SHAPE_SPHERE_GPU = 0,
	SHAPE_TRIANGLE_GPU = 1,
	SHAPE_PLANE_GPU = 2
};

struct SphereGPU {
	glm::vec3 center;
	float radius;
	int materialIndex;
};

struct TriangleGPU {
	glm::vec3 v0;
	glm::vec3 v1;
	glm::vec3 v2;
	int materialIndex;
};

struct PlaneGPU {
	glm::vec3 point;
	glm::vec3 normal;
	int materialIndex;
};

struct ShapeGPU {
	ShapeTypeGPU type;
	union Data {
		SphereGPU sph;
		TriangleGPU tri;
		PlaneGPU pln;
	} data;
};

