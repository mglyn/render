#pragma once
// 多种 Primitive 的 POD 描述（用于 GPU 上统一求交）
#include <glm/glm.hpp>

enum ShapeType {
	SHAPE_SPHERE = 0,
	SHAPE_TRIANGLE = 1,
	SHAPE_PLANE = 2
};

struct MaterialPOD {
	glm::vec3 albedo;    // 漫反射/基础颜色
	float metallic;      // 金属度 [0,1]
	glm::vec3 emission;  // 自发光 (可为0)
};

struct SpherePOD {
	glm::vec3 center;
	float radius;
	MaterialPOD mat;
};

struct TrianglePOD {
	glm::vec3 v0;
	glm::vec3 v1;
	glm::vec3 v2;
	MaterialPOD mat;
};

struct PlanePOD {
	glm::vec3 point;  // 平面上一点
	glm::vec3 normal; // 已归一化
	MaterialPOD mat;
};

struct Shape {
	ShapeType type;
	union Data {
		SpherePOD sph;
		TrianglePOD tri;
		PlanePOD pln;
	} data;
};

