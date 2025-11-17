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
	float ior = 1.0f;    // 折射率
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

	// 工厂函数，简化 Shape 创建
	static Shape make_sphere(const glm::vec3& center, float radius, const MaterialPOD& mat) {
		Shape s{}; // 使用聚合初始化，避免默认构造函数问题
		s.type = SHAPE_SPHERE;
		s.data.sph.center = center;
		s.data.sph.radius = radius;
		s.data.sph.mat = mat;
		return s;
	}

	static Shape make_triangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const MaterialPOD& mat) {
		Shape s{}; // 使用聚合初始化
		s.type = SHAPE_TRIANGLE;
		s.data.tri.v0 = v0;
		s.data.tri.v1 = v1;
		s.data.tri.v2 = v2;
		s.data.tri.mat = mat;
		return s;
	}

	static Shape make_plane(const glm::vec3& point, const glm::vec3& normal, const MaterialPOD& mat) {
		Shape s{}; // 使用聚合初始化
		s.type = SHAPE_PLANE;
		s.data.pln.point = point;
		s.data.pln.normal = normal;
		s.data.pln.mat = mat;
		return s;
	}
};

