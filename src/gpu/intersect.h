#pragma once
#include <optional>

#include "gpu/ray.h"
#include "scene/shape.h"

// 遍历 Shape 数组求交，返回最近命中
__device__ bool intersect(
    const Shape* shapes, int shapeCount, const Ray& r, float tMin, float tMax, HitRecord& rec);

// BVH加速求交
__device__ bool intersectBVH(
    const ModelGPU* models, int modelCount, const Ray& r, float tMin, float tMax, HitRecord& rec);

// 单个三角形求交辅助函数
__device__ bool intersectTriangle(
    const TrianglePOD& tri, const Ray& r, float tMin, float tMax, HitRecord& rec);

// AABB求交测试
__device__ bool intersectAABB(const glm::vec3& minBounds, const glm::vec3& maxBounds, const Ray& r);