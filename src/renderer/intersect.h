#pragma once
#include <optional>

#include "struct/ray.h"
#include "struct/shape.h"
#include "struct/shapeGpu.h"
#include "renderer/hitRecord.h"
#include "struct/modelGpu.h"

// 单个三角形求交辅助函数（GPU 版，使用材质索引）
__device__ bool intersectTriangle(
    const TriangleGPU& tri, const MaterialGpu* materials, const Ray& r, float tMin, float tMax, HitRecord& rec);

// AABB求交测试
__device__ bool intersectAABB(const glm::vec3& minBounds, const glm::vec3& maxBounds, const Ray& r);

// 场景求交（返回最近交点）
__device__ bool intersectScene(
    const Ray& r, float tMin, float tMax, HitRecord& rec,
    const BVHNodeGPU* bvhNodes,
    const TriangleGPU* triangles,
    const int* triIndices,
    const MaterialGpu* materials
);

// 阴影射线求交优化（只检查是否有交点）
__device__ bool shadowIntersectScene(
    const Ray& r, float tMin, float tMax,
    const BVHNodeGPU* bvhNodes,
    const TriangleGPU* triangles,
    const int* triIndices
);