#pragma once
#include <optional>

#include "struct/ray.h"
#include "struct/shape.h"
#include "struct/shapeGpu.h"
#include "renderer/hitRecord.h"
#include "struct/modelGpu.h"

// Shadow ray 求交：检测在 [tMin, tMax] 区间内是否被任意物体遮挡（BVH 模型 + 基础 Shape）
__device__ bool anyHit(const ModelGPU* models, int modelCount,
    const Ray& r, float tMin, float tMax);

// BVH加速求交
__device__ bool intersectBVH(
    const ModelGPU* models, int modelCount, const Ray& r, float tMin, float tMax, HitRecord& rec);

// 单个三角形求交辅助函数（GPU 版，使用材质索引）
__device__ bool intersectTriangle(
    const TriangleGPU& tri, const MaterialGPU* materials, const Ray& r, float tMin, float tMax, HitRecord& rec);

// AABB求交测试
__device__ bool intersectAABB(const glm::vec3& minBounds, const glm::vec3& maxBounds, const Ray& r);