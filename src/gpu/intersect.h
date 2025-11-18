#pragma once
#include <optional>

#include "gpu/ray.h"
#include "scene/shape.h"

// 遍历 Shape 数组求交，返回最近命中
__device__ bool intersect(
    const Shape* shapes, int shapeCount, const Ray& r, float tMin, float tMax, HitRecord& rec);