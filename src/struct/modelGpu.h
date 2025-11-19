#pragma once

#include <glm/glm.hpp>

#include "struct/bvhNodeGpu.h"
#include "struct/shapeGpu.h"

// GPU端模型数据结构
struct ModelGPU {
    BVHNodeGPU* bvhNodes;
    int nodeCount;

    int* triangleIndices;
    TriangleGPU* triangles;
    int triangleCount;

    MaterialGPU* materials;
    int materialCount;
};