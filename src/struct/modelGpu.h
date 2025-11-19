#pragma once

#include <glm/glm.hpp>

#include "struct/bvhNodeGpu.h"
#include "struct/shape.h"

// GPU端模型数据结构
struct ModelGPU {
    BVHNodeGPU* bvhNodes;
    int nodeCount;

    int* triangleIndices;
    TrianglePOD* triangles;
    int triangleCount;
};