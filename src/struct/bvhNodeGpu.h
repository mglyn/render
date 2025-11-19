#pragma once

#include <glm/glm.hpp>

// GPU端BVH节点结构（POD）
struct BVHNodeGPU {
    glm::vec3 minBounds;
    float pad1;
    glm::vec3 maxBounds; 
    float pad2;
    int left;     // 左孩子索引，-1表示叶子节点
    int right;    // 右孩子索引
    int start;    // 叶子节点：三角形起始索引
    int count;    // 叶子节点：三角形数量
};