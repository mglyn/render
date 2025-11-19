#pragma once
#include <optional>

#include "struct/ray.h"
#include "struct/shape.h"
#include "struct/shapeGpu.h"
#include "renderer/hitRecord.h"
#include "struct/modelGpu.h"

// 单个三角形求交辅助函数（GPU 版，使用材质索引）
__device__ bool intersectTriangle(
    const TriangleGPU& tri, const MaterialGPU* materials, const Ray& r, float tMin, float tMax, HitRecord& rec);

// AABB求交测试
__device__ bool intersectAABB(const glm::vec3& minBounds, const glm::vec3& maxBounds, const Ray& r);

template <typename T>
requires std::is_same_v<T, ModelGPU> || std::is_same_v<T, IlluminantGPU>
__device__ bool intersectModels(const T* models, int modelCount, 
    const Ray& r, float tMin, float tMax, HitRecord& rec) 
{
    bool hitAnything = false;
    float closest = tMax;
    HitRecord tempRec;
    
    for (int modelIdx = 0; modelIdx < modelCount; ++modelIdx) {
        const T& model = models[modelIdx];
        if (model.nodeCount == 0 || !model.bvhNodes) continue;
        
        // BVH栈遍历
        int stack[64];
        int stackPtr = 0;
        stack[stackPtr++] = 0; // 根节点
        
        while (stackPtr > 0) {
            int nodeIdx = stack[--stackPtr];
            const BVHNodeGPU& node = model.bvhNodes[nodeIdx];
            
            // AABB测试
            if (!intersectAABB(node.minBounds, node.maxBounds, r)) continue;
            
            if (node.count > 0) {
                // 叶子节点：使用三角形索引数组
                for (int i = 0; i < node.count; ++i) {
                    int idxInIndices = node.start + i;
                    if (idxInIndices >= model.triangleCount) continue; // 防止索引越界
                    
                    int triIdx = model.triangleIndices ? model.triangleIndices[idxInIndices] : idxInIndices;
                    if (triIdx >= model.triangleCount) continue;
                    
                    if (intersectTriangle(model.triangles[triIdx], model.materials, r, tMin, closest, tempRec)) {
                        tempRec.primitiveIndex = triIdx;
                        tempRec.objectIndex = modelIdx;
                        if constexpr (std::is_same_v<T, IlluminantGPU>) {
                            tempRec.emission = model.emission;
                            tempRec.fromIlluminant = 1;
                        } else {
                            tempRec.emission = glm::vec3(0.0f);
                            tempRec.fromIlluminant = 0;
                        }
                        hitAnything = true;
                        closest = tempRec.t;
                        rec = tempRec;
                    }
                }
            } else {
                // 内部节点：基于射线方向优化子节点访问顺序
                // 先访问射线方向更可能命中的子节点
                bool visitLeftFirst = true;
                if (node.left >= 0 && node.right >= 0) {
                    // 计算射线方向与子节点位置关系
                    const BVHNodeGPU& leftChild = model.bvhNodes[node.left];
                    const BVHNodeGPU& rightChild = model.bvhNodes[node.right];
                    
                    // 比较子节点中心在主要方向上的位置
                    glm::vec3 rayDir = r.direction();
                    glm::vec3 leftCenter = (leftChild.minBounds + leftChild.maxBounds) * 0.5f;
                    glm::vec3 rightCenter = (rightChild.minBounds + rightChild.maxBounds) * 0.5f;
                    glm::vec3 rayOrigin = r.origin();
                    
                    float leftDot = glm::dot(rayDir, leftCenter - rayOrigin);
                    float rightDot = glm::dot(rayDir, rightCenter - rayOrigin);
                    
                    visitLeftFirst = leftDot >= rightDot;
                }
                
                // 按优化顺序添加子节点到栈
                if (visitLeftFirst) {
                    if (node.right >= 0 && stackPtr < 63) stack[stackPtr++] = node.right;
                    if (node.left >= 0 && stackPtr < 63) stack[stackPtr++] = node.left;
                } else {
                    if (node.left >= 0 && stackPtr < 63) stack[stackPtr++] = node.left;
                    if (node.right >= 0 && stackPtr < 63) stack[stackPtr++] = node.right;
                }
            }
        }
    }
    
    return hitAnything;
}

// Shadow ray：只关心是否被任何非光源遮挡（不需要返回具体 HitRecord）
template <typename T>
requires std::is_same_v<T, ModelGPU> || std::is_same_v<T, IlluminantGPU>
__device__ bool anyHit(const T* models, int modelCount,
    const Ray& r, float tMin, float tMax)
{
    // 1. 检查 BVH 模型是否有任意命中
    for (int modelIdx = 0; modelIdx < modelCount; ++modelIdx) {
        const T& model = models[modelIdx];
        if (model.nodeCount == 0 || !model.bvhNodes) continue;

        int stack[64];
        int stackPtr = 0;
        stack[stackPtr++] = 0;

        while (stackPtr > 0) {
            int nodeIdx = stack[--stackPtr];
            const BVHNodeGPU& node = model.bvhNodes[nodeIdx];

            if (!intersectAABB(node.minBounds, node.maxBounds, r)) continue;

            if (node.count > 0) {
                for (int i = 0; i < node.count; ++i) {
                    int idxInIndices = node.start + i;
                    if (idxInIndices >= model.triangleCount) continue;
                    int triIdx = model.triangleIndices ? model.triangleIndices[idxInIndices] : idxInIndices;
                    if (triIdx >= model.triangleCount) continue;

                    HitRecord tmp;
                    if (intersectTriangle(model.triangles[triIdx], model.materials, r, tMin, tMax, tmp)) {
                        return true;
                    }
                }
            } else {
                if (node.left >= 0 && stackPtr < 63) stack[stackPtr++] = node.left;
                if (node.right >= 0 && stackPtr < 63) stack[stackPtr++] = node.right;
            }
        }
    }
    return false;
}
