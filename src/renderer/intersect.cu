#include "renderer/intersect.h"
#include "struct/shapeGpu.h"

// AABB与射线求交测试
__device__ bool intersectAABB(const glm::vec3& minBounds, const glm::vec3& maxBounds, const Ray& r) {
    const glm::vec3& orig = r.origin();
    const glm::vec3& dir = r.direction();
    
    float tmin = 0.0f;
    float tmax = FLT_MAX;
    
    for (int i = 0; i < 3; ++i) {
        if (fabsf(dir[i]) < 1e-8f) {
            // 射线平行于坐标轴
            if (orig[i] < minBounds[i] || orig[i] > maxBounds[i]) {
                return false;
            }
        } else {
            float invDir = 1.0f / dir[i];
            float t1 = (minBounds[i] - orig[i]) * invDir;
            float t2 = (maxBounds[i] - orig[i]) * invDir;
            
            if (t1 > t2) {
                float tmp = t1; t1 = t2; t2 = tmp;
            }
            
            tmin = fmaxf(tmin, t1);
            tmax = fminf(tmax, t2);
            
            if (tmin > tmax) {
                return false;
            }
        }
    }
    
    return tmax >= 0.0f;
}

// 单个三角形求交（使用 TriangleGPU 与材质索引）
__device__ bool intersectTriangle(const TriangleGPU& tri, const MaterialGPU* materials, const Ray& r, float tMin, float tMax, HitRecord& rec) {
    const glm::vec3& v0 = tri.v0;
    const glm::vec3& v1 = tri.v1;
    const glm::vec3& v2 = tri.v2;
    
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 pvec = glm::cross(r.direction(), edge2);
    float det = glm::dot(edge1, pvec);
    
    if (fabsf(det) < 1e-6f) return false;
    
    float invDet = 1.0f / det;
    glm::vec3 tvec = r.origin() - v0;
    float u = glm::dot(tvec, pvec) * invDet;
    
    if (u < 0.0f || u > 1.0f) return false;
    
    glm::vec3 qvec = glm::cross(tvec, edge1);
    float v = glm::dot(r.direction(), qvec) * invDet;
    
    if (v < 0.0f || u + v > 1.0f) return false;
    
    float t = glm::dot(edge2, qvec) * invDet;
    
    if (t < tMin || t > tMax) return false;
    
    rec.t = t;
    rec.point = r.at(t);
    rec.normal = glm::normalize(glm::cross(edge1, edge2));

    const MaterialGPU& mat = materials[tri.materialIndex];
    rec.albedo = mat.albedo;
    rec.metallic = mat.metallic;
    
    return true;
}

// BVH遍历求交
__device__ bool intersectBVH(const ModelGPU* models, int modelCount, 
    const Ray& r, float tMin, float tMax, HitRecord& rec) {
    bool hitAnything = false;
    float closest = tMax;
    HitRecord tempRec;
    
    for (int modelIdx = 0; modelIdx < modelCount; ++modelIdx) {
        const ModelGPU& model = models[modelIdx];
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
__device__ bool anyHit(const ModelGPU* models, int modelCount,
    const Ray& r, float tMin, float tMax)
{
    // 1. 检查 BVH 模型是否有任意命中
    for (int modelIdx = 0; modelIdx < modelCount; ++modelIdx) {
        const ModelGPU& model = models[modelIdx];
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