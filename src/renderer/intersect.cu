#include "renderer/intersect.h"
#include "struct/shapeGpu.h"
#include "struct/materialGpu.h"

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

__device__ bool intersectTriangle(const TriangleGPU& tri, const Ray& r, float tMin, float tMax, HitRecord& rec) {
    // (The existing Möller-Trumbore intersection code from intersect.cu, but without the material part)
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
    float w = 1.0f - u - v;
    rec.normal = glm::normalize(w * tri.n0 + u * tri.n1 + v * tri.n2);
    rec.materialIndex = tri.materialIndex;
    
    return true;
}

__device__ bool intersectScene(
    const Ray& r, float tMin, float tMax, HitRecord& rec,
    const BVHNodeGPU* bvhNodes,
    const TriangleGPU* triangles,
    const int* triIndices,
    const MaterialGpu* materials
) {
    bool hit = false;
    float closest_so_far = tMax;
    int node_stack[64];
    int stack_ptr = 0;
    node_stack[stack_ptr++] = 0; // Start with root node

    while (stack_ptr > 0) {
        int node_idx = node_stack[--stack_ptr];
        const BVHNodeGPU& node = bvhNodes[node_idx];

        if (!intersectAABB(node.minBounds, node.maxBounds, r)) {
            continue;
        }

        if (node.count > 0) { // Leaf node
            for (int i = 0; i < node.count; ++i) {
                int tri_idx = triIndices[node.start + i];
                const TriangleGPU& tri = triangles[tri_idx];
                HitRecord temp_rec;
                if (intersectTriangle(tri, r, tMin, closest_so_far, temp_rec)) {
                    hit = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                    rec.primitiveIndex = tri_idx; // Store global triangle index
                }
            }
        } else { // Internal node
            // Push children to stack. Note: could be optimized by pushing farther child first
            node_stack[stack_ptr++] = node.left;
            node_stack[stack_ptr++] = node.right;
        }
    }
    return hit;
}

// 阴影射线求交优化：只检查是否有任何交点，不需要最近交点
__device__ bool shadowIntersectScene(
    const Ray& r, float tMin, float tMax,
    const BVHNodeGPU* bvhNodes,
    const TriangleGPU* triangles,
    const int* triIndices
) {
    int node_stack[64];
    int stack_ptr = 0;
    node_stack[stack_ptr++] = 0; // Start with root node

    while (stack_ptr > 0) {
        int node_idx = node_stack[--stack_ptr];
        const BVHNodeGPU& node = bvhNodes[node_idx];

        if (!intersectAABB(node.minBounds, node.maxBounds, r)) {
            continue;
        }

        if (node.count > 0) { // Leaf node
            for (int i = 0; i < node.count; ++i) {
                int tri_idx = triIndices[node.start + i];
                const TriangleGPU& tri = triangles[tri_idx];
                
                // 简化的三角形求交检查
                const glm::vec3& v0 = tri.v0;
                const glm::vec3& v1 = tri.v1;
                const glm::vec3& v2 = tri.v2;
                
                glm::vec3 edge1 = v1 - v0;
                glm::vec3 edge2 = v2 - v0;
                glm::vec3 pvec = glm::cross(r.direction(), edge2);
                float det = glm::dot(edge1, pvec);
                
                if (fabsf(det) < 1e-6f) continue;
                
                float invDet = 1.0f / det;
                glm::vec3 tvec = r.origin() - v0;
                float u = glm::dot(tvec, pvec) * invDet;
                
                if (u < 0.0f || u > 1.0f) continue;
                
                glm::vec3 qvec = glm::cross(tvec, edge1);
                float v = glm::dot(r.direction(), qvec) * invDet;
                
                if (v < 0.0f || u + v > 1.0f) continue;
                
                float t = glm::dot(edge2, qvec) * invDet;
                
                if (t >= tMin && t <= tMax) {
                    return true; // 找到交点，立即返回
                }
            }
        } else { // Internal node
            // Push children to stack
            node_stack[stack_ptr++] = node.left;
            node_stack[stack_ptr++] = node.right;
        }
    }
    return false; // 没有找到交点
}

