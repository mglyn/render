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
    rec.materialIndex = tri.materialIndex;
    rec.emission = glm::vec3(0.0f); // 默认不发光，由 intersectModels 根据模型类型设置
    
    return true;
}

