#include "gpu/intersect.h"

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

// 单个三角形求交
__device__ bool intersectTriangle(const TrianglePOD& tri, const Ray& r, float tMin, float tMax, HitRecord& rec) {
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
    rec.albedo = tri.mat.albedo;
    rec.metallic = tri.mat.metallic;
    rec.emission = tri.mat.emission;
    
    return true;
}

// BVH遍历求交
__device__ bool intersectBVH(const ModelGPU* models, int modelCount, const Ray& r, float tMin, float tMax, HitRecord& rec) {
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
                    
                    if (intersectTriangle(model.triangles[triIdx], r, tMin, closest, tempRec)) {
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

__device__ bool intersect(
    const Shape* shapes, int shapeCount, const Ray& r, float tMin, float tMax, HitRecord& rec) {
    bool hitAnything = false;
    float closest = tMax;
    for (int i = 0; i < shapeCount; ++i) {
        const Shape& s = shapes[i];
        HitRecord temp;
        bool hit = false;
        switch (s.type) {
            case SHAPE_SPHERE: {
                // 球体相交
                glm::vec3 oc = r.origin() - s.data.sph.center;
                float a = glm::dot(r.direction(), r.direction());
                float halfB = glm::dot(oc, r.direction());
                float c = glm::dot(oc, oc) - s.data.sph.radius * s.data.sph.radius;
                float discriminant = halfB * halfB - a * c;
                if (discriminant >= 0.0f) {
                    float sqrtD = sqrtf(discriminant);
                    float root = (-halfB - sqrtD) / a;
                    if (root < tMin || root > closest) {
                        root = (-halfB + sqrtD) / a;
                    }
                    if (root >= tMin && root <= closest) {
                        temp.t = root;
                        temp.point = r.at(root);
                        temp.normal = glm::normalize((temp.point - s.data.sph.center) / s.data.sph.radius);
                        temp.albedo = s.data.sph.mat.albedo;
                        temp.metallic = s.data.sph.mat.metallic;
                        temp.emission = s.data.sph.mat.emission;
                        hit = true;
                    }
                }
            } break;
            case SHAPE_TRIANGLE: {
                // Möller-Trumbore
                const glm::vec3& v0 = s.data.tri.v0;
                const glm::vec3& v1 = s.data.tri.v1;
                const glm::vec3& v2 = s.data.tri.v2;
                glm::vec3 edge1 = v1 - v0;
                glm::vec3 edge2 = v2 - v0;
                glm::vec3 pvec = glm::cross(r.direction(), edge2);
                float det = glm::dot(edge1, pvec);
                if (fabsf(det) > 1e-6f) {
                    float invDet = 1.0f / det;
                    glm::vec3 tvec = r.origin() - v0;
                    float u = glm::dot(tvec, pvec) * invDet;
                    if (u >= 0.0f && u <= 1.0f) {
                        glm::vec3 qvec = glm::cross(tvec, edge1);
                        float v = glm::dot(r.direction(), qvec) * invDet;
                        if (v >= 0.0f && u + v <= 1.0f) {
                            float t = glm::dot(edge2, qvec) * invDet;
                            if (t >= tMin && t <= closest) {
                                temp.t = t;
                                temp.point = r.at(t);
                                temp.normal = glm::normalize(glm::cross(edge1, edge2));
                                temp.albedo = s.data.tri.mat.albedo;
                                temp.metallic = s.data.tri.mat.metallic;
                                temp.emission = s.data.tri.mat.emission;
                                hit = true;
                            }
                        }
                    }
                }
            } break;
            case SHAPE_PLANE: {
                const glm::vec3& n = s.data.pln.normal;
                float denom = glm::dot(n, r.direction());
                if (fabsf(denom) > 1e-6f) {
                    float t = glm::dot(s.data.pln.point - r.origin(), n) / denom;
                    if (t >= tMin && t <= closest) {
                        temp.t = t;
                        temp.point = r.at(t);
                        temp.normal = n;
                        temp.albedo = s.data.pln.mat.albedo;
                        temp.metallic = s.data.pln.mat.metallic;
                        temp.emission = s.data.pln.mat.emission;
                        hit = true;
                    }
                }
            } break;
        }
        if (hit) {
            hitAnything = true;
            closest = temp.t;
            rec = temp;
        }
    }
    return hitAnything;
}