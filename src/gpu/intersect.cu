#include "gpu/intersect.h"

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