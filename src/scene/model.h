#pragma once
#include <vector>
#include <string>
#include <memory>
#include <limits>
#include <glm/glm.hpp>
#include "shape.h"

// 轴对齐包围盒 (AABB)
struct AABB {
    glm::vec3 min{ std::numeric_limits<float>::infinity() };
    glm::vec3 max{ -std::numeric_limits<float>::infinity() };

    void expand(const glm::vec3 &p) {
        min = glm::min(min, p);
        max = glm::max(max, p);
    }
    void expandTri(const TrianglePOD &t) {
        expand(t.v0); expand(t.v1); expand(t.v2);
    }
    glm::vec3 centroid() const { return 0.5f * (min + max); }
    glm::vec3 extent() const { return max - min; }
    bool valid() const { return min.x <= max.x && min.y <= max.y && min.z <= max.z; }
    
    // SAH相关方法
    float surfaceArea() const {
        if (!valid()) return 0.0f;
        glm::vec3 d = extent();
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
    
    // 计算三角形重心（用于SAH划分）
    glm::vec3 getTriangleCentroid(const TrianglePOD &t) const {
        return (t.v0 + t.v1 + t.v2) / 3.0f;
    }
};

// BVH调试信息
struct BVHDebugInfo {
    int maxDepth = 0;
    int maxLeafTriangles = 0;
    int totalNodes = 0;
    int leafNodes = 0;
    int internalNodes = 0;
    float totalSurfaceArea = 0.0f;
    float avgLeafTriangles = 0.0f;
    float sahCost = 0.0f;
    
    void reset() {
        maxDepth = 0;
        maxLeafTriangles = 0;
        totalNodes = 0;
        leafNodes = 0;
        internalNodes = 0;
        totalSurfaceArea = 0.0f;
        avgLeafTriangles = 0.0f;
        sahCost = 0.0f;
    }
};

// BVH节点
struct BVHNode {
    AABB bounds;      // 包围盒
    int left = -1;    // 左孩子索引（内部节点）
    int right = -1;   // 右孩子索引（内部节点）
    int start = -1;   // 叶子：三角形起始索引
    int count = 0;    // 叶子：三角形数量
    bool isLeaf() const { return count > 0; }
};

// Model：管理一个三角网格及其BVH加速结构
class Model {
public:
    Model() = default;
    explicit Model(MaterialPOD mat): defaultMaterial_(mat) {}

    // 从 OBJ 文件加载数据（只支持 v / vn / vt / f）
    bool loadObj(const std::string &path, const MaterialPOD &mat);
    void buildBVH(int maxLeafSize = 4);

    const std::vector<TrianglePOD>& triangles() const { return triangles_; }
    const std::vector<int>& getTriangleIndices() const { return triIndices_; }
    const std::vector<BVHNode>& bvh() const { return bvh_; }
    const MaterialPOD& material() const { return defaultMaterial_; }
    bool empty() const { return triangles_.empty(); }

private:
    std::vector<TrianglePOD> triangles_;
    std::vector<int> triIndices_;     // 构建BVH用的索引
    std::vector<BVHNode> bvh_;
    MaterialPOD defaultMaterial_{};
    
    // SAH常数
    static constexpr float SAH_TRAVERSAL_COST = 1.0f;
    static constexpr float SAH_INTERSECTION_COST = 1.0f;
    static constexpr int SAH_BUCKETS = 12;

    int buildRecursive(int begin, int end, int maxLeafSize);
    int buildRecursiveSAH(int begin, int end, int maxLeafSize, int depth = 0);
    
    // 调试辅助方法
    void collectNodeStats(int nodeIdx, int depth, BVHDebugInfo& info) const;
    
    // 调试信息
    BVHDebugInfo debugInfo_;
};
