#pragma once
#include <vector>
#include <string>
#include <memory>
#include <glm/glm.hpp>
#include "shape.h"
#include "bvh/bvh.h"

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
    
    // 调试信息
    BVHDebugInfo debugInfo_;
};
