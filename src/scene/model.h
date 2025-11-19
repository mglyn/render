#pragma once
#include <vector>
#include <string>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "shape.h"
#include "bvh/bvh.h"

// Model：管理一个三角网格及其BVH加速结构
class Model {
public:
    Model();
    explicit Model(Material mat);

    // 从 OBJ 文件加载数据（只支持 v / vn / vt / f）
    bool loadObj(const std::string &path, const Material &mat);
    void buildBVH(int maxLeafSize = 4);

    // 设置模型变换
    void setPosition(const glm::vec3& pos);
    void setRotation(const glm::vec3& rot); // 欧拉角 (度)
    void setScale(const glm::vec3& s);
    void updateModelMatrix(); // 公开此方法
    const glm::mat4& getModelMatrix() const { return modelMatrix_; }

    const std::vector<Triangle>& triangles() const { return triangles_; }
    const std::vector<int>& getTriangleIndices() const { return triIndices_; }
    const std::vector<BVHNode>& bvh() const { return bvh_; }
    const Material& material() const { return defaultMaterial_; }
    const std::vector<Material>& materials() const { return materials_; }
    const std::vector<int>& triangleMaterialIndices() const { return triMaterialIndices_; }
    bool empty() const { return triangles_.empty(); }

private:
    // void updateModelMatrix(); // 移至 public

    std::vector<Triangle> triangles_;
    std::vector<int> triIndices_;            // 构建BVH用的索引
    std::vector<BVHNode> bvh_;
    Material defaultMaterial_{};

    // 多材质支持
    std::vector<Material> materials_;        // 模型内所有材质表
    std::vector<int> triMaterialIndices_;    // 每个三角形对应的材质索引

    // 模型变换
    glm::vec3 position_{0.0f};
    glm::vec3 rotation_{0.0f}; // 欧拉角 (度)
    glm::vec3 scale_{1.0f};
    glm::mat4 modelMatrix_{1.0f};

    // 发光属性
    glm::vec3 emission_{0.0f, 0.0f, 0.0f};
};
