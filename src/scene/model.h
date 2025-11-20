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
    explicit Model(Material mat, glm::vec3 emission = glm::vec3(0.0f));

    // 从 OBJ 文件加载数据（只支持 v / vn / vt / f）
    bool loadObj(const std::string &path, const Material &mat);

    const std::string& getName() const { return name_; }

    // 设置模型变换
    void setPosition(const glm::vec3& pos);
    void setRotation(const glm::vec3& rot); // 欧拉角 (度)
    void setScale(const glm::vec3& s);
    void updateModelMatrix(); // 公开此方法
    const glm::mat4& getModelMatrix() const { return modelMatrix_; }

    const glm::vec3& getPosition() const { return position_; }
    const glm::vec3& getRotation() const { return rotation_; }
    const glm::vec3& getScale() const { return scale_; }

    const std::vector<Triangle>& triangles() const { return triangles_; }
    const std::vector<int>& getTriangleIndices() const { return triIndices_; }
    const Material& material() const { return defaultMaterial_; }
    const glm::vec3& emission() const { return emission_; }
    bool empty() const { return triangles_.empty(); }

    void addTriangle(const Triangle& triangle);
    void clear();

    // 材质修改方法
    void setAllTrianglesMaterial(const Material& material);
    void setEmission(const glm::vec3& emission);

private:
    std::vector<Triangle> triangles_;
    std::vector<int> triIndices_;            // 构建BVH用的索引
    Material defaultMaterial_{};

    // 模型变换
    glm::vec3 position_{0.0f};
    glm::vec3 rotation_{0.0f}; // 欧拉角 (度)
    glm::vec3 scale_{1.0f};
    glm::mat4 modelMatrix_{1.0f};

    // 发光属性
    glm::vec3 emission_{0.0f, 0.0f, 0.0f};

    // 模型名称（文件路径）
    std::string name_;
};
