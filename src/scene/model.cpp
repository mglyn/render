#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <tuple>

#include "model.h"
#include "bvh/bvh.h"
#include <rapidobj/rapidobj.hpp>

Model::Model() : name_("") {
    updateModelMatrix();
}

Model::Model(Material mat, glm::vec3 emission) : 
defaultMaterial_(mat), 
emission_(emission),
name_("")
{
    updateModelMatrix();
}

bool Model::loadObj(const std::string &path, const Material &mat)
{
    name_ = path; // 设置模型名称为文件路径
    defaultMaterial_ = mat;

    // 使用rapidobj解析OBJ文件
    auto result = rapidobj::ParseFile(path, rapidobj::MaterialLibrary::Ignore());
    
    if (result.error) {
        return false;
    }

    // 清空现有数据
    triangles_.clear();
    triIndices_.clear();

    // 辅助函数：从多边形顶点创建三角形
    auto createTriangle = [&](const std::vector<rapidobj::Index>& face_indices, size_t i0, size_t i1, size_t i2) {
        auto index0 = face_indices[i0];
        auto index1 = face_indices[i1];
        auto index2 = face_indices[i2];

        // 提取顶点位置
        glm::vec3 pos0{
            result.attributes.positions[index0.position_index * 3 + 0],
            result.attributes.positions[index0.position_index * 3 + 1],
            result.attributes.positions[index0.position_index * 3 + 2]
        };
        glm::vec3 pos1{
            result.attributes.positions[index1.position_index * 3 + 0],
            result.attributes.positions[index1.position_index * 3 + 1],
            result.attributes.positions[index1.position_index * 3 + 2]
        };
        glm::vec3 pos2{
            result.attributes.positions[index2.position_index * 3 + 0],
            result.attributes.positions[index2.position_index * 3 + 1],
            result.attributes.positions[index2.position_index * 3 + 2]
        };

        Triangle triangle{pos0, pos1, pos2};
        triangle.mat = defaultMaterial_;

        // 如果有法线数据，提取法线
        if (!result.attributes.normals.empty() && 
            index0.normal_index >= 0 && index1.normal_index >= 0 && index2.normal_index >= 0) {
            
            glm::vec3 normal0{
                result.attributes.normals[index0.normal_index * 3 + 0],
                result.attributes.normals[index0.normal_index * 3 + 1],
                result.attributes.normals[index0.normal_index * 3 + 2]
            };
            glm::vec3 normal1{
                result.attributes.normals[index1.normal_index * 3 + 0],
                result.attributes.normals[index1.normal_index * 3 + 1],
                result.attributes.normals[index1.normal_index * 3 + 2]
            };
            glm::vec3 normal2{
                result.attributes.normals[index2.normal_index * 3 + 0],
                result.attributes.normals[index2.normal_index * 3 + 1],
                result.attributes.normals[index2.normal_index * 3 + 2]
            };

            triangle.n0 = normal0;
            triangle.n1 = normal1;
            triangle.n2 = normal2;
        } else {
            // 计算平面法线
            glm::vec3 flat_normal = glm::normalize(glm::cross(pos1 - pos0, pos2 - pos0));
            triangle.n0 = triangle.n1 = triangle.n2 = flat_normal;
        }

        // 如果有纹理坐标，提取纹理坐标
        if (!result.attributes.texcoords.empty() && 
            index0.texcoord_index >= 0 && index1.texcoord_index >= 0 && index2.texcoord_index >= 0) {
            
            triangle.t0 = glm::vec2{
                result.attributes.texcoords[index0.texcoord_index * 2 + 0],
                result.attributes.texcoords[index0.texcoord_index * 2 + 1]
            };
            triangle.t1 = glm::vec2{
                result.attributes.texcoords[index1.texcoord_index * 2 + 0],
                result.attributes.texcoords[index1.texcoord_index * 2 + 1]
            };
            triangle.t2 = glm::vec2{
                result.attributes.texcoords[index2.texcoord_index * 2 + 0],
                result.attributes.texcoords[index2.texcoord_index * 2 + 1]
            };
        }

        triangles_.push_back(triangle);
    };

    for (const auto& shape : result.shapes) {
        size_t index_offset = 0;
        for (size_t num_face_vertices : shape.mesh.num_face_vertices) {
            if (num_face_vertices >= 3) {
                // 提取当前面的所有顶点索引
                std::vector<rapidobj::Index> face_indices;
                for (size_t i = 0; i < num_face_vertices; ++i) {
                    face_indices.push_back(shape.mesh.indices[index_offset + i]);
                }

                // 使用扇形三角化：第一个顶点连接到其他所有顶点对
                for (size_t i = 1; i + 1 < num_face_vertices; ++i) {
                    createTriangle(face_indices, 0, i, i + 1);
                }
            }
            index_offset += num_face_vertices;
        }
    }

    // 构建三角形索引
    triIndices_.resize(triangles_.size());
    for (size_t i = 0; i < triangles_.size(); ++i) {
        triIndices_[i] = static_cast<int>(i);
    }

    return true;
}

void Model::setPosition(const glm::vec3& pos) {
    position_ = pos;
}

void Model::setRotation(const glm::vec3& rot) {
    rotation_ = rot;
}

void Model::setScale(const glm::vec3& s) {
    scale_ = s;
}

void Model::updateModelMatrix() {
    modelMatrix_ = glm::mat4(1.0f);
    modelMatrix_ = glm::translate(modelMatrix_, position_);
    modelMatrix_ = glm::rotate(modelMatrix_, glm::radians(rotation_.x), glm::vec3(1, 0, 0));
    modelMatrix_ = glm::rotate(modelMatrix_, glm::radians(rotation_.y), glm::vec3(0, 1, 0));
    modelMatrix_ = glm::rotate(modelMatrix_, glm::radians(rotation_.z), glm::vec3(0, 0, 1));
    modelMatrix_ = glm::scale(modelMatrix_, scale_);
}

void Model::addTriangle(const Triangle& triangle) {
    triangles_.push_back(triangle);
    triIndices_.push_back(static_cast<int>(triangles_.size() - 1));
}

void Model::setAllTrianglesMaterial(const Material& material) {
    for (auto& triangle : triangles_) {
        triangle.mat = material;
    }
    defaultMaterial_ = material;
}

void Model::setEmission(const glm::vec3& emission) {
    emission_ = emission;
}

void Model::clear(){
    triangles_.clear();
    triIndices_.clear();
    defaultMaterial_ = Material();
    
    position_ = glm::vec3(0.0f);
    rotation_ = glm::vec3(0.0f); 
    scale_ = glm::vec3(1.0f);
    modelMatrix_ = glm::mat4(1.0f);

    emission_ = glm::vec3(0.0f, 0.0f, 0.0f);

    name_.clear();
}


