#include "model.h"
#include "bvh/bvh.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <tuple>

Model::Model() {
    updateModelMatrix();
}

Model::Model(Material mat, glm::vec3 emission) : 
defaultMaterial_(mat), 
emission_(emission) 
{
    // 单材质构造时，将默认材质加入材质表
    materials_.push_back(defaultMaterial_);
    updateModelMatrix();
}

static Triangle makeTri(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c, const Material &m){
    // This function is now insufficient as it doesn't handle normals and texcoords.
    // It's better to construct Triangles directly.
    Triangle t;
    t.v0 = a; t.v1 = b; t.v2 = c;
    t.mat = m;
    // Normals and texcoords will be uninitialized, must be set elsewhere.
    return t;
}

bool Model::loadObj(const std::string &path, const Material &mat)
{
    defaultMaterial_ = mat;
    materials_.clear();
    triMaterialIndices_.clear();
    // 目前先简单地将传入材质作为唯一材质，所有三角形使用 index 0
    materials_.push_back(defaultMaterial_);
    std::ifstream ifs(path);
    if (!ifs.is_open())
        return false;
    std::vector<glm::vec3> positions;
    positions.reserve(1024);
    std::vector<glm::vec3> normals;
    normals.reserve(1024);
    std::vector<glm::vec2> texcoords;
    texcoords.reserve(1024);
    std::string line;
    while (std::getline(ifs, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream iss(line);
        std::string tag;
        iss >> tag;
        if (tag == "v")
        {
            glm::vec3 p;
            iss >> p.x >> p.y >> p.z;
            positions.push_back(p);
        }
        else if (tag == "vn")
        {
            glm::vec3 n;
            iss >> n.x >> n.y >> n.z;
            normals.push_back(n);
        }
        else if (tag == "vt")
        {
            glm::vec2 uv;
            iss >> uv.x >> uv.y;
            texcoords.push_back(uv);
        }
        else if (tag == "f")
        {
            // 支持三角或多边形，进行扇形拆分
            std::vector<std::string> verts;
            std::string vstr;
            while (iss >> vstr)
                verts.push_back(vstr);
            if (verts.size() < 3)
                continue;
            auto parseIdx = [](const std::string &s)
            {
                // 形如 v/vt/vn 或 v//vn 或 v/vt
                int v = -1, vt = -1, vn = -1;
                std::stringstream ss(s);
                std::string item;
                int idx = 0;
                while (std::getline(ss, item, '/'))
                {
                    if (!item.empty())
                    {
                        int val = std::stoi(item) - 1; // OBJ indices are 1-based
                        if (idx == 0) v = val;
                        else if (idx == 1) vt = val;
                        else if (idx == 2) vn = val;
                    }
                    idx++;
                }
                return std::tuple<int, int, int>(v, vt, vn);
            };

            auto getPos = [&](int i) { return (i >= 0 && i < positions.size()) ? positions[i] : glm::vec3(0.f); };
            auto getNorm = [&](int i) { return (i >= 0 && i < normals.size()) ? normals[i] : glm::vec3(0.f); };
            auto getTex = [&](int i) { return (i >= 0 && i < texcoords.size()) ? texcoords[i] : glm::vec2(0.f); };
            
            // Fan triangulation: v0, v_k, v_{k+1}
            auto t0_indices = parseIdx(verts[0]);

            for (size_t k = 1; k + 1 < verts.size(); ++k)
            {
                auto t1_indices = parseIdx(verts[k]);
                auto t2_indices = parseIdx(verts[k + 1]);

                int v0i = std::get<0>(t0_indices);
                int v1i = std::get<0>(t1_indices);
                int v2i = std::get<0>(t2_indices);

                if (v0i < 0 || v1i < 0 || v2i < 0) continue;

                Triangle tri;
                tri.v0 = getPos(v0i);
                tri.v1 = getPos(v1i);
                tri.v2 = getPos(v2i);

                int n0i = std::get<2>(t0_indices);
                int n1i = std::get<2>(t1_indices);
                int n2i = std::get<2>(t2_indices);
                tri.n0 = getNorm(n0i);
                tri.n1 = getNorm(n1i);
                tri.n2 = getNorm(n2i);
                // If no normals in file, compute flat normal
                if (n0i < 0 || n1i < 0 || n2i < 0) {
                    glm::vec3 flat_normal = glm::normalize(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                    tri.n0 = tri.n1 = tri.n2 = flat_normal;
                }


                int t0i = std::get<1>(t0_indices);
                int t1i = std::get<1>(t1_indices);
                int t2i = std::get<1>(t2_indices);
                tri.t0 = getTex(t0i);
                tri.t1 = getTex(t1i);
                tri.t2 = getTex(t2i);

                tri.mat = defaultMaterial_;
                triangles_.push_back(tri);
                triMaterialIndices_.push_back(0);
            }
        }
    }
    triIndices_.resize(triangles_.size());
    for (size_t i = 0; i < triangles_.size(); ++i)
        triIndices_[i] = static_cast<int>(i);
    return true;
}

void Model::buildBVH(){
    if (triangles_.empty()) return;
    
    std::vector<Triangle> worldTriangles = triangles_;
    for (auto& tri : worldTriangles) {
        tri.v0 = glm::vec3(modelMatrix_ * glm::vec4(tri.v0, 1.0f));
        tri.v1 = glm::vec3(modelMatrix_ * glm::vec4(tri.v1, 1.0f));
        tri.v2 = glm::vec3(modelMatrix_ * glm::vec4(tri.v2, 1.0f));
    }
    
    // Rebuild triIndices before calling bvh::build
    triIndices_.resize(worldTriangles.size());
    for (size_t i = 0; i < worldTriangles.size(); ++i) {
        triIndices_[i] = static_cast<int>(i);
    }

    bvh::build(bvh_, worldTriangles, triIndices_, 4); // Assuming maxLeafSize of 4 as a default
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
    triMaterialIndices_.push_back(0);
    if (materials_.empty()) {
        materials_.push_back(defaultMaterial_);
    }
}


