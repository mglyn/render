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

Model::Model(Material mat) : defaultMaterial_(mat) {
    // 单材质构造时，将默认材质加入材质表
    materials_.push_back(defaultMaterial_);
    updateModelMatrix();
}

static TrianglePOD makeTri(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c, const Material &m)
{
    TrianglePOD t{};
    t.v0 = a;
    t.v1 = b;
    t.v2 = c;
    t.mat = m;
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
                        int val = std::stoi(item) - 1;
                        if (idx == 0)
                            v = val;
                        else if (idx == 1)
                            vt = val;
                        else if (idx == 2)
                            vn = val;
                    }
                    idx++;
                }
                return std::tuple<int, int, int>(v, vt, vn);
            };
            auto getPos = [&](int i)
            { return positions[i]; };
            // 扇形: v0, vi, vi+1
            int v0i;
            {
                auto t0 = parseIdx(verts[0]);
                v0i = std::get<0>(t0);
            }
            for (size_t k = 1; k + 1 < verts.size(); ++k)
            {
                auto t1 = parseIdx(verts[k]);
                auto t2 = parseIdx(verts[k + 1]);
                int v1i = std::get<0>(t1);
                int v2i = std::get<0>(t2);
                if (v0i < 0 || v1i < 0 || v2i < 0)
                    continue;
                triangles_.push_back(makeTri(getPos(v0i), getPos(v1i), getPos(v2i), defaultMaterial_));
                triMaterialIndices_.push_back(0); // 先全部指向默认材质
            }
        }
    }
    triIndices_.resize(triangles_.size());
    for (size_t i = 0; i < triangles_.size(); ++i)
        triIndices_[i] = static_cast<int>(i);
    return true;
}

void Model::buildBVH(int maxLeafSize)
{
    std::vector<TrianglePOD> worldTriangles = triangles_;
    for (auto& tri : worldTriangles) {
        tri.v0 = glm::vec3(modelMatrix_ * glm::vec4(tri.v0, 1.0f));
        tri.v1 = glm::vec3(modelMatrix_ * glm::vec4(tri.v1, 1.0f));
        tri.v2 = glm::vec3(modelMatrix_ * glm::vec4(tri.v2, 1.0f));
    }
    bvh::build(bvh_, worldTriangles, triIndices_, maxLeafSize);
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


