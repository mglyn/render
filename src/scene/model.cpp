#include "model.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <tuple>

static TrianglePOD makeTri(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c, const MaterialPOD &m)
{
    TrianglePOD t{};
    t.v0 = a;
    t.v1 = b;
    t.v2 = c;
    t.mat = m;
    return t;
}

bool Model::loadObj(const std::string &path, const MaterialPOD &mat)
{
    defaultMaterial_ = mat;
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
            }
        }
    }
    triIndices_.resize(triangles_.size());
    for (size_t i = 0; i < triangles_.size(); ++i)
        triIndices_[i] = static_cast<int>(i);
    return true;
}

// 递归构建BVH（中值划分，按最长轴）
int Model::buildRecursive(int begin, int end, int maxLeafSize)
{
    BVHNode node; // 初始化
    AABB bounds;
    for (int i = begin; i < end; ++i)
        bounds.expandTri(triangles_[triIndices_[i]]);
    node.bounds = bounds;
    int n = end - begin;
    if (n <= maxLeafSize)
    {
        node.start = begin;
        node.count = n;
        int idx = (int)bvh_.size();
        bvh_.push_back(node);
        return idx;
    }
    glm::vec3 ext = bounds.extent();
    int axis = 0;
    if (ext.y > ext.x && ext.y >= ext.z)
        axis = 1;
    else if (ext.z > ext.x && ext.z >= ext.y)
        axis = 2;
    float midCoord = 0.f;
    for (int i = begin; i < end; ++i)
    {
        const TrianglePOD &t = triangles_[triIndices_[i]];
        midCoord += (t.v0[axis] + t.v1[axis] + t.v2[axis]) / 3.f;
    }
    midCoord /= n;
    int pivot = std::partition(triIndices_.begin() + begin, triIndices_.begin() + end, [&](int triIdx)
                               { const TrianglePOD &t=triangles_[triIdx]; float c=(t.v0[axis]+t.v1[axis]+t.v2[axis])/3.f; return c < midCoord; }) -
                triIndices_.begin();
    if (pivot == begin || pivot == end)
        pivot = begin + n / 2; // 退化处理
    int idx = (int)bvh_.size();
    bvh_.push_back(node);
    int left = buildRecursive(begin, pivot, maxLeafSize);
    int right = buildRecursive(pivot, end, maxLeafSize);
    bvh_[idx].left = left;
    bvh_[idx].right = right;
    return idx;
}

void Model::buildBVH(int maxLeafSize)
{
    bvh_.clear();
    if (triangles_.empty())
        return;
    buildRecursive(0, (int)triangles_.size(), maxLeafSize);
}
