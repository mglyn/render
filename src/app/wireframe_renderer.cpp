#include "wireframe_renderer.h"
#include "scene/scene.h"
#include <cstdio>
#include <iostream>
#include <glm/gtc/type_ptr.hpp>

std::string WireframeRenderer::loadFile(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string s;
    s.resize(sz);
    fread(&s[0], 1, sz, f);
    fclose(f);
    return s;
}

GLuint WireframeRenderer::compileShader(GLenum t, const char *src)
{
    GLuint s = glCreateShader(t);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[512];
        glGetShaderInfoLog(s, 512, nullptr, buf);
        std::cerr << "Shader err: " << buf << std::endl;
    }
    return s;
}

bool WireframeRenderer::initShaders()
{
    std::string vs_str = loadFile("shaders/wireframe.vert");
    std::string fs_str = loadFile("shaders/wireframe.frag");
    if (vs_str.empty() || fs_str.empty()) {
        vs_str = loadFile("src/shaders/wireframe.vert");
        fs_str = loadFile("src/shaders/wireframe.frag");
    }
    if (vs_str.empty() || fs_str.empty()) {
        std::cerr << "Wireframe shaders missing" << std::endl;
        return false;
    }

    GLuint v = compileShader(GL_VERTEX_SHADER, vs_str.c_str());
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs_str.c_str());
    _prog = glCreateProgram();
    glAttachShader(_prog, v);
    glAttachShader(_prog, f);
    glLinkProgram(_prog);
    glDeleteShader(v);
    glDeleteShader(f);

    _mvpLoc = glGetUniformLocation(_prog, "mvp");
    _colorLoc = glGetUniformLocation(_prog, "color");
    return true;
}

bool WireframeRenderer::init(int w, int h, GPUResources *gpu, Scene* scene)
{
    _width = w;
    _height = h;
    scene_ = scene;

    if (!initShaders()) return false;

    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_vbo);
    glGenBuffers(1, &_ebo);

    return true;
}

void WireframeRenderer::renderFrame(Camera &camera)
{
    // 1. 从 Scene 构建顶点和索引数据
    _vertices.clear();
    _indices.clear();
    unsigned int baseIndex = 0;
    if (scene_) {
        for (const auto &sh : scene_->getShapes()) {
            if (sh.type == SHAPE_TRIANGLE) {
                _vertices.push_back(sh.data.tri.v0);
                _vertices.push_back(sh.data.tri.v1);
                _vertices.push_back(sh.data.tri.v2);
                _indices.push_back(baseIndex); _indices.push_back(baseIndex + 1);
                _indices.push_back(baseIndex + 1); _indices.push_back(baseIndex + 2);
                _indices.push_back(baseIndex + 2); _indices.push_back(baseIndex);
                baseIndex += 3;
            }
            // 可以扩展支持球体等其他图元的线框
        }
    }

    if (_vertices.empty()) return;

    // 2. 上传数据到 VBO 和 EBO
    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, _vertices.size() * sizeof(glm::vec3), _vertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, _indices.size() * sizeof(unsigned int), _indices.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    // 3. 设置渲染状态并绘制
    glUseProgram(_prog);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); // 设置为线框模式
    
    glm::mat4 proj = camera.getProjectionMatrix((float)_width / (float)_height);
    glm::mat4 view = camera.getViewMatrix();
    glm::mat4 mvp = proj * view; // Model矩阵为单位矩阵

    glUniformMatrix4fv(_mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform3f(_colorLoc, 1.0f, 1.0f, 1.0f); // 白色

    glBindVertexArray(_vao);
    glDrawElements(GL_LINES, (GLsizei)_indices.size(), GL_UNSIGNED_INT, 0);
    
    // 恢复状态
    glBindVertexArray(0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glUseProgram(0);
}

void WireframeRenderer::destroy()
{
    if (_prog) { glDeleteProgram(_prog); _prog = 0; }
    if (_vao) { glDeleteVertexArrays(1, &_vao); _vao = 0; }
    if (_vbo) { glDeleteBuffers(1, &_vbo); _vbo = 0; }
    if (_ebo) { glDeleteBuffers(1, &_ebo); _ebo = 0; }
}
