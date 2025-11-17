#include "wireframe_renderer.h"
#include <cstdio>
#include <iostream>

std::string WireframeRenderer::loadFile(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return {};
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
    if (!ok)
    {
        char buf[512];
        glGetShaderInfoLog(s, 512, nullptr, buf);
        std::cerr << "Shader err: " << buf << std::endl;
    }
    return s;
}
bool WireframeRenderer::initShaders()
{
    std::string vs = loadFile("shaders/fullscreen.vert");
    std::string fs = loadFile("shaders/texture.frag");
    if (vs.empty() || fs.empty())
    {
        if (vs.empty())
            vs = loadFile("src/shaders/fullscreen.vert");
        if (fs.empty())
            fs = loadFile("src/shaders/texture.frag");
    }
    if (vs.empty() || fs.empty())
    {
        std::cerr << "Wireframe shaders missing" << std::endl;
        return false;
    }
    GLuint v = compileShader(GL_VERTEX_SHADER, vs.c_str());
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs.c_str());
    _prog = glCreateProgram();
    glAttachShader(_prog, v);
    glAttachShader(_prog, f);
    glLinkProgram(_prog);
    glDeleteShader(v);
    glDeleteShader(f);
    return true;
}
bool WireframeRenderer::initQuad()
{
    float quadVerts[] = {-1, -1, 0, 0, 1, -1, 1, 0, 1, 1, 1, 1, -1, 1, 0, 1};
    unsigned int idx[] = {0, 1, 2, 2, 3, 0};
    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_vbo);
    glGenBuffers(1, &_ebo);
    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
    glBindVertexArray(0);
    return true;
}

bool WireframeRenderer::init(int w, int h, GPUResources *gpu)
{
    _width = w;
    _height = h;
    _gpu = gpu;
    if (!initShaders())
        return false;
    if (!initQuad())
        return false; // scene
    {
        Shape s{};
        s.type = SHAPE_TRIANGLE;
        s.data.tri.v0 = glm::vec3(-0.8f, -0.2f, -2.0f);
        s.data.tri.v1 = glm::vec3(0.2f, 0.6f, -2.2f);
        s.data.tri.v2 = glm::vec3(0.9f, -0.3f, -2.1f);
        s.data.tri.mat.albedo = glm::vec3(1);
        s.data.tri.mat.metallic = 0;
        s.data.tri.mat.emission = glm::vec3(0);
        _shapes.push_back(s);
    }
    return true;
}

static inline unsigned packRGBA(const glm::vec3 &c)
{
    unsigned r = (unsigned)(255.99f * c.r);
    unsigned g = (unsigned)(255.99f * c.g);
    unsigned b = (unsigned)(255.99f * c.b);
    return (0xFFu << 24) | (b << 16) | (g << 8) | r;
}

// 简单双端点连线 (Bresenham 近似浮点版)
static void drawLine(unsigned int *buf, int W, int H, int x0, int y0, int x1, int y1, const glm::vec3 &col)
{
    auto put = [&](int x, int y)
    { if(x>=0&&x<W&&y>=0&&y<H) buf[y*W+x]=packRGBA(col); };
    int dx = abs(x1 - x0), dy = -abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    while (true)
    {
        put(x0, y0);
        if (x0 == x1 && y0 == y1)
            break;
        int e2 = 2 * err;
        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        }
    }
}

void WireframeRenderer::renderFrame(Camera &camera)
{
    GLuint pbo = _gpu->getWritePBO();
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    unsigned int *ptr = (unsigned int *)glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, _width * _height * 4, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    if (!ptr)
    {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        return;
    }
    std::fill(ptr, ptr + _width * _height, packRGBA(glm::vec3(0))); // clear
    glm::vec3 camPos = camera.getPosition();
    glm::vec3 front = glm::normalize(camera.getFront());
    glm::vec3 right = glm::normalize(camera.getRight());
    glm::vec3 up = glm::normalize(camera.getUp());
    float aspect = float(_width) / float(_height);
    float fovRad = camera.getFov() * 0.0174532925f;
    float viewportH = 2.0f * tanf(fovRad * 0.5f);
    float viewportW = aspect * viewportH;
    glm::vec3 horiz = viewportW * right;
    glm::vec3 vert = viewportH * up;
    glm::vec3 lowerLeft = camPos + front - horiz * 0.5f - vert * 0.5f;
    auto project = [&](const glm::vec3 &p)
    { glm::vec3 dir = p - camPos; float depth=glm::dot(dir,front); if(depth<=0.001f) depth=0.001f; glm::vec3 planePoint = camPos + front*depth; glm::vec3 local = p - planePoint; float u = glm::dot(local,right)/(viewportW*0.5f); float v = glm::dot(local,up)/(viewportH*0.5f); int sx = int((u+1.f)*0.5f*_width); int sy = int((v+1.f)*0.5f*_height); return glm::ivec2(sx,sy); };
    for (const auto &sh : _shapes)
    {
        if (sh.type == SHAPE_TRIANGLE)
        {
            glm::ivec2 a = project(sh.data.tri.v0);
            glm::ivec2 b = project(sh.data.tri.v1);
            glm::ivec2 c = project(sh.data.tri.v2);
            glm::vec3 col(1, 1, 1);
            drawLine(ptr, _width, _height, a.x, a.y, b.x, b.y, col);
            drawLine(ptr, _width, _height, b.x, b.y, c.x, c.y, col);
            drawLine(ptr, _width, _height, c.x, c.y, a.x, a.y, col);
        }
    }
    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    _gpu->finalizeWrite();
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _gpu->getDisplayPBO());
    glBindTexture(GL_TEXTURE_2D, _gpu->getTexture());
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(_prog);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _gpu->getTexture());
    glUniform1i(glGetUniformLocation(_prog, "uTexture"), 0);
    glBindVertexArray(_vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void WireframeRenderer::destroy()
{
    if (_prog)
    {
        glDeleteProgram(_prog);
        _prog = 0;
    }
    if (_vao)
    {
        glDeleteVertexArrays(1, &_vao);
        _vao = 0;
    }
    if (_vbo)
    {
        glDeleteBuffers(1, &_vbo);
        _vbo = 0;
    }
    if (_ebo)
    {
        glDeleteBuffers(1, &_ebo);
        _ebo = 0;
    }
    _shapes.clear();
}
