#pragma once
#include "renderer_base.h"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <vector>
#include "scene/camera.h"
#include "scene/shape.h"
#include "gpu/gpu_resources.h"

class WireframeRenderer : public RendererBase {
public:
    bool init(int w,int h, GPUResources* gpu) override;
    void renderFrame(Camera& camera) override;
    void destroy() override;
    const char* name() const override { return "Wireframe"; }
private:
    int _width=0,_height=0; GPUResources* _gpu=nullptr; GLuint _prog=0,_vao=0,_vbo=0,_ebo=0; std::vector<Shape> _shapes;
    bool initShaders(); bool initQuad(); std::string loadFile(const char* path); GLuint compileShader(GLenum t,const char* src);
};
