#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

#include "cuda_path_tracing_renderer.h"
#include "renderer/path_tracer.h"

std::string CudaPathTracingRenderer::loadFile(const char *path)
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

GLuint CudaPathTracingRenderer::compileShader(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char buf[1024];
        glGetShaderInfoLog(s, 1024, nullptr, buf);
        std::cerr << "Shader compile error: " << buf << std::endl;
    }
    return s;
}

bool CudaPathTracingRenderer::uploadScene() {
    if (!scene_ || scene_->getShapes().empty()) {
        _shapeCount = 0;
        if (_shapesDev) {
            cudaFree(_shapesDev);
            _shapesDev = nullptr;
        }
        return true; // No scene to upload, but not an error
    }

    const auto hostShapes = scene_->getShapes();
    _shapeCount = static_cast<int>(hostShapes.size());

    // Free existing memory if shape count differs or buffer not allocated
    if (_shapesDev) {
        // A more robust check would be to see if the scene content has actually changed.
        // For now, we re-upload if the pointer is different, which happens on init.
        cudaFree(_shapesDev);
        _shapesDev = nullptr;
    }

    cudaError_t err = cudaMalloc(&_shapesDev, _shapeCount * sizeof(Shape));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for shapes: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMemcpy(_shapesDev, hostShapes.data(), static_cast<size_t>(_shapeCount) * sizeof(Shape), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy shapes to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(_shapesDev);
        _shapesDev = nullptr;
        return false;
    }

    std::cout << "Uploaded " << _shapeCount << " shapes to GPU." << std::endl;
    return true;
}

bool CudaPathTracingRenderer::uploadBVH() {
    if (!scene_) return false;
    
    // 上传BVH数据到GPU
    if (!scene_->uploadBVHToGPU()) {
        std::cerr << "Failed to upload BVH to GPU" << std::endl;
        return false;
    }
    
    const auto& gpuModels = scene_->getGPUModels();
    _modelCount = static_cast<int>(gpuModels.size());
    
    if (_modelCount > 0) {
        // 分配并上传模型数据结构
        cudaError_t err = cudaMalloc(&_modelsDev, _modelCount * sizeof(ModelGPU));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate device memory for models: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        err = cudaMemcpy(_modelsDev, gpuModels.data(), _modelCount * sizeof(ModelGPU), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy models to device: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        std::cout << "Uploaded BVH data for " << _modelCount << " models to GPU." << std::endl;
    }
    
    return true;
}

bool CudaPathTracingRenderer::init(int width, int height, GPUResources *gpu, Scene* scene)
{
    _width = width;
    _height = height;
    _gpu = gpu;
    scene_ = scene; // Store scene pointer

    if (!initShaders())
        return false;
    if (!initQuad())
        return false;
    if (!allocateAccumBuffer())
        return false;
    if (!uploadScene()) // Upload scene data to GPU
        return false;
    
    // 上传BVH模型数据
    if (!uploadBVH())
        return false;
    
    // 清除脏标记，避免在第一次renderFrame时重复上传
    if (scene_) {
        scene_->clearDirty();
    }

    return true;
}

bool CudaPathTracingRenderer::initShaders()
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
        std::cerr << "Failed to load shaders." << std::endl;
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

bool CudaPathTracingRenderer::initQuad()
{
    float quadVerts[] = {-1.f, -1.f, 0.f, 0.f, 1.f, -1.f, 1.f, 0.f, 1.f, 1.f, 1.f, 1.f, -1.f, 1.f, 0.f, 1.f};
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

bool CudaPathTracingRenderer::allocateAccumBuffer()
{
    size_t pixelCount = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    if (pixelCount == 0)
        return false;
    cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&_accumBufferDev), pixelCount * sizeof(glm::vec3));
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate accumulation buffer: " << cudaGetErrorString(err) << std::endl;
        _accumBufferDev = nullptr;
        return false;
    }
    cudaMemset(_accumBufferDev, 0, pixelCount * sizeof(glm::vec3));
    _accumFrameCount = 0;
    return true;
}

void CudaPathTracingRenderer::freeAccumBuffer()
{
    if (_accumBufferDev)
    {
        cudaFree(_accumBufferDev);
        _accumBufferDev = nullptr;
    }
}
void CudaPathTracingRenderer::resetAccumulation()
{
    if (_accumBufferDev)
    {
        cudaMemset(_accumBufferDev, 0, static_cast<size_t>(_width) * static_cast<size_t>(_height) * sizeof(glm::vec3));
    }
    _accumFrameCount = 0;
    _frame = 0;
}

bool CudaPathTracingRenderer::cameraChanged(const Camera &camera) const
{
    if (!_hasPrevCamState)
        return true;
    float posDiff = glm::length(camera.getPosition() - _prevCamPos);
    float frontDiff = glm::length(camera.getFront() - _prevCamFront);
    float upDiff = glm::length(camera.getUp() - _prevCamUp);
    float fovDiff = fabsf(camera.getFov() - _prevFov);
    const float eps = 1e-4f;
    return posDiff > eps || frontDiff > eps || upDiff > eps || fovDiff > 1e-4f;
}
void CudaPathTracingRenderer::updateCameraState(const Camera &camera)
{
    _prevCamPos = camera.getPosition();
    _prevCamFront = camera.getFront();
    _prevCamUp = camera.getUp();
    _prevFov = camera.getFov();
    _hasPrevCamState = true;
}

void CudaPathTracingRenderer::renderFrame(Camera &camera)
{
    if (cameraChanged(camera))
        resetAccumulation();
    updateCameraState(camera);

    // 检查场景是否有变动，如有则重新上传
    if (scene_ && scene_->isDirty()) {
        // 上传基础形状（球体、平面等）
        uploadScene();
        // 上传BVH模型数据
        uploadBVH();
        scene_->clearDirty();
        resetAccumulation(); // 场景变化，需要重置累积
    }

    unsigned int *devPtr = nullptr;
    size_t size = 0;
    if (_gpu->mapWrite(reinterpret_cast<void **>(&devPtr), &size))
    {
        int kSamplesPerPixel = _spp;
        int kMaxDepth = _maxDepth;
        int nextAccum = _accumFrameCount + 1;
        launchPathTracer(devPtr, _width, _height, camera, _shapesDev, _shapeCount, _modelsDev, _modelCount, kSamplesPerPixel, kMaxDepth, _accumBufferDev, nextAccum, _frame);
        _gpu->unmapWrite();
        _gpu->finalizeWrite();
        _accumFrameCount++;
    }
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
    _frame++;
}

void CudaPathTracingRenderer::destroy()
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
    freeAccumBuffer();
    if (_shapesDev)
    {
        cudaFree(_shapesDev);
        _shapesDev = nullptr;
    }
    if (_modelsDev) {
        cudaFree(_modelsDev);
        _modelsDev = nullptr;
    }
    if (scene_) {
        scene_->freeBVHGPU();
    }
}
