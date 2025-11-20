#pragma once
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <array>

class GPUResources {
public:
    GPUResources();
    ~GPUResources();
    bool createTriplePBO(int width, int height);
    void destroyPBO();

    bool mapWrite(void** devPtr, size_t* size);
    void unmapWrite();
    void finalizeWrite();
    GLuint getDisplayPBO() const;
    GLuint getDisplayTexture() const { return tex_; }

    GLuint getWritePBO() const { return pbo_[writeIndex_]; }
    GLuint getTexture() const { return tex_; }

    int width() const { return width_; }
    int height() const { return height_; }

private:
    static constexpr int kBufferCount = 3;
    int width_ = 0, height_ = 0;
    std::array<GLuint, kBufferCount> pbo_{ {0,0,0} };
    std::array<cudaGraphicsResource*, kBufferCount> cudaRes_{ {nullptr,nullptr,nullptr} };
    GLuint tex_ = 0;

    int writeIndex_ = 0;
    int displayIndex_ = -1;
};
