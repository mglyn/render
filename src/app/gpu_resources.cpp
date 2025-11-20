#include <iostream>
#include <glad/glad.h>

#include "gpu_resources.h"

GPUResources::GPUResources() {}
GPUResources::~GPUResources(){ destroyPBO(); }

bool GPUResources::createTriplePBO(int width, int height){
    if(width <=0 || height <=0) return false;
    width_ = width; height_ = height;
    for(int i=0;i<kBufferCount;++i){
        glGenBuffers(1, &pbo_[i]);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_[i]);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width_*height_*4, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaRes_[i], pbo_[i], cudaGraphicsRegisterFlagsWriteDiscard);
        if(err != cudaSuccess){
            std::cerr << "cudaGraphicsGLRegisterBuffer failed on index " << i << ": " << cudaGetErrorString(err) << std::endl;
            destroyPBO();
            return false;
        }
    }
    glGenTextures(1, &tex_);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    writeIndex_ = 0;
    displayIndex_ = -1;
    return true;
}

void GPUResources::destroyPBO(){
    for(int i=0;i<kBufferCount;++i){
        if(cudaRes_[i]){ cudaGraphicsUnregisterResource(cudaRes_[i]); cudaRes_[i] = nullptr; }
        if(pbo_[i]){ glDeleteBuffers(1, &pbo_[i]); pbo_[i] = 0; }
    }
    if(tex_){ glDeleteTextures(1, &tex_); tex_ = 0; }
}

bool GPUResources::mapWrite(void** devPtr, size_t* size){
    cudaGraphicsResource* res = cudaRes_[writeIndex_];
    if(!res) return false;
    cudaError_t err = cudaGraphicsMapResources(1, &res, 0);
    if(err != cudaSuccess){
        std::cerr << "cudaGraphicsMapResources failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    err = cudaGraphicsResourceGetMappedPointer(devPtr, size, res);
    if(err != cudaSuccess){
        std::cerr << "cudaGraphicsResourceGetMappedPointer failed: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnmapResources(1, &res, 0);
        return false;
    }
    return true;
}

void GPUResources::unmapWrite(){
    cudaGraphicsResource* res = cudaRes_[writeIndex_];
    if(res){ cudaGraphicsUnmapResources(1, &res, 0); }
}

void GPUResources::finalizeWrite(){
    displayIndex_ = writeIndex_;
    writeIndex_ = (writeIndex_ + 1) % kBufferCount;
    if(writeIndex_ == displayIndex_) {
        writeIndex_ = (writeIndex_ + 1) % kBufferCount;
    }
}

GLuint GPUResources::getDisplayPBO() const {
    if(displayIndex_ < 0) return pbo_[writeIndex_];
    return pbo_[displayIndex_];
}
