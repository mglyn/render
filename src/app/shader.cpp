#include "app/shader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <glm/glm.hpp>

Shader::Shader() : _programID(0) {}

Shader::~Shader() {
    if (_programID != 0) {
        glDeleteProgram(_programID);
    }
}

bool Shader::load(const std::string& vertexPath, const std::string& fragmentPath) {
    // 1. 从文件路径中获取顶点/片段着色器源代码
    std::string vertexCode;
    std::string fragmentCode;
    std::ifstream vShaderFile;
    std::ifstream fShaderFile;
    // 保证ifstream对象可以抛出异常：
    vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        // 打开文件
        vShaderFile.open(vertexPath);
        fShaderFile.open(fragmentPath);
        std::stringstream vShaderStream, fShaderStream;
        // 读取文件的缓冲内容到数据流中
        vShaderStream << vShaderFile.rdbuf();
        fShaderStream << fShaderFile.rdbuf();
        // 关闭文件处理器
        vShaderFile.close();
        fShaderFile.close();
        // 转换数据流到string
        vertexCode = vShaderStream.str();
        fragmentCode = fShaderStream.str();
    }
    catch (std::ifstream::failure& e) {
        std::cerr << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
        return false;
    }
    const char* vShaderCode = vertexCode.c_str();
    const char* fShaderCode = fragmentCode.c_str();

    // 2. 编译着色器
    GLuint vertex, fragment;
    // 顶点着色器
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    if (!checkCompileErrors(vertex, "VERTEX")) {
        return false;
    }
    // 片段着色器
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    if (!checkCompileErrors(fragment, "FRAGMENT")) {
        glDeleteShader(vertex);
        return false;
    }

    // 3. 链接着色器程序
    _programID = glCreateProgram();
    glAttachShader(_programID, vertex);
    glAttachShader(_programID, fragment);
    glLinkProgram(_programID);
    if (!checkCompileErrors(_programID, "PROGRAM")) {
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        return false;
    }

    // 删除着色器，因为它们已经链接到我们的程序中了，不再需要
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    return true;
}

void Shader::use() const {
    glUseProgram(_programID);
}

void Shader::setVec2(const std::string& name, const glm::vec2& value) {
    glUniform2f(glGetUniformLocation(_programID, name.c_str()), value.x, value.y);
}

bool Shader::checkCompileErrors(GLuint shader, const std::string& type) {
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            return false;
        }
    }
    else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            return false;
        }
    }
    return true;
}
