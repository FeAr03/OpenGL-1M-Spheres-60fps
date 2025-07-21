#ifndef VBO_H
#define VBO_H

#include <glad/glad.h>

class VBO
{
public:
    GLuint ID;
    VBO(void* vertices, GLsizeiptr size);
    VBO() : ID(0) {}
    void Bind();
    void Unbind();
    void Delete();
};

#endif