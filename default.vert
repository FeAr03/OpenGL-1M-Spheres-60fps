#version 330 core

// Positions/Coordinates
layout (location = 0) in vec3 aPos;
// Colors
layout (location = 1) in vec3 aColor;
// Texture Coordinates
layout (location = 2) in vec2 aTex;
// Normals (not necessarily normalized)
layout (location = 3) in vec3 aNormal;
// Instanced sphere position (vec3)
layout (location = 4) in vec3 instanceOffset; // initial position

// Outputs the color for the Fragment Shader
out vec3 color;
out vec2 texCoord;
out vec3 Normal;
out vec3 crntPos;

// Imports the camera matrix from the main function
uniform mat4 camMatrix;
uniform float uTime;
uniform vec3 lightPos;
// Imports the camera position from the main function
uniform vec3 camPos;

void main()
{
    // Animate revolution around the light in the XZ plane
    float angle = uTime * 0.5;
    mat4 rotation = mat4(1.0);
    rotation[0][0] =  cos(angle);
    rotation[0][2] =  sin(angle);
    rotation[2][0] = -sin(angle);
    rotation[2][2] =  cos(angle);
    vec3 relPos = instanceOffset - lightPos;
    vec3 newPos = lightPos + (rotation * vec4(relPos, 1.0)).xyz;

    // --- BILLBOARDING: Make hemisphere's curved side always face the camera, with 90 degree rotation ---
    vec3 toCamera = normalize(camPos - newPos);// Normalizing the vector to the camera position
    float eps = 0.001;
    vec3 up = abs(toCamera.y) > 1.0 - eps ? vec3(20.5,22,-50.5) : vec3(20.5,22,-50.5);
    vec3 right = normalize(cross(up , toCamera));
    vec3 billboardUp = cross(toCamera, right);
    // Build rotation matrix to align +X (not +Z) with toCamera for a 90 degree rotation
    mat3 billboardMat = mat3(toCamera, billboardUp, right); // Rotates by 90 degrees compared to Z alignment
    vec3 localPos = aPos;
    vec3 rotated = billboardMat * localPos;
    vec3 finalPos = newPos + rotated * 0.2;

    crntPos = finalPos;
    gl_Position = camMatrix * vec4(finalPos, 1.0);
    color = aColor;
    texCoord = aTex;
    Normal = billboardMat * aNormal;
}