#ifdef __cplusplus
extern "C" void launchAnimateSpheresKernel(float* d_data, int n, float time);
#endif
#include<glad/glad.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include<iostream>
#include<GLFW/glfw3.h>
#include<stb/stb_image.h>
#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>
#include<cstdlib>
#include<ctime>
#include<vector>
#include<math.h>
#include <chrono>
#include <thread>
#include <array>

#include"Texture.h"
#include"shaderClass.h"
#include"VAO.h"
#include"VBO.h"
#include"EBO.h"
#include"Camera.h"
#include "FrustumCulling.h"

using namespace std;

// --- Spatial Grid Structures ---
struct GridCell {
	std::vector<int> sphereIndices;
};

const int GRID_SIZE = 16; // Number of cells per axis
const float GRID_WORLD_MIN = -150.0f;
const float GRID_WORLD_MAX = 150.0f;
const float GRID_CELL_SIZE = (GRID_WORLD_MAX - GRID_WORLD_MIN) / GRID_SIZE;
std::vector<GridCell> gridCells(GRID_SIZE* GRID_SIZE* GRID_SIZE);

int getGridIndex(int x, int y, int z) {
	return x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
}

void assignSpheresToGrid(const std::vector<glm::vec3>& spherePositions) {
	for (int i = 0; i < spherePositions.size(); ++i) {
		glm::vec3 p = spherePositions[i];
		int gx = std::clamp(int((p.x - GRID_WORLD_MIN) / GRID_CELL_SIZE), 0, GRID_SIZE - 1);
		int gy = std::clamp(int((p.y - GRID_WORLD_MIN) / GRID_CELL_SIZE), 0, GRID_SIZE - 1);
		int gz = std::clamp(int((p.z - GRID_WORLD_MIN) / GRID_CELL_SIZE), 0, GRID_SIZE - 1);
		gridCells[getGridIndex(gx, gy, gz)].sphereIndices.push_back(i);
	}
}

const unsigned int width = 1920;
const unsigned int height = 1080;
const float M_PI = 3.14159265359f;

GLfloat lightVertices[] =
{ //     COORDINATES     //
	-0.1f, -0.1f,  0.1f,
	-0.1f, -0.1f, -0.1f,
	 0.1f, -0.1f, -0.1f,
	 0.1f, -0.1f,  0.1f,
	-0.1f,  0.1f,  0.1f,
	-0.1f,  0.1f, -0.1f,
	 0.1f,  0.1f, -0.1f,
	 0.1f,  0.1f,  0.1f
};

GLuint lightIndices[] =
{
	0, 1, 2,
	0, 2, 3,
	0, 4, 7,
	0, 7, 3,
	3, 7, 6,
	3, 6, 2,
	2, 6, 5,
	2, 5, 1,
	1, 5, 4,
	1, 4, 0,
	4, 5, 6,
	4, 6, 7
};


int main()
{
	cudaGraphicsResource* cudaVBO;

	srand(static_cast<unsigned int>(time(0)));
	// Initialize GLFW
	glfwInit();

	// Tell GLFW what version of OpenGL we are using 
	// In this case we are using OpenGL 3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	// Tell GLFW we are using the CORE profile
	// So that means we only have the modern functions
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create a GLFWwindow object of 800 by 800 pixels, naming it "YoutubeOpenGL"
	GLFWwindow* window = glfwCreateWindow(width, height, "YoutubeOpenGL", NULL, NULL);
	// Error check if the window fails to create
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	// Introduce the window into the current context
	glfwMakeContextCurrent(window);

	//Load GLAD so it configures OpenGL
	gladLoadGL();
	// Specify the viewport of OpenGL in the Window
	// In this case the viewport goes from x = 0, y = 0, to x = 800, y = 800
	glViewport(0, 0, width, height);


	// Generates Shader object using shaders default.vert and default.frag
	Shader shaderProgram("default.vert", "default.frag");
								
	// --- Sphere mesh generation: only generate one sphere mesh ---
	// This creates the geometry for a single sphere, which will be reused for all 1000 spheres
	// The Vertex struct is defined above
	struct Vertex {
		glm::vec3 position;
		glm::vec3 color;
		glm::vec2 texCoord;
		glm::vec3 normal;
	};
	struct MeshData {
		vector<Vertex> vertices;
		vector<unsigned int> indices;
		VAO vao;
		VBO vbo;
		EBO ebo;
		// Use default constructor and defaulted move/copy semantics (remove explicit move/copy/delete)
	};
	std::vector<Vertex> vertices;
	std::vector<unsigned int> indices;
	vector<MeshData> lodMeshes(3); // LOD meshes for different levels of detail

	int lodSectors[3] = {36, 15, 7};
	int lodStacks[3] = {36, 15, 7};

	// --- Generate random positions for spheres (instance data) ---
	std::vector<glm::vec3> spherePositions;
	int numSpheres = 1000000; // You can adjust this for performance
	for (int i = 0; i < numSpheres; ++i) {
		float x = static_cast<float>((rand() % 300 - 150));
		float y = static_cast<float>((rand() % 300 - 150));
		float z = static_cast<float>((rand() % 300 - 150));
		spherePositions.push_back(glm::vec3(x, y, z));
	}
	// --- Assign spheres to grid cells (spatial partitioning) ---
	assignSpheresToGrid(spherePositions);

	// --- Set up instance VBO for sphere positions (vec3) BEFORE LOD mesh setup ---
	GLuint instanceVBO;
	glGenBuffers(1, &instanceVBO);
	glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
	glBufferData(GL_ARRAY_BUFFER, numSpheres * sizeof(glm::vec3), spherePositions.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	for (int lod = 0; lod < 3; ++lod) {
		int sectorCount = lodSectors[lod];
		int stackCount = lodStacks[lod];
		float radius = 1.0f;
		float x, y, z, xy;
		float nx, ny, nz, lengthInv = 1.0f / radius;
		float s, t;
		float sectorStep = 2 * M_PI / sectorCount;
		float stackStep = M_PI / stackCount;
		float sectorAngle, stackAngle;
		for (int i = 0; i <= stackCount; ++i) {
			stackAngle = M_PI / 2 - i * stackStep;
			xy = radius * cosf(stackAngle);
			z = radius * sinf(stackAngle);
			for (int j = 0; j <= sectorCount; ++j) {
				sectorAngle = (j * sectorStep) / 2;
				x = xy * cosf(sectorAngle);
				y = xy * sinf(sectorAngle);
				nx = x * lengthInv;
				ny = y * lengthInv;
				nz = z * lengthInv;
				s = (float)j / sectorCount;
				t = (float)i / stackCount;
				glm::mat4 rot90 = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0, 0, 1));
				glm::vec4 rotatedPos = rot90 * glm::vec4(x, y, z, 1.0f);
				glm::vec4 rotatedNormal = rot90 * glm::vec4(nx, ny, nz, 0.0f);
				lodMeshes[lod].vertices.push_back(Vertex{
					glm::vec3(rotatedPos),
					glm::vec3(1.0f, 1.0f, 1.0f),
					glm::vec2(s, t),
					glm::vec3(rotatedNormal)
					});
			}
		}
		for (int i = 0; i < stackCount; ++i) {
			int k1 = i * (sectorCount + 1);
			int k2 = k1 + sectorCount + 1;
			for (int j = 0; j < sectorCount; ++j, ++k1, ++k2) {
				if (i != 0) {
					lodMeshes[lod].indices.push_back(k1);
					lodMeshes[lod].indices.push_back(k2);
					lodMeshes[lod].indices.push_back(k1 + 1);
				}
				if (i != (stackCount - 1)) {
					lodMeshes[lod].indices.push_back(k1 + 1);
					lodMeshes[lod].indices.push_back(k2);
					lodMeshes[lod].indices.push_back(k2 + 1);
				}
			}
		}
		lodMeshes[lod].vao.Bind();
		lodMeshes[lod].vbo = VBO(lodMeshes[lod].vertices.data(), lodMeshes[lod].vertices.size() * sizeof(Vertex));
		lodMeshes[lod].ebo = EBO(lodMeshes[lod].indices.data(), lodMeshes[lod].indices.size() * sizeof(unsigned int));
		lodMeshes[lod].vbo.Bind();
		lodMeshes[lod].ebo.Bind();

		GLsizei stride = sizeof(Vertex);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
		glEnableVertexAttribArray(0);

		// aColor (location = 1)
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);

		// aTex (location = 2)
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));
		glEnableVertexAttribArray(2);

		// aNormal (location = 3)
		glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, (void*)(8 * sizeof(float)));
		glEnableVertexAttribArray(3);

		// --- Set up per-instance attribute (location 4) for instance positions for each LOD VAO ---
		glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
		glEnableVertexAttribArray(4);
		glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
		glVertexAttribDivisor(4, 1);

		lodMeshes[lod].vao.Unbind();
		lodMeshes[lod].vbo.Unbind();
		lodMeshes[lod].ebo.Unbind();
	}

	// Register instanceVBO with CUDA (once, after OpenGL context is created)
	cudaGraphicsGLRegisterBuffer(&cudaVBO, instanceVBO, cudaGraphicsMapFlagsWriteDiscard);
	

	// Shader for light cube
	Shader lightShader("light.vert", "light.frag");
	// Generates Vertex Array Object and binds it
	VAO lightVAO;
	lightVAO.Bind();
	// Generates Vertex Buffer Object and links it to vertices
	VBO lightVBO(lightVertices, sizeof(lightVertices));
	// Generates Element Buffer Object and links it to indices
	EBO lightEBO(lightIndices, sizeof(lightIndices));
	// Links VBO attributes such as coordinates and colors to VAO
	lightVAO.LinkAttrib(lightVBO, 0, 3, GL_FLOAT, 3 * sizeof(float), (void*)0);
	// Unbind all to prevent accidentally modifying them
	lightVAO.Unbind();
	lightVBO.Unbind();
	lightEBO.Unbind();



	glm::vec4 lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
	// Move the light to the center
	glm::vec3 lightPos = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::mat4 lightModel = glm::mat4(1.0f);
	lightModel = glm::translate(lightModel, lightPos);


	/*Texture brickTex("brick.png", GL_TEXTURE_2D, GL_TEXTURE0, GL_RGBA, GL_UNSIGNED_BYTE);
	brickTex.texUnit(shaderProgram, "tex0", 0);*/

	// Original code from the tutorial
	Texture brickTex("obama_512.png", GL_TEXTURE_2D, GL_TEXTURE0, GL_RGBA, GL_UNSIGNED_BYTE);
	brickTex.Bind();
	brickTex.texUnit(shaderProgram, "tex0", 0);
	brickTex.Unbind();



	// Enables the Depth Buffer
	glEnable(GL_DEPTH_TEST);

	// Creates camera object
	Camera camera(width, height, glm::vec3(0.0f, 0.0f, 2.0f));


	double prevTime = 0.0;
	double crntTime = 0.0;
	double timeDiff = 0.0;
	unsigned int counter = 0;
	// Main while loop
	while (!glfwWindowShouldClose(window))
	{
		crntTime = glfwGetTime();
		timeDiff = crntTime - prevTime;
		counter++;
		if (timeDiff >= 1.0 / 30.0)
		{
			// Create new title
			std::string FPS = std::to_string((1.0 / timeDiff) * counter);
			std::string ms = std::to_string((timeDiff / counter) * 1000);
			std::string newTitle = "1 Million spheres - " + FPS + "FPS / " + ms + "ms";
			glfwSetWindowTitle(window, newTitle.c_str());

			// Resets times and counter
			prevTime = crntTime;
			counter = 0;
		}
		// Specify the color of the background
		glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
		// Clean the back buffer and depth buffer
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Handles camera inputs
		camera.Inputs(window);
		// Updates and exports the camera matrix to the Vertex Shader
		camera.updateMatrix(45.0f, 1.0f, 100.0f);

		// --- CUDA: Map instanceVBO and update instance data on GPU ---
		cudaGraphicsMapResources(1, &cudaVBO, 0);
		size_t num_bytes;
		glm::vec3* d_instanceData;
		cudaGraphicsResourceGetMappedPointer((void**)&d_instanceData, &num_bytes, cudaVBO);
		// Animate sphere positions on GPU using CUDA kernel
		launchAnimateSpheresKernel((float*)d_instanceData, numSpheres, (float)crntTime);
		cudaGraphicsUnmapResources(1, &cudaVBO, 0);

		// --- Frustum culling: build visible sphere list for each LOD (CPU fallback, still needed for now) ---
		glm::mat4 view = camera.getViewMatrix();
		glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)width / height, 1.0f, 100.0f);
		std::array<FrustumPlane, 6> frustum;
		ExtractFrustumPlanes(proj * view, frustum);
		std::vector<glm::vec3> lodVisiblePositions[3];
		float lodRadii[3] = {0.5f, 0.25f, 0.125f}; // Use correct radii if LODs differ in size
		for (int lod = 0; lod < 3; ++lod) lodVisiblePositions[lod].reserve(numSpheres);
		// --- Spatial grid culling ---
		for (int gx = 0; gx < GRID_SIZE; ++gx) {
			for (int gy = 0; gy < GRID_SIZE; ++gy) {
				for (int gz = 0; gz < GRID_SIZE; ++gz) {
					glm::vec3 cellCenter = glm::vec3(
						GRID_WORLD_MIN + (gx + 0.5f) * GRID_CELL_SIZE,
						GRID_WORLD_MIN + (gy + 0.5f) * GRID_CELL_SIZE,
						GRID_WORLD_MIN + (gz + 0.5f) * GRID_CELL_SIZE
					);
					float cellRadius = sqrtf(3.0f) * GRID_CELL_SIZE * 0.5f;
					if (SphereInFrustum(frustum, cellCenter, cellRadius)) {
						const auto& indices = gridCells[getGridIndex(gx, gy, gz)].sphereIndices;
						for (int idx : indices) {
							const glm::vec3& pos = spherePositions[idx];
							float dist = glm::length(camera.Position - pos);
							int lod = 2;
							if (dist < 50.0f) lod = 0;
							else if (dist < 100.0f) lod = 1;
							if (SphereInFrustum(frustum, pos, lodRadii[lod])) {
								lodVisiblePositions[lod].push_back(pos);
							}
						}
					}
				}
			}
		}
		// --- Debug: Print number of visible spheres per LOD ---
		static int frameCount = 0;
		/*if (frameCount++ % 60 == 0) {
			printf("Visible LOD0: %zu, LOD1: %zu, LOD2: %zu\n", lodVisiblePositions[0].size(), lodVisiblePositions[1].size(), lodVisiblePositions[2].size());
		}*/
		// --- Render all LODs efficiently ---
		shaderProgram.Activate();
		camera.Matrix(shaderProgram, "camMatrix");
		glUniform1f(glGetUniformLocation(shaderProgram.ID, "uTime"), 0.0f);
		glUniform3f(glGetUniformLocation(shaderProgram.ID, "lightPos"), 0.0f, 0.0f, 0.0f);
		glUniform3f(glGetUniformLocation(shaderProgram.ID, "camPos"), camera.Position.x, camera.Position.y, camera.Position.z);
		glUniform4f(glGetUniformLocation(shaderProgram.ID, "lightColor"), lightColor.x, lightColor.y, lightColor.z, lightColor.w);
		brickTex.Bind();
		for (int lod = 0; lod <= 2; lod++) {
			int numVisible = static_cast<int>(lodVisiblePositions[lod].size());
			if (numVisible > 0) {
				glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
				glBufferData(GL_ARRAY_BUFFER, numVisible * sizeof(glm::vec3), lodVisiblePositions[lod].data(), GL_DYNAMIC_DRAW);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				lodMeshes[lod].vao.Bind();
				glDrawElementsInstanced(
					GL_TRIANGLES,
					static_cast<GLsizei>(lodMeshes[lod].indices.size()),
					GL_UNSIGNED_INT,
					0,
					numVisible
				);
				lodMeshes[lod].vao.Unbind();
			}
		}
		brickTex.Unbind();
		// --- Light rendering (unchanged) ---
		lightShader.Activate();
		camera.Matrix(lightShader, "camMatrix");
		lightVAO.Bind();
		glDrawElements(GL_TRIANGLES, sizeof(lightIndices) / sizeof(int), GL_UNSIGNED_INT, 0);
		lightVAO.Unbind();

		// --- Animate the light source (cube) in a circle ---
		// float lightAngle = static_cast<float>(glfwGetTime()) * 0.5f; // Light rotation speed
		// float lightRadius = 2.0f; // Distance from center
		// lightPos = glm::vec3(
		//     cos(lightAngle) * lightRadius,
		//     0.0f,
		//     sin(lightAngle) * lightRadius
		// );
		lightPos = glm::vec3(0.0f, 0.0f, 0.0f); // Fixed position
		lightModel = glm::mat4(1.0f);
		lightModel = glm::translate(lightModel, lightPos);
		lightShader.Activate();
		glUniformMatrix4fv(glGetUniformLocation(lightShader.ID, "model"), 1, GL_FALSE, glm::value_ptr(lightModel));
		glUniform4f(glGetUniformLocation(lightShader.ID, "lightColor"), lightColor.x, lightColor.y, lightColor.z, lightColor.w);
		shaderProgram.Activate();
		glUniform4f(glGetUniformLocation(shaderProgram.ID, "lightColor"), lightColor.x, lightColor.y, lightColor.z, lightColor.w);
		glUniform3f(glGetUniformLocation(shaderProgram.ID, "lightPos"), lightPos.x, lightPos.y, lightPos.z);

		// Swap the back buffer with the front buffer
		glfwSwapBuffers(window);
		// Take care of all GLFW events
		glfwPollEvents();
	}

	shaderProgram.Delete();
	lightVAO.Delete();
	lightVBO.Delete();
	lightEBO.Delete();
	lightShader.Delete();
	// Delete window before ending the program
	glfwDestroyWindow(window);
	// Terminate GLFW before ending the program
	glfwTerminate();
	return 0;
}

