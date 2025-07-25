
# OpenGL 1M Spheres (CUDA-Accelerated)

## Overview
Welcome! This project is a fun and fast demo of real-time rendering and animation of up to 2 million spheres in C++ using OpenGL, with all the heavy lifting done on the GPU via CUDA. It's a playground for:

- Instanced rendering (so you can draw a ton of spheres at once)
- CUDA/OpenGL interop for real-time GPU-side animation
- Fast culling using a 3D grid and frustum planes
- Level of Detail (LOD) so you get more FPS
- Modern OpenGL (3.3+) and GLSL shaders

## Features
- Renders and animates up to 2 million spheres at interactive frame rates
- All sphere movement is handled on the GPU with CUDA
- Fast culling and LOD for better performance
- Real-time FPS and frame time in the window title




## How to Build & Run

### What you need
- Windows (tested), Linux should work too
- CMake (or just build manually)
- CUDA Toolkit (11.x or newer)
- OpenGL 3.3+
- [GLFW](https://www.glfw.org/), [GLAD](https://glad.dav1d.de/), [GLM](https://github.com/g-truc/glm), [stb_image.h](https://github.com/nothings/stb)


### Quick Start (VS Code example)
1. Clone this repo:
   ```sh
   git clone https://github.com/FeAr03/OpenGL-1M-Spheres-60fps.git
   cd OpenGL-1M-Spheres-60fps
   ```
2. Open in VS Code. Make sure CUDA and all dependencies are installed and in your include/library path.
3. Build using the provided tasks or however you like.
4. Run the executable. You should see a window full of animated spheres and a live FPS counter.

## What's in here?
- `main.cpp` - Main app logic
- `shaderClass.h/cpp` - Shader management
- `Camera.h/cpp` - Camera controls
- `FrustumCulling.h` - Frustum culling
- `Texture.h/cpp` - Texture loading
- `VAO/VBO/EBO.h/cpp` - OpenGL buffer management
- `cuda_kernels.cu` - CUDA animation kernel
- `default.vert`, `default.frag` - GLSL shaders

## Credits
- [GLFW](https://www.glfw.org/)
- [GLAD](https://glad.dav1d.de/)
- [GLM](https://github.com/g-truc/glm)
- [stb_image.h](https://github.com/nothings/stb)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

## License
MIT License. See [LICENSE](LICENSE) for details.

---

**Tip:** For best performance, use a modern NVIDIA GPU. You can tweak the number of spheres in `main.cpp` to match your hardware.
