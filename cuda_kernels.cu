#include <cuda_runtime.h>
//#include <cuda_gl_interop.h>
#include <math.h>
#include <stdio.h>
#include <device_launch_parameters.h>

// Kernel to animate sphere positions
extern "C" __global__ void animateSpheresKernel(float* data, int n, float time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Animate each sphere in a circle (example)
        float x = data[3 * idx + 0];
        float y = data[3 * idx + 1];
        float z = data[3 * idx + 2];
        data[3 * idx + 0] = x + 0.1f * cosf(time + idx);
        data[3 * idx + 1] = y + 0.1f * sinf(time + idx);
        data[3 * idx + 2] = z;
    }
}

// Host launcher for main.cpp
extern "C" void launchAnimateSpheresKernel(float* d_data, int n, float time) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    animateSpheresKernel<<<numBlocks, blockSize>>>(d_data, n, time);
    cudaDeviceSynchronize();
}
