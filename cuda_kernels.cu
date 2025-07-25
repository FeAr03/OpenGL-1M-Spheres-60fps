#include <cuda_runtime.h>
//#include <cuda_gl_interop.h>
#include <math.h>
#include <stdio.h>
#include <device_launch_parameters.h>

// Kernel to animate sphere positions
extern "C" __global__ void animateSpheresKernel(float* data, int n, float time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Animate each sphere in a circle (make movement obvious)
        float x = data[3 * idx + 0];
        float y = data[3 * idx + 1];
        float z = data[3 * idx + 2];
        data[3 * idx + 0] = x + 0.10f * cosf(time + idx); // Increased offset
        data[3 * idx + 1] = y + 0.10f * sinf(time + idx);
        data[3 * idx + 2] = z;
    }
}

// Host launcher for main.cpp
extern "C" void launchAnimateSpheresKernel(float* d_data, int n, float time) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    animateSpheresKernel<<<numBlocks, blockSize>>>(d_data, n, time);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
	//printf("Inside launchAnimateSpheresKernel, after synchronize\n");
    if (err != cudaSuccess) {
        printf("CUDA post-sync error: %s\n", cudaGetErrorString(err));
    }
}
