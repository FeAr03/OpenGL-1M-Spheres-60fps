// FrustumCullingCUDA_Scan.cu
// CUDA frustum culling and LOD selection using flag array and prefix sum (scan) for compaction (no atomics)
// This file is meant to be compiled with nvcc and linked with your main project.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <math_constants.h>

// Kernel: mark visible spheres and LOD in flag arrays
__global__ void markVisibleKernel(
    const float* spherePos, int N,
    const float* frustumPlanes, // 6*4 floats
    const float* camPos, const float* lodRadii,
    int* flag0, int* flag1, int* flag2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float3 pos = make_float3(spherePos[3*i+0], spherePos[3*i+1], spherePos[3*i+2]);
    // Frustum test (conservative: use largest LOD radius)
    bool visible = true;
    for (int p = 0; p < 6; ++p) {
        const float4 plane = make_float4(
            frustumPlanes[p*4+0],
            frustumPlanes[p*4+1],
            frustumPlanes[p*4+2],
            frustumPlanes[p*4+3]
        );
        float d = pos.x * plane.x + pos.y * plane.y + pos.z * plane.z + plane.w;
        if (d < -lodRadii[2]) visible = false;
    }
    int lod = 2;
    if (visible) {
        float dx = pos.x - camPos[0];
        float dy = pos.y - camPos[1];
        float dz = pos.z - camPos[2];
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        if (dist < 50.0f) lod = 0;
        else if (dist < 100.0f) lod = 1;
    }
    flag0[i] = (visible && lod == 0) ? 1 : 0;
    flag1[i] = (visible && lod == 1) ? 1 : 0;
    flag2[i] = (visible && lod == 2) ? 1 : 0;
}

// Kernel: compact visible positions using scan results
__global__ void compactKernel(
    const float* spherePos, int N,
    const int* flag, const int* scan, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (flag[i]) {
        int idx = scan[i];
        out[3*idx+0] = spherePos[3*i+0];
        out[3*idx+1] = spherePos[3*i+1];
        out[3*idx+2] = spherePos[3*i+2];
    }
}

extern "C" void launchFrustumCullingScan(
    const float* d_spherePositions, int numSpheres,
    const float* d_frustumPlanes, const float* d_cameraPos, const float* d_lodRadii,
    float* d_lod0Out, int* lod0Count,
    float* d_lod1Out, int* lod1Count,
    float* d_lod2Out, int* lod2Count,
    cudaStream_t stream = 0)
{
    int blockSize = 256;
    int gridSize = (numSpheres + blockSize - 1) / blockSize;
    // Allocate flag arrays
    thrust::device_vector<int> flag0(numSpheres), flag1(numSpheres), flag2(numSpheres);
    // Mark visible and LOD
    markVisibleKernel<<<gridSize, blockSize, 0, stream>>>(
        d_spherePositions, numSpheres,
        d_frustumPlanes, d_cameraPos, d_lodRadii,
        thrust::raw_pointer_cast(flag0.data()),
        thrust::raw_pointer_cast(flag1.data()),
        thrust::raw_pointer_cast(flag2.data()));
    // Allocate scan arrays
    thrust::device_vector<int> scan0(numSpheres), scan1(numSpheres), scan2(numSpheres);
    // Exclusive scan
    thrust::exclusive_scan(flag0.begin(), flag0.end(), scan0.begin());
    thrust::exclusive_scan(flag1.begin(), flag1.end(), scan1.begin());
    thrust::exclusive_scan(flag2.begin(), flag2.end(), scan2.begin());
    // Get output counts
    *lod0Count = flag0[numSpheres-1] + scan0[numSpheres-1];
    *lod1Count = flag1[numSpheres-1] + scan1[numSpheres-1];
    *lod2Count = flag2[numSpheres-1] + scan2[numSpheres-1];
    // Compact output
    compactKernel<<<gridSize, blockSize, 0, stream>>>(
        d_spherePositions, numSpheres,
        thrust::raw_pointer_cast(flag0.data()), thrust::raw_pointer_cast(scan0.data()), d_lod0Out);
    compactKernel<<<gridSize, blockSize, 0, stream>>>(
        d_spherePositions, numSpheres,
        thrust::raw_pointer_cast(flag1.data()), thrust::raw_pointer_cast(scan1.data()), d_lod1Out);
    compactKernel<<<gridSize, blockSize, 0, stream>>>(
        d_spherePositions, numSpheres,
        thrust::raw_pointer_cast(flag2.data()), thrust::raw_pointer_cast(scan2.data()), d_lod2Out);
}
