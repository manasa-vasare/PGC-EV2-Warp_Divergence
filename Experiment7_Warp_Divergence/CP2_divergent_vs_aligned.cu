
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>

#define N (1 << 24)
#define BLOCK_SIZE 256
#define ITERATIONS 10

#define CUDA_CHECK(call)                                             \
    do {                                                             \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            std::cerr << "CUDA Error: "                               \
                      << cudaGetErrorString(err)                       \
                      << "\nLine: " << __LINE__ << std::endl;          \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Long path: 64 iterations
__device__ __forceinline__
float long_work(float x)
{
    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        x = x * 1.000001f + 0.000001f;
    }
    return x;
}

// Short path: 8 iterations
__device__ __forceinline__
float short_work(float x)
{
    #pragma unroll 1
    for (int i = 0; i < 8; i++)
    {
        x = x * 0.999999f - 0.000001f;
    }
    return x;
}

// ------------------------------------------------------------
// KERNEL 1: DIVERGENT BRANCH
// Exact condition required by the assignment:
// threadIdx.x % 2
// ------------------------------------------------------------
__global__
void divergent_kernel(float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    float x = (float)(idx % 1000) + 1.0f;

    if (threadIdx.x % 2 == 0)
    {
        output[idx] = long_work(x);
    }
    else
    {
        output[idx] = short_work(x);
    }
}

// ------------------------------------------------------------
// KERNEL 2: WARP-ALIGNED BRANCH
// Exact condition required by the assignment:
// (threadIdx.x / 32) % 2
// ------------------------------------------------------------
__global__
void aligned_kernel(float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    float x = (float)(idx % 1000) + 1.0f;

    if ((threadIdx.x / 32) % 2 == 0)
    {
        output[idx] = long_work(x);
    }
    else
    {
        output[idx] = short_work(x);
    }
}

// ------------------------------------------------------------
// BENCHMARK
// ------------------------------------------------------------
template <typename Kernel>
float benchmark(Kernel kernel)
{
    // Warm-up
    for (int i = 0; i < 3; i++)
        kernel();

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < ITERATIONS; i++)
        kernel();

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / ITERATIONS;
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int main()
{
    float *d_output = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_output, N * sizeof(float)));

    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::cout << "========================================\n";
    std::cout << " CP2: DIVERGENT VS ALIGNED BRANCH\n";
    std::cout << "========================================\n";

    std::cout << "Array size       : " << N << "\n";
    std::cout << "Threads per block: " << BLOCK_SIZE << "\n";
    std::cout << "Warp size        : 32\n";
    std::cout << "Grid size        : " << grid_size << "\n";
    std::cout << "Iterations        : " << ITERATIONS << "\n";

    std::cout << "\nRunning divergent kernel...\n";

    float divergent_time = benchmark([&]() {
        divergent_kernel<<<grid_size, BLOCK_SIZE>>>(d_output);
    });

    std::cout << "Running aligned kernel...\n";

    float aligned_time = benchmark([&]() {
        aligned_kernel<<<grid_size, BLOCK_SIZE>>>(d_output);
    });

    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "\n========================================\n";
    std::cout << " RESULTS\n";
    std::cout << "========================================\n";

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Divergent branch: "
              << divergent_time << " ms\n";

    std::cout << "Aligned branch  : "
              << aligned_time << " ms\n";

    std::cout << "\nSpeedup (divergent / aligned): "
              << divergent_time / aligned_time
              << "x\n";

    std::cout << "========================================\n";
    std::cout << " CP2 COMPLETED SUCCESSFULLY\n";
    std::cout << "========================================\n";

    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
