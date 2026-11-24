
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


// ============================================================
// LONG COMPUTATION PATH: 64 ITERATIONS
// ============================================================
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


// ============================================================
// SHORT COMPUTATION PATH: 8 ITERATIONS
// ============================================================
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


// ============================================================
// INPUT INITIALIZATION KERNEL
// Generates values between 0.00 and 0.99
// ============================================================
__global__
void initialize_input(float *input)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        input[idx] = (float)(idx % 100) / 100.0f;
    }
}


// ============================================================
// KERNEL 1: DATA-DEPENDENT BRANCH
// Input value determines which path is executed.
// ============================================================
__global__
void data_dependent_kernel(
    const float *input,
    float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    float x = input[idx];

    if (x > 0.5f)
    {
        output[idx] = long_work(x);
    }
    else
    {
        output[idx] = short_work(x);
    }
}


// ============================================================
// KERNEL 2: BRANCH-FREE / PREDICATED VERSION
// Computes both paths, then selects the result.
// ============================================================
__global__
void branch_free_kernel(
    const float *input,
    float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    float x = input[idx];

    float long_result = long_work(x);
    float short_result = short_work(x);

    output[idx] = (x > 0.5f)
                    ? long_result
                    : short_result;
}


// ============================================================
// BENCHMARK FUNCTION
// ============================================================
template <typename Kernel>
float benchmark(Kernel kernel)
{
    // Warm-up
    for (int i = 0; i < 3; i++)
    {
        kernel();
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < ITERATIONS; i++)
    {
        kernel();
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / ITERATIONS;
}


// ============================================================
// MAIN PROGRAM
// ============================================================
int main()
{
    float *d_input = nullptr;
    float *d_output = nullptr;

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(
        &d_input, N * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_output, N * sizeof(float)));

    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::cout << "========================================\n";
    std::cout << " CP3: DATA-DEPENDENT BRANCH\n";
    std::cout << "========================================\n";

    std::cout << "Array size        : " << N << "\n";
    std::cout << "Threads per block : " << BLOCK_SIZE << "\n";
    std::cout << "Warp size         : 32\n";
    std::cout << "Grid size         : " << grid_size << "\n";
    std::cout << "Benchmark runs    : " << ITERATIONS << "\n";

    // Initialize input array
    std::cout << "\nInitializing input data...\n";

    initialize_input<<<grid_size, BLOCK_SIZE>>>(d_input);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark data-dependent branch
    std::cout << "Running data-dependent kernel...\n";

    float data_time = benchmark([&]() {
        data_dependent_kernel<<<grid_size, BLOCK_SIZE>>>(
            d_input, d_output);
    });

    // Benchmark branch-free version
    std::cout << "Running branch-free kernel...\n";

    float branch_free_time = benchmark([&]() {
        branch_free_kernel<<<grid_size, BLOCK_SIZE>>>(
            d_input, d_output);
    });

    CUDA_CHECK(cudaDeviceSynchronize());

    // Print results
    std::cout << "\n========================================\n";
    std::cout << " PERFORMANCE RESULTS\n";
    std::cout << "========================================\n";

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Data-dependent branch: "
              << data_time << " ms\n";

    std::cout << "Branch-free version : "
              << branch_free_time << " ms\n";

    std::cout << "\n========================================\n";
    std::cout << " SPEEDUP COMPARISON\n";
    std::cout << "========================================\n";

    std::cout << "Data branch / branch-free: "
              << data_time / branch_free_time
              << "x\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "\n========================================\n";
    std::cout << " CP3 COMPLETED SUCCESSFULLY\n";
    std::cout << "========================================\n";

    return 0;
}
