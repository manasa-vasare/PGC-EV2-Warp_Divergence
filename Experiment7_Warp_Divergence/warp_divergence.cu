
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>

#define N (1 << 24)          // 16,777,216 elements
#define BLOCK_SIZE 256
#define ITERATIONS 10

// CUDA error checking
#define CUDA_CHECK(call)                                             \
    do {                                                             \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            std::cerr << "CUDA Error: "                               \
                      << cudaGetErrorString(err)                       \
                      << "\nFile: " << __FILE__                        \
                      << "\nLine: " << __LINE__ << std::endl;          \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)


// ============================================================
// LONG COMPUTATION PATH: 64 iterations
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
// SHORT COMPUTATION PATH: 8 iterations
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
// KERNEL 1: DIVERGENT BRANCH
// Threads within the same warp take different paths.
// ============================================================
__global__
void divergent_branch(float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    float x = (float)(idx % 1000) + 1.0f;

    if (idx % 2 == 0)
    {
        output[idx] = long_work(x);
    }
    else
    {
        output[idx] = short_work(x);
    }
}


// ============================================================
// KERNEL 2: WARP-ALIGNED BRANCH
// All 32 threads in a warp follow the same path.
// ============================================================
__global__
void aligned_branch(float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    float x = (float)(idx % 1000) + 1.0f;

    if ((idx / 32) % 2 == 0)
    {
        output[idx] = long_work(x);
    }
    else
    {
        output[idx] = short_work(x);
    }
}


// ============================================================
// KERNEL 3: DATA-DEPENDENT BRANCH
// Path selection depends on input data.
// ============================================================
__global__
void data_dependent_branch(
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
// KERNEL 4: BRANCH-FREE VERSION
// Computes both paths and selects the result.
// ============================================================
__global__
void branch_free(
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
// Uses CUDA events to measure kernel execution time.
// ============================================================
template <typename Kernel>
float benchmark(Kernel kernel)
{
    // Warm-up executions
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

    // Repeated executions
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

    // Average execution time
    return total_ms / ITERATIONS;
}


// ============================================================
// MAIN PROGRAM
// ============================================================
int main()
{
    std::cout << "========================================\n";
    std::cout << " CUDA WARP DIVERGENCE EXPERIMENT\n";
    std::cout << "========================================\n";

    std::cout << "Array size        : " << N << std::endl;
    std::cout << "Threads per block : " << BLOCK_SIZE << std::endl;
    std::cout << "Warp size         : 32" << std::endl;
    std::cout << "Benchmark runs    : " << ITERATIONS << std::endl;

    // --------------------------------------------------------
    // HOST MEMORY ALLOCATION
    // --------------------------------------------------------
    std::vector<float> h_input(N);
    std::vector<float> h_output(N);

    // Generate input data
    for (int i = 0; i < N; i++)
    {
        h_input[i] = (float)(i % 100) / 100.0f;
    }

    // --------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    // --------------------------------------------------------
    float *d_input = nullptr;
    float *d_output = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_input, N * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_output, N * sizeof(float)));

    // Copy input data to GPU
    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        N * sizeof(float),
        cudaMemcpyHostToDevice));

    // --------------------------------------------------------
    // KERNEL CONFIGURATION
    // --------------------------------------------------------
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::cout << "Grid size         : " << grid_size << std::endl;

    // --------------------------------------------------------
    // EXPERIMENT 1: DIVERGENT BRANCH
    // --------------------------------------------------------
    std::cout << "\nRunning divergent branch..." << std::endl;

    float divergent_time = benchmark([&]() {
        divergent_branch<<<grid_size, BLOCK_SIZE>>>(d_output);
    });


    // --------------------------------------------------------
    // EXPERIMENT 2: WARP-ALIGNED BRANCH
    // --------------------------------------------------------
    std::cout << "Running aligned branch..." << std::endl;

    float aligned_time = benchmark([&]() {
        aligned_branch<<<grid_size, BLOCK_SIZE>>>(d_output);
    });


    // --------------------------------------------------------
    // EXPERIMENT 3: DATA-DEPENDENT BRANCH
    // --------------------------------------------------------
    std::cout << "Running data-dependent branch..." << std::endl;

    float data_branch_time = benchmark([&]() {
        data_dependent_branch<<<grid_size, BLOCK_SIZE>>>(
            d_input, d_output);
    });


    // --------------------------------------------------------
    // EXPERIMENT 4: BRANCH-FREE
    // --------------------------------------------------------
    std::cout << "Running branch-free version..." << std::endl;

    float branch_free_time = benchmark([&]() {
        branch_free<<<grid_size, BLOCK_SIZE>>>(
            d_input, d_output);
    });


    // --------------------------------------------------------
    // COPY RESULTS BACK TO CPU
    // --------------------------------------------------------
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        N * sizeof(float),
        cudaMemcpyDeviceToHost));


    // --------------------------------------------------------
    // PERFORMANCE RESULTS
    // --------------------------------------------------------
    std::cout << "\n========================================\n";
    std::cout << " PERFORMANCE RESULTS\n";
    std::cout << "========================================\n";

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Divergent branch     : "
              << divergent_time << " ms\n";

    std::cout << "Aligned branch       : "
              << aligned_time << " ms\n";

    std::cout << "Data-dependent branch: "
              << data_branch_time << " ms\n";

    std::cout << "Branch-free version  : "
              << branch_free_time << " ms\n";


    // --------------------------------------------------------
    // SPEEDUP CALCULATION
    // --------------------------------------------------------
    std::cout << "\n========================================\n";
    std::cout << " SPEEDUP COMPARISON\n";
    std::cout << "========================================\n";

    if (aligned_time > 0.0f)
    {
        std::cout << "Divergent vs aligned speedup: "
                  << divergent_time / aligned_time
                  << "x\n";
    }

    if (branch_free_time > 0.0f)
    {
        std::cout << "Data branch vs branch-free: "
                  << data_branch_time / branch_free_time
                  << "x\n";
    }


    // --------------------------------------------------------
    // SAMPLE OUTPUT
    // --------------------------------------------------------
    std::cout << "\nSample output[0]: "
              << h_output[0] << std::endl;

    std::cout << "Sample output[100]: "
              << h_output[100] << std::endl;


    // --------------------------------------------------------
    // FREE GPU MEMORY
    // --------------------------------------------------------
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));


    std::cout << "\n========================================\n";
    std::cout << " EXPERIMENT COMPLETED SUCCESSFULLY\n";
    std::cout << "========================================\n";

    return 0;
}
