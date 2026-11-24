#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

#define N (1 << 24)
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define ITERATIONS 10
#define WORK_ITERATIONS 64

#define CUDA_CHECK(call)                                           \
{                                                                  \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)     \
                  << "\nLine: " << __LINE__ << std::endl;           \
        exit(EXIT_FAILURE);                                         \
    }                                                              \
}

// Each path performs similar computational work,
// but uses a different coefficient.

__device__ __forceinline__
float path0(float x)
{
    #pragma unroll 1
    for (int i = 0; i < WORK_ITERATIONS; i++)
        x = x * 1.000001f + 0.000001f;
    return x;
}

__device__ __forceinline__
float path1(float x)
{
    #pragma unroll 1
    for (int i = 0; i < WORK_ITERATIONS; i++)
        x = x * 1.000002f + 0.000002f;
    return x;
}

__device__ __forceinline__
float path2(float x)
{
    #pragma unroll 1
    for (int i = 0; i < WORK_ITERATIONS; i++)
        x = x * 1.000003f + 0.000003f;
    return x;
}

__device__ __forceinline__
float path3(float x)
{
    #pragma unroll 1
    for (int i = 0; i < WORK_ITERATIONS; i++)
        x = x * 1.000004f + 0.000004f;
    return x;
}

__device__ __forceinline__
float path4(float x)
{
    #pragma unroll 1
    for (int i = 0; i < WORK_ITERATIONS; i++)
        x = x * 1.000005f + 0.000005f;
    return x;
}

__device__ __forceinline__
float path5(float x)
{
    #pragma unroll 1
    for (int i = 0; i < WORK_ITERATIONS; i++)
        x = x * 1.000006f + 0.000006f;
    return x;
}

__device__ __forceinline__
float path6(float x)
{
    #pragma unroll 1
    for (int i = 0; i < WORK_ITERATIONS; i++)
        x = x * 1.000007f + 0.000007f;
    return x;
}

__device__ __forceinline__
float path7(float x)
{
    #pragma unroll 1
    for (int i = 0; i < WORK_ITERATIONS; i++)
        x = x * 1.000008f + 0.000008f;
    return x;
}

// Kernel with PATHS distinct execution paths per warp.

template<int PATHS>
__global__ void path_count_kernel(
    const float* input,
    float* output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    int lane = threadIdx.x % WARP_SIZE;

    // Divide the 32 lanes into PATHS groups.
    int lanes_per_path = WARP_SIZE / PATHS;

    int path_id = lane / lanes_per_path;

    float x = input[idx];
    float result = x;

    if constexpr (PATHS == 1)
    {
        result = path0(x);
    }
    else if constexpr (PATHS == 2)
    {
        if (path_id == 0)
            result = path0(x);
        else
            result = path1(x);
    }
    else if constexpr (PATHS == 4)
    {
        if (path_id == 0)
            result = path0(x);
        else if (path_id == 1)
            result = path1(x);
        else if (path_id == 2)
            result = path2(x);
        else
            result = path3(x);
    }
    else if constexpr (PATHS == 8)
    {
        if (path_id == 0)
            result = path0(x);
        else if (path_id == 1)
            result = path1(x);
        else if (path_id == 2)
            result = path2(x);
        else if (path_id == 3)
            result = path3(x);
        else if (path_id == 4)
            result = path4(x);
        else if (path_id == 5)
            result = path5(x);
        else if (path_id == 6)
            result = path6(x);
        else
            result = path7(x);
    }

    output[idx] = result;
}

// CPU reference for checking correctness.

float cpu_reference(float x, int path)
{
    float coefficient = 1.0f + (path + 1) * 0.000001f;
    float addition = (path + 1) * 0.000001f;

    for (int i = 0; i < WORK_ITERATIONS; i++)
        x = x * coefficient + addition;

    return x;
}

template<int PATHS>
bool validate_result(const float* input, const float* output)
{
    int lanes_per_path = WARP_SIZE / PATHS;

    // Check first 1000 elements.
    for (int i = 0; i < 1000; i++)
    {
        int lane = i % WARP_SIZE;
        int path = lane / lanes_per_path;

        float expected = cpu_reference(input[i], path);

        if (fabs(output[i] - expected) > 0.0001f)
            return false;
    }

    return true;
}

template<int PATHS>
float run_experiment(
    float* d_input,
    float* d_output,
    float* h_input,
    float* h_output)
{
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warm-up launches.
    for (int i = 0; i < 3; i++)
    {
        path_count_kernel<PATHS><<<grid, BLOCK_SIZE>>>(
            d_input, d_output, N);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < ITERATIONS; i++)
    {
        path_count_kernel<PATHS><<<grid, BLOCK_SIZE>>>(
            d_input, d_output, N);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &elapsed_ms, start, stop));

    // Average time per kernel launch.
    elapsed_ms /= ITERATIONS;

    CUDA_CHECK(cudaMemcpy(
        h_output, d_output,
        N * sizeof(float),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return elapsed_ms;
}

int main()
{
    std::cout << "========================================\n";
    std::cout << " CP5: PATH COUNT PER WARP\n";
    std::cout << "========================================\n";

    std::cout << "Array size          : " << N << "\n";
    std::cout << "Threads per block   : " << BLOCK_SIZE << "\n";
    std::cout << "Warp size           : " << WARP_SIZE << "\n";
    std::cout << "Grid size           : "
              << (N + BLOCK_SIZE - 1) / BLOCK_SIZE << "\n";
    std::cout << "Work per path       : "
              << WORK_ITERATIONS << " iterations\n";
    std::cout << "Timed repetitions    : " << ITERATIONS << "\n";

    float* h_input = new float[N];
    float* h_output = new float[N];

    for (int i = 0; i < N; i++)
        h_input[i] = ((i % 100) + 1) / 100.0f;

    float *d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_input, h_input,
        N * sizeof(float),
        cudaMemcpyHostToDevice));

    std::cout << "\nRunning 1-path kernel...\n";
    float t1 = run_experiment<1>(
        d_input, d_output, h_input, h_output);
    bool c1 = validate_result<1>(h_input, h_output);

    std::cout << "Running 2-path kernel...\n";
    float t2 = run_experiment<2>(
        d_input, d_output, h_input, h_output);
    bool c2 = validate_result<2>(h_input, h_output);

    std::cout << "Running 4-path kernel...\n";
    float t4 = run_experiment<4>(
        d_input, d_output, h_input, h_output);
    bool c4 = validate_result<4>(h_input, h_output);

    std::cout << "Running 8-path kernel...\n";
    float t8 = run_experiment<8>(
        d_input, d_output, h_input, h_output);
    bool c8 = validate_result<8>(h_input, h_output);

    std::cout << "\n========================================\n";
    std::cout << " RESULTS\n";
    std::cout << "========================================\n";

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "1 path  : " << t1 << " ms | Correctness: "
              << (c1 ? "PASS" : "FAIL") << "\n";

    std::cout << "2 paths : " << t2 << " ms | Correctness: "
              << (c2 ? "PASS" : "FAIL") << "\n";

    std::cout << "4 paths : " << t4 << " ms | Correctness: "
              << (c4 ? "PASS" : "FAIL") << "\n";

    std::cout << "8 paths : " << t8 << " ms | Correctness: "
              << (c8 ? "PASS" : "FAIL") << "\n";

    std::cout << "\n========================================\n";
    std::cout << " RELATIVE TO 1-PATH BASELINE\n";
    std::cout << "========================================\n";

    std::cout << "2-path ratio : " << t2 / t1 << "x\n";
    std::cout << "4-path ratio : " << t4 / t1 << "x\n";
    std::cout << "8-path ratio : " << t8 / t1 << "x\n";

    bool all_passed = c1 && c2 && c4 && c8;

    std::cout << "\n========================================\n";

    if (all_passed)
        std::cout << " ALL CORRECTNESS CHECKS PASSED\n";
    else
        std::cout << " CORRECTNESS CHECK FAILED\n";

    std::cout << " CP5 COMPLETED\n";
    std::cout << "========================================\n";

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    delete[] h_input;
    delete[] h_output;

    return all_passed ? 0 : 1;
}