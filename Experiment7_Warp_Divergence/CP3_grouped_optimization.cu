#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>

#define N (1 << 24)
#define BLOCK_SIZE 256
#define ITERATIONS 10

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << "\nLine: " << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// ============================================
// GPU WORK FUNCTIONS
// ============================================

__device__ __forceinline__ float long_work(float x) {
    #pragma unroll 1
    for (int i = 0; i < 64; i++) {
        x = x * 1.000001f + 0.000001f;
    }
    return x;
}

__device__ __forceinline__ float short_work(float x) {
    #pragma unroll 1
    for (int i = 0; i < 8; i++) {
        x = x * 0.999999f - 0.000001f;
    }
    return x;
}

// ============================================
// INITIALIZE INPUT ARRAY
// ============================================

__global__ void initialize_input(float *input) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        input[idx] = (float)(idx % 100) / 100.0f;
    }
}

// ============================================
// VERSION 1: DATA-DEPENDENT BRANCH
// ============================================

__global__ void data_dependent_kernel(
    const float *input,
    float *output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    float x = input[idx];

    if (x > 0.5f) {
        output[idx] = long_work(x);
    }
    else {
        output[idx] = short_work(x);
    }
}

// ============================================
// VERSION 2: GROUPED DATA
// ============================================

__global__ void grouped_kernel(
    const float *input,
    float *output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    float x = input[idx];

    if (x > 0.5f) {
        output[idx] = long_work(x);
    }
    else {
        output[idx] = short_work(x);
    }
}

// ============================================
// VERSION 3: COMPUTE BOTH / SELECT
// ============================================

__global__ void branch_free_kernel(
    const float *input,
    float *output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    float x = input[idx];

    float a = long_work(x);
    float b = short_work(x);

    output[idx] = (x > 0.5f) ? a : b;
}

// ============================================
// BENCHMARK FUNCTION
// ============================================

template <typename Launch>
float benchmark(Launch launch) {

    // Warm-up launches
    for (int i = 0; i < 3; i++) {
        launch();
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // Timed launches
    for (int i = 0; i < ITERATIONS; i++) {
        launch();
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    ));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / ITERATIONS;
}

// ============================================
// CPU WORK FUNCTIONS FOR VALIDATION
// ============================================

float host_long_work(float x) {
    for (int i = 0; i < 64; i++) {
        x = x * 1.000001f + 0.000001f;
    }

    return x;
}

float host_short_work(float x) {
    for (int i = 0; i < 8; i++) {
        x = x * 0.999999f - 0.000001f;
    }

    return x;
}

// ============================================
// VALIDATE OUTPUT
// ============================================

bool validate(
    const std::vector<float>& input,
    const std::vector<float>& output
) {
    for (int i = 0; i < N; i++) {

        float x = input[i];

        float expected;

        if (x > 0.5f) {
            expected = host_long_work(x);
        }
        else {
            expected = host_short_work(x);
        }

        if (std::abs(output[i] - expected) > 1e-5f) {
            return false;
        }
    }

    return true;
}

// ============================================
// MAIN PROGRAM
// ============================================

int main() {

    float *d_input = nullptr;
    float *d_grouped = nullptr;
    float *d_output = nullptr;

    size_t bytes = (size_t)N * sizeof(float);

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_grouped, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::cout << "========================================\n";
    std::cout << " CP3: DATA-DEPENDENT BRANCH + GROUPING\n";
    std::cout << "========================================\n";

    std::cout << "Array size         : " << N << "\n";
    std::cout << "Threads per block  : " << BLOCK_SIZE << "\n";
    std::cout << "Warp size          : 32\n";
    std::cout << "Grid size          : " << grid_size << "\n";
    std::cout << "Timed repetitions  : " << ITERATIONS << "\n";

    // Initialize input
    initialize_input<<<grid_size, BLOCK_SIZE>>>(d_input);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Host arrays
    std::vector<float> h_input(N);
    std::vector<float> h_grouped;
    std::vector<float> h_output(N);

    // Copy input back for grouping
    CUDA_CHECK(cudaMemcpy(
        h_input.data(),
        d_input,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    // ========================================
    // GROUP DATA ON CPU
    // ========================================

    std::vector<float> short_items;
    std::vector<float> long_items;

    short_items.reserve(N);
    long_items.reserve(N);

    for (float x : h_input) {

        if (x > 0.5f) {
            long_items.push_back(x);
        }
        else {
            short_items.push_back(x);
        }
    }

    h_grouped.reserve(N);

    // Place short-path values together
    h_grouped.insert(
        h_grouped.end(),
        short_items.begin(),
        short_items.end()
    );

    // Place long-path values together
    h_grouped.insert(
        h_grouped.end(),
        long_items.begin(),
        long_items.end()
    );

    std::cout << "\n========== GROUPING INFO ==========\n";

    std::cout << "Short-path elements : "
              << short_items.size() << "\n";

    std::cout << "Long-path elements  : "
              << long_items.size() << "\n";

    std::cout << "Grouping performed on CPU.\n";

    std::cout << "Grouping time is excluded from kernel timing.\n";

    // Copy grouped data to GPU
    CUDA_CHECK(cudaMemcpy(
        d_grouped,
        h_grouped.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    // ========================================
    // EXPERIMENT 1: NORMAL DATA-DEPENDENT
    // ========================================

    float normal_ms = benchmark([&]() {
        data_dependent_kernel<<<grid_size, BLOCK_SIZE>>>(
            d_input,
            d_output,
            N
        );
    });

    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    bool normal_ok = validate(h_input, h_output);

    // ========================================
    // EXPERIMENT 2: GROUPED DATA
    // ========================================

    float grouped_ms = benchmark([&]() {
        grouped_kernel<<<grid_size, BLOCK_SIZE>>>(
            d_grouped,
            d_output,
            N
        );
    });

    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    bool grouped_ok = validate(h_grouped, h_output);

    // ========================================
    // EXPERIMENT 3: COMPUTE BOTH / SELECT
    // ========================================

    float branchfree_ms = benchmark([&]() {
        branch_free_kernel<<<grid_size, BLOCK_SIZE>>>(
            d_input,
            d_output,
            N
        );
    });

    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    bool branchfree_ok = validate(h_input, h_output);

    // ========================================
    // DISPLAY RESULTS
    // ========================================

    std::cout << "\n========================================\n";
    std::cout << "              RESULTS\n";
    std::cout << "========================================\n";

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Data-dependent branch : "
              << normal_ms << " ms"
              << " | Correctness: "
              << (normal_ok ? "PASS" : "FAIL")
              << "\n";

    std::cout << "Grouped data kernel  : "
              << grouped_ms << " ms"
              << " | Correctness: "
              << (grouped_ok ? "PASS" : "FAIL")
              << "\n";

    std::cout << "Compute-both/select  : "
              << branchfree_ms << " ms"
              << " | Correctness: "
              << (branchfree_ok ? "PASS" : "FAIL")
              << "\n";

    // ========================================
    // SPEEDUP CALCULATIONS
    // ========================================

    std::cout << "\n========================================\n";
    std::cout << "       SPEEDUP VS BASELINE\n";
    std::cout << "========================================\n";

    std::cout << "Grouped data speedup : "
              << normal_ms / grouped_ms
              << "x\n";

    std::cout << "Branch-free speedup  : "
              << normal_ms / branchfree_ms
              << "x\n";

    std::cout << "\nNote: CPU grouping and extra memory transfer\n";
    std::cout << "are excluded from grouped kernel timing.\n";

    // ========================================
    // FREE GPU MEMORY
    // ========================================

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_grouped));
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "\n========================================\n";
    std::cout << "       EXPERIMENT COMPLETED\n";
    std::cout << "========================================\n";

    if (normal_ok && grouped_ok && branchfree_ok) {
        std::cout << "All correctness checks PASSED.\n";
        return EXIT_SUCCESS;
    }
    else {
        std::cout << "WARNING: A correctness check FAILED.\n";
        return EXIT_FAILURE;
    }
}