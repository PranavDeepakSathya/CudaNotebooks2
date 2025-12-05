#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Forward declaration of the kernel inside wmma_kernel.cu
// This tells the compiler "this function exists, trust me, the linker will find it."
__global__ void wmma_ker(__nv_bfloat16 *a, __nv_bfloat16 *b, float *c, int n);

// Error checking helper
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

int main() {
    int N = 16;
    size_t bytes_bf16 = N * N * sizeof(__nv_bfloat16);
    size_t bytes_float = N * N * sizeof(float);

    // 1. Host Memory Allocation
    std::vector<float> h_a_f32(N * N);
    std::vector<float> h_b_f32(N * N);
    std::vector<__nv_bfloat16> h_a_bf16(N * N);
    std::vector<__nv_bfloat16> h_b_bf16(N * N);
    std::vector<float> h_c_gpu(N * N);
    std::vector<float> h_c_ref(N * N, 0.0f);

    // 2. Initialize Data
    std::mt19937 gen(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::cout << "Initializing data for N=" << N << "..." << std::endl;
    for (int i = 0; i < N * N; i++) {
        h_a_f32[i] = dist(gen);
        h_b_f32[i] = dist(gen);
        
        // Convert float to bfloat16 using CUDA host intrinsic
        h_a_bf16[i] = __float2bfloat16(h_a_f32[i]);
        h_b_bf16[i] = __float2bfloat16(h_b_f32[i]);
    }

    // 3. Device Memory Allocation
    __nv_bfloat16 *d_a, *d_b;
    float *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes_bf16));
    CHECK_CUDA(cudaMalloc(&d_b, bytes_bf16));
    CHECK_CUDA(cudaMalloc(&d_c, bytes_float));

    // 4. Copy to Device
    CHECK_CUDA(cudaMemcpy(d_a, h_a_bf16.data(), bytes_bf16, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b_bf16.data(), bytes_bf16, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_c, 0, bytes_float));

    // 5. Setup Events for Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 6. Launch Kernel
    std::cout << "Launching Kernel..." << std::endl;
    
    // Warmup
    wmma_ker<<<1, 32>>>(d_a, d_b, d_c, N); 
    
    // Timed Run
    cudaEventRecord(start);
    wmma_ker<<<1, 32>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;

    // 7. Copy Back
    CHECK_CUDA(cudaMemcpy(h_c_gpu.data(), d_c, bytes_float, cudaMemcpyDeviceToHost));

    // 8. CPU Verification (Naive Matrix Mul)
    std::cout << "Verifying..." << std::endl;
    float max_diff = 0.0f;
    
    // Since input is Col Major for A and Row Major for B (based on your kernel fragments), 
    // we need to be careful.
    // Your Kernel:
    // A: wmma::col_major -> Expects Input A to be stored column-major? 
    //    Actually, wmma::load_matrix_sync interprets the memory based on that tag.
    //    If your flat array `a` is row-major (standard C), loading it as `col_major` transposes it.
    
    // For simplicity, let's assume standard row-major matmul logic for verification
    // and see if it matches. If not, the `col_major` loading in kernel is the "transpose".
    
    // Standard Matmul: C = A * B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                // Converting back to float for CPU math simulation
                float a_val = __bfloat162float(h_a_bf16[i * N + k]); // Row major access
                
                // WAIT: Your kernel loads A as COL_MAJOR. 
                // That means it treats index [i*N+k] as row k, col i.
                // Let's stick to simple checks first.
                
                float b_val = __bfloat162float(h_b_bf16[k * N + j]);
                sum += a_val * b_val;
            }
            h_c_ref[i * N + j] = sum;
        }
    }

    // Compare
    // Note: If the kernel load layout (col_major) was intentional to transpose A, 
    // this verification will fail wildly. If it was accidental, we'll see it here.
    for (int i = 0; i < N * N; i++) {
        float diff = std::abs(h_c_gpu[i] - h_c_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "Max Difference: " << max_diff << std::endl;
    if (max_diff < 0.5f) { // BF16 precision is low, so tolerance is loose
        std::cout << "TEST PASSED" << std::endl;
    } else {
        std::cout << "TEST FAILED (Check matrix layout/transposition)" << std::endl;
        std::cout << "GPU[0]: " << h_c_gpu[0] << ", CPU[0]: " << h_c_ref[0] << std::endl;
    }

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}