
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h> 
#include <cuda.h>
#include<cuda/barrier>
namespace ptx = cuda::ptx;
using barrier = cuda::barrier<cuda::thread_scope_block>;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
  cudaDriverEntryPointQueryResult driver_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;
  CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status));
  assert(driver_status == cudaDriverEntryPointSuccess);
  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}

constexpr int N_warps = 32; 
constexpr int atom_M = 16; 
constexpr int atom_N = 16; 
constexpr int atom_K = 16;
constexpr int tiled_M = 4;
constexpr int tiled_N = 8;
constexpr int tiled_K = 8;

static_assert(tiled_M * tiled_N == N_warps, "Error");

constexpr int M = atom_M * tiled_M; 
constexpr int N = atom_N * tiled_N; 
constexpr int K = atom_K * tiled_K; 

constexpr uint32_t rank = 2;
uint64_t A_size[rank] = {K,M}; 
uint64_t B_size[rank] = {N,K}; 
uint64_t A_stride[rank-1] = {K*sizeof(__nv_bfloat16)}; 
uint64_t B_stride[rank-1] = {N*sizeof(__nv_bfloat16)};
uint32_t A_smem_size[rank] = {atom_K, atom_M}; 
uint32_t B_smem_size[rank] = {atom_N, atom_K}; 
uint32_t elem_stride[rank] = {1,1};

auto pfn_cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

__global__ void wmma_ker(const __grid_constant__ CUtensorMap tensorMapA, 
                         const __grid_constant__ CUtensorMap tensorMapB, 
                         float *c) 
{
  
    extern __shared__ uint8_t raw_smem[];

    // 1. Fix Pointer Math: Calculate offsets in bytes using raw_smem (uint8_t)
    size_t A_sz_bytes = M * K * sizeof(__nv_bfloat16);
    size_t B_sz_bytes = K * N * sizeof(__nv_bfloat16);

    __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(raw_smem);
    __nv_bfloat16* Bs = reinterpret_cast<__nv_bfloat16*>(raw_smem + A_sz_bytes);

    // Initialize Barrier
    __shared__ barrier bar;
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
    }
    __syncthreads();
    
    barrier::arrival_token token;

    if (threadIdx.x == 0) {
        ptx::cp_async_bulk_tensor(
            ptx::space_shared, ptx::space_global,
            As, &tensorMapA, {0,0}, // Coordinates
            cuda::device::barrier_native_handle(bar)
        );

        cuda::device::barrier_arrive_tx(bar, 1, A_sz_bytes);
    }
    else
    {
      token = bar.arrive();
    }

    bar.wait(std::move(token));


}

void cpu_matmul(const std::vector<float>& h_A, const std::vector<float>& h_B, std::vector<float>& h_C_ref) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }
}

int main() 
{
  size_t A_len = M * K;
  size_t B_len = K * N;
  size_t C_len = M * N;

  size_t A_bytes = sizeof(__nv_bfloat16) * A_len;
  size_t B_bytes = sizeof(__nv_bfloat16) * B_len;
  size_t C_bytes = sizeof(float) * C_len;

  std::vector<float> h_A_float(A_len);
  std::vector<float> h_B_float(B_len);
  std::vector<float> h_C_ref(C_len);
  std::vector<float> h_C_gpu(C_len);

  std::mt19937 gen(1337);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  for(auto &x : h_A_float) x = dis(gen);
  for(auto &x : h_B_float) x = dis(gen);

  std::vector<__nv_bfloat16> h_A_bf16(A_len);
  std::vector<__nv_bfloat16> h_B_bf16(B_len);

  for(size_t i = 0; i < A_len; ++i) h_A_bf16[i] = __float2bfloat16(h_A_float[i]);
  for(size_t i = 0; i < B_len; ++i) h_B_bf16[i] = __float2bfloat16(h_B_float[i]);

  __nv_bfloat16 *d_A, *d_B;
  float *d_C;
  
  cudaMalloc(&d_A, A_bytes);
  cudaMalloc(&d_B, B_bytes);
  cudaMalloc(&d_C, C_bytes);

  cudaMemcpy(d_A, h_A_bf16.data(), A_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B_bf16.data(), B_bytes, cudaMemcpyHostToDevice);
  cudaMemset(d_C, 0, C_bytes);

  int threads_per_warp = 32;
  int block_size = N_warps * threads_per_warp;
  size_t smem_size = A_bytes + B_bytes + C_bytes;

  cudaFuncSetAttribute(wmma_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  CUtensorMap tMap_A{};
  CUtensorMap tMap_B{}; 
  
  CUresult res_A = pfn_cuTensorMapEncodeTiled(
    &tMap_A, 
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    rank,
    d_A, 
    A_size,
    A_stride,
    A_smem_size,
    elem_stride,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  assert(res_A == CUDA_SUCCESS);
  
  CUresult res_B = pfn_cuTensorMapEncodeTiled(
    &tMap_B, 
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    rank,
    d_B, 
    B_size,
    B_stride,
    B_smem_size,
    elem_stride,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  assert(res_B == CUDA_SUCCESS);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  wmma_ker<<<1, block_size, smem_size>>>(tMap_A, tMap_B, d_C);

  cudaEventRecord(start);
  wmma_ker<<<1, block_size, smem_size>>>(tMap_A, tMap_B, d_C);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_C_gpu.data(), d_C, C_bytes, cudaMemcpyDeviceToHost);

  cpu_matmul(h_A_float, h_B_float, h_C_ref);

  float max_error = 0.0f;
  for(size_t i = 0; i < C_len; ++i) {
      float diff = std::abs(h_C_ref[i] - h_C_gpu[i]);
      if(diff > max_error) max_error = diff;
  }

  double flops = 2.0 * M * N * K;
  double gflops = (flops * 1e-9) / (milliseconds / 1000.0);

  printf("M: %d, N: %d, K: %d\n", M, N, K);
  printf("Time: %.4f ms\n", milliseconds);
  printf("Performance: %.4f GFLOPS\n", gflops);
  printf("Max Error: %f\n", max_error);

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  cudaEventDestroy(start); cudaEventDestroy(stop);

  return 0;
}