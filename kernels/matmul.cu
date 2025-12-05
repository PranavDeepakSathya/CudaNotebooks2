#include<stdio.h>
#include<cuda_runtime.h> 
#include<cuda.h> 
#include<mma.h> 
#include<cuda_bf16.h>
#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;
constexpr int WGMMA_M = 16; 
constexpr int WGMMA_N = 16; 
constexpr int WGMMA_K = 16; 
constexpr int n_prod = 1; 
constexpr int n_cons = 4; 
constexpr int bM = 32; 
constexpr int bN = 32; 
constexpr int bK = 32;


__global__ void matmul(__nv_bfloat16* A, __nv_bfloat16 *B, __nv_bfloat16 *C, 
                        const __grid_constant__ CUtensorMap tensor_map_A, const __grid_constant__ CUtensorMap tensor_map_B)
{



}
