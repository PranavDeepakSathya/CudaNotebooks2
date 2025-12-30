
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaTypedefs.h> 
#include <cuda_bf16.h>

// Modern C++ CUDA headers for TMA
#include <cuda/barrier>
#include <cuda/ptx>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

#define CUDA_CHECK(call) \
    { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(err); \
        } \
    }

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* ptr = nullptr;
    CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &driver_status));
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
}

// --- Constants --- 
constexpr int byte_aligner = 128; 
constexpr uint32_t M = 16; 
constexpr uint32_t N = 16; 
constexpr uint32_t BM = 8; 
constexpr uint32_t BN = 8;
constexpr uint32_t GM = M/BM; 
constexpr uint32_t GN = N/BN;
constexpr uint32_t rank = 2; 

// Note: Stride calculation is bytes to skip to reach the next element in the next dimension
constexpr uint64_t tensor_shape[rank] = {N, M}; // {Fast-Dim, Slow-Dim}
constexpr uint64_t tensor_stride[rank-1] = {N * sizeof(__nv_bfloat16)}; 
constexpr uint32_t smem_box_shape[rank] = {BN, BM}; 
constexpr uint32_t element_stride[rank] = {1, 1};
constexpr size_t gmem_tensor_size = M * N * sizeof(__nv_bfloat16);

__global__ void tma_kernel(__nv_bfloat16* A, const __grid_constant__ CUtensorMap tensor_map)
{
    // Global Coordinates
    uint x = blockIdx.x * blockDim.x; 
    uint y = blockIdx.y * blockDim.y;

    // Shared Memory Buffer
    // alignas(128) is crucial for TMA
    __shared__ alignas(byte_aligner) __nv_bfloat16 As[BM][BN];  
    __shared__ alignas(byte_aligner) __nv_bfloat16 Bs[BM][BN];  

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    if (threadIdx.x == 0) {
        init(&bar, blockDim.x * blockDim.y); // Expecting all threads in block
        cde::fence_proxy_async_shared_cta(); 
    }
    __syncthreads(); 

    barrier::arrival_token token;
    if (threadIdx.x == 0)
    {
        // Issue TMA Load
        // Note: x (column) is dim-0, y (row) is dim-1 based on tensor_shape definition
        cde::cp_async_bulk_tensor_2d_global_to_shared(&As, &tensor_map, x, y, bar);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&Bs, &tensor_map, x, y, bar);
        // Transaction count is size of the box in bytes
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(As) + sizeof(Bs));
    } 
    else {
        token = bar.arrive();
    }

    // Wait for TMA transfer to complete
    bar.wait(std::move(token));
    cde::fence_proxy_async_shared_cta();
    __syncthreads(); // Ensure all threads see the data before reading

    // Manual Write Back to Verify Data
    uint smem_col = threadIdx.x; 
    uint smem_row = threadIdx.y; 
    
    // Bounds check to be safe (though block dims match constants here)
    if(smem_col < BN && smem_row < BM) {
        uint gmem_row = y + smem_row;
        uint gmem_col = x + smem_col;
        
        // Since we used SWIZZLE_NONE, we can read As[][] linearly.
        // If we used SWIZZLE_32B, As[][] would contain scrambled data here.
        A[gmem_row * N + gmem_col] = Bs[smem_row][smem_col];
    }
}

int main()
{
    __nv_bfloat16 *A_h, *A_d; 
    
    // Using simple malloc for host to ensure compatibility
    A_h = (__nv_bfloat16*)malloc(gmem_tensor_size);
    CUDA_CHECK(cudaMalloc(&A_d, gmem_tensor_size)); 
    
    // Initialize
    for (int i = 0; i < M * N; i++)
    {
        A_h[i] = __float2bfloat16((float)i); 
    }
    
    // Reset Device Memory to 0 to ensure we are actually reading the kernel output
    CUDA_CHECK(cudaMemset(A_d, 0, gmem_tensor_size));
    
    // Copy input data (technically not needed since we write it all back, 
    // but good if we were doing a transform)
    // Actually, to prove TMA works, let's upload A_h to a source buffer 
    // and write to a destination buffer. But for this specific fix, 
    // we will stick to your logic: Overwriting A_d with itself via SMEM.
    CUDA_CHECK(cudaMemcpy(A_d, A_h, gmem_tensor_size, cudaMemcpyHostToDevice)); 

    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    CUtensorMap tensor_map{};
    
    // The pointer provided to TensorMap must be the source of the data
    void *tensor_ptr = (void*)A_d; 
    
    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map,

        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 
        rank,
        tensor_ptr, 
        tensor_shape,
        tensor_stride,
        smem_box_shape,
        element_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
 \
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (res != CUDA_SUCCESS) {
        printf("Tensor Map Encode Failed! Error: %d\n", res);
        return -1;
    }

    dim3 grid(GN, GM); 
    dim3 block(BN, BM);
    
    // Launch
    tma_kernel<<<grid, block>>>(A_d, tensor_map);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Readback
    CUDA_CHECK(cudaMemcpy(A_h, A_d, gmem_tensor_size, cudaMemcpyDeviceToHost));
    


    for (int i = 0; i < M*N; i++) {
        printf("%d, ", (int)__bfloat162float(A_h[i]));
    }
    printf("\n");

    free(A_h);
    cudaFree(A_d);
    return 0;
}