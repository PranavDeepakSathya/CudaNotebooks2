    #include <cstdio>
    #include <cstdlib>
    #include <cassert>
    #include <vector>
    #include <random>

    // CUDA Includes
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
    #include <cudaTypedefs.h>
    #include <cuda/barrier>

    using barrier = cuda::barrier<cuda::thread_scope_block>;
    namespace ptx = cuda::ptx;

    __device__ inline bool is_elected()
    {
        unsigned int tid = threadIdx.x;
        unsigned int warp_id = tid / 32;
        unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0); // Broadcast from lane 0.
        return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF)); // Elect a leader thread among warp 0.
    }


    #define CUDA_CHECK(call)                                                      \
        do {                                                                      \
            cudaError_t err = call;                                               \
            if (err != cudaSuccess) {                                             \
                fprintf(stderr, "CUDA Error at %s:%d - %s: %s\n",                 \
                        __FILE__, __LINE__, cudaGetErrorName(err),                \
                        cudaGetErrorString(err));                                 \
                exit(EXIT_FAILURE);                                               \
            }                                                                     \
        } while (0)

    PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
        cudaDriverEntryPointQueryResult driver_status;
        void* ptr = nullptr;
        CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &driver_status));
        assert(driver_status == cudaDriverEntryPointSuccess);
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
    }


    __global__ void tma_test_kernel(
        const __grid_constant__ CUtensorMap map_A,
        const __grid_constant__ CUtensorMap map_B) 
    {
    
    __shared__ alignas(128) As[BM][BK];
    __shared__ alignas(128) Bs[BK][BN];

    __shared__ barrier bar;

    int t = threadIdx.x; 
    int bdim = blockDim.x;

    int b = blockIdx.x; 
    int w = t/32; 
    int l = t % 32; 

    if (t == 0)
    {
        init(&bar, bdim);
    }

    ___syncthreads();

    int gm = b/GN; 
    int gn = b % GN;
     

    for (int bk = 0; bk < K; bk += BK)
    {
        bm = gm*BM; 
        bn = gn*BN;

        if (is_elected())
        {
            barrier::arrival_token token; 
            int32_t a_coords = {bm,bk};
            int32_t b_coords = {bk,bn};
            ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, &As, &map_A, a_coords, cuda::device::barrier_native_handle(bar));
            ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, &Bs, &map_B, b_coords, cuda::device::barrier_native_hande(bar));
            token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(As)+sizeof(Bs));
        }
        else
        {
            token = bar.arrive();
        }
        bar.wait(std::move(token));


    }





        
    }

    // --- 3. Host Logic ---

    // Constants
    constexpr int M = 2048;
    constexpr int N = 2048; 
    constexpr int K = 2048;
    constexpr int BM = 128; // Tile Size
    constexpr int BN = 128; 
    constexpr int BK = 128;
    constexpr int GM = M/BM;
    constexpr int GN = N/BN;

    int main() {
        printf("[Init] Fetching Driver API...\n");
        auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

        // --- A. Allocation & Data Gen ---
        printf("[Alloc] Allocating Tensors M=%d N=%d K=%d...\n", M, N, K);
        
        size_t size_A = M * K * sizeof(nv_bfloat16);
        size_t size_B = K * N * sizeof(nv_bfloat16);
        
        nv_bfloat16 *h_A = (nv_bfloat16*)malloc(size_A);
        nv_bfloat16 *h_B = (nv_bfloat16*)malloc(size_B);
        
        // Fill with random bfloat16 (just using float cast for simplicity)
        for(int i=0; i < M*K; i++) h_A[i] = __float2bfloat16((float)(rand()) / RAND_MAX);
        for(int i=0; i < K*N; i++) h_B[i] = __float2bfloat16((float)(rand()) / RAND_MAX);

        nv_bfloat16 *d_A, *d_B;
        CUDA_CHECK(cudaMalloc(&d_A, size_A));
        CUDA_CHECK(cudaMalloc(&d_B, size_B));
        
        CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

        // --- B. Tensor Map Creation ---
        printf("[TMA] Encoding Tensor Maps...\n");
        
        CUtensorMap tmap_A{}; 
        CUtensorMap tmap_B{};
        
        constexpr uint32_t rank = 2;
        
        // Config for A (Row Major: M x K)
        uint64_t size_A_dims[rank] = {K, M}; // {Cols, Rows} - Note: Driver API is usually {fastest, slowest}
        uint64_t stride_A_bytes[rank-1] = {K * sizeof(nv_bfloat16)};
        uint32_t box_A[rank] = {BK, BM}; // Tile Size
        uint32_t elem_strides[rank] = {1, 1};

        cuTensorMapEncodeTiled(
            &tmap_A,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            rank,
            d_A, // Global Pointer
            size_A_dims,
            stride_A_bytes,
            box_A,
            elem_strides,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE, // Use 128B for real speed later
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );

        // Config for B (Row Major: K x N)
        uint64_t size_B_dims[rank] = {N, K}; 
        uint64_t stride_B_bytes[rank-1] = {N * sizeof(nv_bfloat16)};
        uint32_t box_B[rank] = {BN, BK}; 

        cuTensorMapEncodeTiled(
            &tmap_B,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            rank,
            d_B, 
            size_B_dims,
            stride_B_bytes,
            box_B,
            elem_strides,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );

        // --- C. Launch ---
        printf("[Launch] Launching Kernel (1024 threads)...\n");
        
        // Note: CUtensorMap is passed by Value to the kernel. 
        // The driver handles placing it in constant bank.
        tma_test_kernel<<<1, 1024>>>(tmap_A, tmap_B);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        printf("[Success] Done.\n");

        free(h_A); free(h_B);
        cudaFree(d_A); cudaFree(d_B);
        return 0;
    }