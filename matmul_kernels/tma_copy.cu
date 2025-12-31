#include <cute/tensor.hpp>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

using namespace cute;

// 1. Drop the "cde" namespace nonsense.
// 2. Use uint64_t for the barrier (CuTe style).

__device__ void barrierInit(uint64_t &tma_load_mbar, int numThreads) 
{
  // FIX: Correct warp index calculation
  int warp_idx = threadIdx.x / 32;
  
  // Elect one leader in the warp (usually lane 0)
  int lane_predicate = cute::elect_one_sync();
  
  // Only the first thread of the first warp initializes the barrier
  if (warp_idx == 0 && lane_predicate) {
    tma_load_mbar = 0;
    cute::initialize_barrier(tma_load_mbar, numThreads);
  }
  
  __syncthreads();
  
  // FIX: Use raw PTX for the fence instead of the missing 'cde' function
  // This ensures the barrier initialization is visible to the TMA engine.
  asm volatile("fence.proxy.async.shared::cta; " ::: "memory");
}



__global__ void check()
{
  __shared__ alignas(128) float A[128];
  
  
  __shared__ uint64_t tma_barrier;
  
  // Now you can pass it safely
  barrierInit(tma_barrier, 128);
}

int main()
{
  check<<<1, 128>>>();
  cudaDeviceSynchronize();
  return 0;
}