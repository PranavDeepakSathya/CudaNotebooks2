#include<cuda_bf16.h> 
#include<mma.h>

using namespace nvcuda;

 __global__ void wmma_ker(__nv_bfloat16 *a, __nv_bfloat16 *b, float *c, int n) 
{
   wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

   wmma::fill_fragment(c_frag, 0.0f);

   wmma::load_matrix_sync(a_frag, a, 16);
   wmma::load_matrix_sync(b_frag, b, 16);   

   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}
