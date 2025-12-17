#include <stdio.h>
#include <cute/tensor.hpp>

using namespace cute;

int main()
{
    // 1. Use compile-time constants (Int<X>) for static layouts.
    // This allows CuTe to reason about alignment and vectorization at compile time.
    auto M = Int<4096>{};
    auto K = Int<4096>{};
    auto wmma_M = Int<16>{};
    auto wmma_K = Int<16>{};

    // 2. Define the Layout
    // Shape: (4096, 4096), Stride: (4096, 1) -> Row Major
    auto A_init = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));

    // 3. Define the Tile (The Fix)
    // zipped_divide needs a SHAPE to split by, not a Layout.
    // We just want to split the (M, K) coords into ((16, rest), (16, rest)).
    auto A_atom_tiler = make_shape(wmma_M, wmma_K);

    printf("Original Layout:\n");
    print(A_init);
    printf("\n\nTile Shape:\n");
    print(A_atom_tiler);
    printf("\n\n");

    // 4. Perform the Division
    // zipped_divide splits A_init by A_atom_tiler and reorders modes.
    auto zd = zipped_divide(A_init, A_atom_tiler);

    printf("Zipped Layout:\n");
    print(zd);
    printf("\n");

    return 0;
}