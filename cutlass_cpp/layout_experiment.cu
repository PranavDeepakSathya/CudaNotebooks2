#include<stdio.h> 
#include <cute/tensor.hpp>


int main()

{
  auto layout = cute::make_layout(cute::make_shape(4,5, cute::make_shape(3,7)), cute::make_stride(1,4,cute::make_stride(6,7)));
  cute::print(layout);
  printf("\n");
  for (int i = 0; i < 21; i++)
  {
    auto x = layout(0,0,(i));
    cute::print(x);
    printf(", ");
  }
  printf("\n");
  /*
  so how are we able to index into L = (4,5,(3,7)):(1,4,(6,7))?  we are giving a 1d co-ordinate for the inner shape 
   0, 6, 12, 7, 13, 19, 14, 20, 26, 21, 27, 33, 28, 34, 40, 35, 41, 47, 42, 48, 54, is what we get
  now we wonder if this: get_layout_out(0,0,(x)) wrt to layout L = (a,b,(c,d)): (m,n,(p,q))
  is equivalent to get_layout_out(x) wrt to L' = (c,d):(p,q)
  let us test it out
  */
  auto layout_inner = cute::make_layout(cute::make_shape(3,7), cute::make_stride(6,7)); 
  for (int i = 0; i < 21; i++)
  {
    auto x = layout_inner(i);
    cute::print(x); 
    printf(", ");
  }
  printf("\n"); 
   /*
  Indeed! we get the same thing 0, 6, 12, 7, 13, 19, 14, 20, 26, 21, 27, 33, 28, 34, 40, 35, 41, 47, 42, 48, 54, 
  now, let us make another observation
  what is the images of (0,1,(x)) under get layout, well obviousy, there is some nice pattern to see so we see it first
  4, 10, 16, 11, 17, 23, 18, 24, 30, 25, 31, 37, 32, 38, 44, 39, 45, 51, 46, 52, 58,
  this is just 4 +_all (0, 6, 12, 7, 13, 19, 14, 20, 26, 21, 27, 33, 28, 34, 40, 35, 41, 47, 42, 48, 54,)!!! 
  
  so how do we then, given a profile ((),( , ())) and a flat layout, how do we calculate eq co-ordinates? well I guess I have to read 
  colfax paper for that LMAO 
  But okay lets try nevertheless, 
  they say 
  Each coordinate into the shape (3,(2,3)) has two equivalent coordinates and all equivalent coordinates map to the same natural 
  coordinate. To emphasize again, because all of the above coordinates are valid inputs, a Layout with Shape (3,(2,3)) can be 
  used as if it is a 1-D array of 18 elements by using the 1-D coordinates, a 2-D matrix of 3x6 elements by using the 2-D coordinates,
  or a h-D tensor of 3x(2x3) elements by using the h-D (natural) coordinates.

  then, for our shape (4,5,(3,7)) we can treat it as a 3d object by using L' = (4,5, size((3,7))): (1,4, (co_size(6,7)))? 
  (4,5,(3,7)):(1,4,(6,7))

  */

    for (int i = 0; i < 21; i++)
  {
    auto x = layout(0,1,(i));
    cute::print(x);
    printf(", ");
  }

  printf("\n");
  auto inner_size = cute::size(layout_inner);
  auto inner_co_size = cute::cosize(layout_inner);
  auto size = cute::size(layout); 
  
  for (int x = 0; x < size; x++)
  {
    int x_0 = x % 4; 
    int x_1 = ((int)(x/4)) % 5;
    int x_2 = ((int)(x/(4*5))) % 3;
    int x_3 = ((int)(x/(4*5*3))) % 7;
    auto canonical_coord = cute::make_coord(x_0,x_1,cute::make_coord(x_2,x_3));
    int x_3d_0 = x_0;
    int x_3d_1 = x_1;
    int x_3d_2 =((int)(x/(4*5))) % inner_size;
    auto three_d_coord_zero_end = cute::make_coord(x_3d_0, x_3d_1, 0); 


    cute::print(canonical_coord);
    printf(" : L_applied  = ");
    cute::print(layout(canonical_coord));
    printf(" ---> ");
    cute::print(three_d_coord_zero_end);
    printf(" ++ ");
    cute::print(x_3d_2);
    printf(" : L_3d_ized_applied  = ");
    auto alt_out = layout(three_d_coord_zero_end); 
    auto tail = layout_inner(x_3d_2);
    auto sum = alt_out + tail; 
    cute::print(sum);
    printf("\n");
  }
  /*
  so we have the theorem that if a tuple has canonical layout (a,b,(c,d)):(p,q,(r,s)) 
  the output of the co-ordinate (x,w,(z,w)) under L is the same as the output of (x,w,0,0) + L inner(z,w) 
  but L_inner(z,w) maps to the same thing as L_inner(colex_L_inner(z,w)) 
  because applying the layout is dynamic in the sense that if you give it some 1d co-ordniate, we do colex_inv 
  then do the coord_map, however if shaped nd coord is itself given, we directly take the co-ord_map 
  
  */
  return 0;
 
}