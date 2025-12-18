from cute_viz import display_tv_layout
import torch
from functools import partial
from typing import List

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute import KeepPTX, KeepCUBIN
import cupy as cp
from cutlass.torch import dtype as torch_dtype
import cutlass.cute.runtime as cute_rt

N_elems = 50 
a = torch.arange(N_elems, dtype=torch_dtype(cutlass.Float32))
a_ptr = cute_rt.make_ptr(cutlass.Float32, a.data_ptr())

@cute.jit
def slicer(ptr:cute.Pointer): 
  layout = cute.make_layout((5,(10,)), stride=(1,(5,)))
  tensor = cute.make_tensor(ptr, layout)
  cute.print_tensor(tensor[(None,(0))])
  
  
slicer(a_ptr)