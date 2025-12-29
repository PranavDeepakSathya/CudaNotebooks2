from typing import Tuple, List, Iterator,Union, Any
import itertools
import numpy as np
import math
from enum import Enum, auto
from .biject import Bijection

class ExecutionType(Enum):
    PARALLEL = auto() 
    SERIAL = auto()   


class GemmTopology:
  def __init__(self, 
                shape: Tuple[int, int, int], 
                tiler: np.ndarray, 
                sigma: Tuple[int, int, int], 
                perm: 'Bijection'): 
      
    self.tiler = np.asarray(tiler, dtype=int)
    
    assert self.tiler.ndim == 2 and self.tiler.shape[0] == 3 
    assert np.all(np.sum(self.tiler, axis=1) == np.array(shape))
    assert np.array_equal(np.sort(np.array(sigma)), np.array([0, 1, 2]))

    self.shape = shape 
    self.perm = perm 
    
    self.q = self.tiler.shape[1] 
    self.N_tiles = self.q**3
    self.tile_tensor_shape = (self.q, self.q, self.q)
    
    zeros_col = np.zeros((3, 1), dtype=int)
    self.cuts = np.cumsum(np.hstack([zeros_col, self.tiler]), axis=1)
    
    base_strides = (1, self.q, self.q**2)
    self.strides = (base_strides[sigma[0]], 
                    base_strides[sigma[1]], 
                    base_strides[sigma[2]])
  
  def _resolve_coords(self, x: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if isinstance(x, (int, np.integer)):
        assert 0 <= x < self.N_tiles
        return (x % self.q, (x // self.q) % self.q, x // (self.q**2))
    else:
        assert len(x) == 3
        return x
    
  def get_tile_id(self, x: Tuple[int,int,int]): 
    assert 0 <= x[0] < self.q and 0 <= x[1] < self.q and 0 <= x[2] < self.q
    return x[0] + (self.q * x[1]) + (self.q * self.q * x[2])

  def get_tile_shape(self, x: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    m, k, n = self._resolve_coords(x)
    return (self.tiler[0, m], self.tiler[1, k], self.tiler[2, n])

  def get_tile_slice(self, x: Union[int, Tuple[int, int, int]]) -> Tuple[slice, slice, slice]:
    m, k, n = self._resolve_coords(x)
    return (slice(self.cuts[0, m], self.cuts[0, m+1]), 
            slice(self.cuts[1, k], self.cuts[1, k+1]), 
            slice(self.cuts[2, n], self.cuts[2, n+1]))

  def get_rank(self, x: Union[int, Tuple[int, int, int]]) -> int:
    c = self._resolve_coords(x)
    linear_rank = (c[0] * self.strides[0] + 
                    c[1] * self.strides[1] + 
                    c[2] * self.strides[2])
    return self.perm(linear_rank)

  def __repr__(self):
    return f"<Topology {self.shape} | q={self.q} | σ={self.strides}>"
    
    



class GemmNode: 
  def __init__(self, 
                topology: 'GemmTopology', 
                execution_type: ExecutionType = ExecutionType.PARALLEL): 
      
    self.topology = topology
    self.execution_type = execution_type
    
    self.children = {i: None for i in range(self.topology.N_tiles)}
    self.parent = None
    self.parent_tile_id = None 

  def set_child(self, tile_id: int, subtopology: 'GemmTopology', sub_exec_type: ExecutionType = ExecutionType.PARALLEL):
    if tile_id not in self.children:
      raise IndexError(f"Tile ID {tile_id} is out of bounds for parent topology with {self.topology.N_tiles} tiles.")

    parent_tile_shape = self.topology.get_tile_shape(tile_id)
    child_total_shape = subtopology.shape

    if np.any(np.array(parent_tile_shape) != np.array(child_total_shape)):
      raise ValueError(f"Shape Mismatch at Tile {tile_id}!\n"
                        f"Parent Tile Shape: {parent_tile_shape}\n"
                        f"Child Total Shape: {child_total_shape}\n"
                        f"A strategy must perfectly fill the tile it occupies.")

    self.children[tile_id] = GemmNode(subtopology, sub_exec_type)
    self.children[tile_id].parent = GemmNode(self.topology)
    self.children[tile_id].parent_tile_id = tile_id

  def __repr__(self):
    defined_children = sum(1 for c in self.children.values() if c is not None)
    return f"<GemmNode ({self.execution_type.name}) | {self.topology} | Defined Children: {defined_children}/{self.topology.N_tiles}>"
  
  def get_owned_slice_rel_parent(self):
    if self.parent == None: 
      M,K,N = self.topology.shape 
      return (slice(0,M), slice(0,K), slice(0,N)) 
    else: 
      return self.parent.topology.get_tile_slice(self.parent_tile_id)
    
  def print_layout_table(self):

    print(f"\nNode Layout: {self.execution_type.name} | Topology: {self.topology.shape}")
    print("-" * 85)
    print(f"{'ID':<4} | {'Rank':<4} | {'Slice (M, K, N)':<30} | {'Topology':<20} | {'Status'}")
    print("-" * 85)

    for tile_id, child in sorted(self.children.items()):
      if child is None:
        continue
      
      rank = self.topology.get_rank(tile_id)
      slc = self.topology.get_tile_slice(tile_id)
      def fmt_s(s): return f"{s.start}:{s.stop}"
      slice_str = f"[{fmt_s(slc[0])}, {fmt_s(slc[1])}, {fmt_s(slc[2])}]"

      child_top = f"Top{child.topology.shape}"
      defined = sum(1 for c in child.children.values() if c is not None)
      status = f"{defined}/{child.topology.N_tiles} children"

      print(f"{tile_id:<4} | {rank:<4} | {slice_str:<30} | {child_top:<20} | {status}")
    print("-" * 85 + "\n")