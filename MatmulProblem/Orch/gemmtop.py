from typing import Tuple, List, Iterator,Union, Any
import itertools
import numpy as np
import math
from enum import Enum, auto
from .biject import Bijection

class ExecutionType(Enum):
    PARALLEL = auto() # e.g., Grid tiling (Spatial)
    SERIAL = auto()   # e.g., K-loop or Producer-Consumer (Temporal)



class GemmTopology: 
  """
  The Mathematical Definition (S, T, σ, γ).
  Defines the Geometry (Shape/Tiling) and Topology (Order/Rank) of one level.
  """
  def __init__(self, 
               shape: Tuple[int, int, int], 
               tiler: np.ndarray, 
               sigma: Tuple[int, int, int], 
               perm: 'Bijection'): 
    
    # --- 1. Validation (The "S" and "T") ---
    self.tiler = np.asarray(tiler, dtype=int)
    assert self.tiler.ndim == 2 and self.tiler.shape[0] == 3 
    
    # Verify partitions sum to total shape
    assert np.all(np.sum(self.tiler, axis=1) == np.array(shape))
    
    # Verify Sigma is a valid permutation of (0,1,2)
    assert np.array_equal(np.sort(np.array(sigma)), np.array([0, 1, 2]))

    self.shape = shape 
    self.perm = perm # The Gamma Bijection (Swizzle)
    
    # --- 2. Geometry Setup (The Tiler Tensor) ---
    self.q = self.tiler.shape[1] # The 'q' split factor
    self.N_tiles = self.q**3
    self.tile_tensor_shape = (self.q, self.q, self.q)
    # Pre-calculate Cuts (Prefix Sums) for O(1) Slice lookups
    # hstack with 0 column to get range starts [0, s1, s1+s2, ...]
    zeros_col = np.zeros((3, 1), dtype=int)
    self.cuts = np.cumsum(np.hstack([zeros_col, self.tiler]), axis=1)
    
    # --- 3. Topology Setup (The "σ" and "γ") ---
    # Base Colex strides: (1, q, q^2)
    base_strides = (1, self.q, self.q**2)
    
    # Permute strides according to sigma (The Layout Permutation)
    self.strides = (base_strides[sigma[0]], 
                    base_strides[sigma[1]], 
                    base_strides[sigma[2]])
  
  # --- Helper: Polymorphic Coordinate Resolver ---
  def _resolve_coords(self, x: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Implements colex^{-1} if input is an int."""
    if isinstance(x, (int, np.integer)):
        assert 0 <= x < self.N_tiles, f"Tile ID {x} out of bounds"
        return (x % self.q, (x // self.q) % self.q, x // (self.q**2))
    else:
        assert len(x) == 3
        return x
      
  def get_tile_id(self, x: Tuple[int,int,int]): 
    assert 0 <= x[0] and 0 <= x[1] and 0 <= x[2] 
    assert x[0] < self.q and x[1] < self.q and x[2] < self.q
    return x[0] + (self.q*x[1]) + (self.q*self.q*x[2])

  # --- Map 1: The Tile-Shape Map (shp_T) ---
  def get_tile_shape(self, x: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    m, k, n = self._resolve_coords(x)
    return (self.tiler[0, m], self.tiler[1, k], self.tiler[2, n])

  # --- Map 2: The Tile-Slice Map (slc_T) ---
  def get_tile_slice(self, x: Union[int, Tuple[int, int, int]]) -> Tuple[slice, slice, slice]:
    m, k, n = self._resolve_coords(x)
    ms = slice(self.cuts[0, m], self.cuts[0, m+1])
    ks = slice(self.cuts[1, k], self.cuts[1, k+1])
    ns = slice(self.cuts[2, n], self.cuts[2, n+1])
    return (ms, ks, ns)

  # --- Map 3: The Rank Map (R) ---
  def get_rank(self, x: Union[int, Tuple[int, int, int]]) -> int:
    """R: (x,y,z) -> gamma( phi_sigma(x,y,z) )"""
    c = self._resolve_coords(x)
    # 1. Apply Strides (Layout)
    linear_rank = (c[0]*self.strides[0] + c[1]*self.strides[1] + c[2]*self.strides[2])
    # 2. Apply Swizzle (Bijection)
    return self.perm(linear_rank)

  def __repr__(self):
      return f"<Topology {self.shape} | q={self.q} | σ={self.strides}>"
    
    



class GemmNode: 
  """
  A recursive node in the orchestration hierarchy.
  Links the current level's mathematical topology to the next level's nodes.
  """
  def __init__(self, 
               topology: 'GemmTopology', 
               execution_type: ExecutionType = ExecutionType.PARALLEL): 
    
    self.topology = topology
    self.execution_type = execution_type
    
    # Initialize map: Tile_ID -> None (Placeholder for future strategies)
    # The domain is strictly defined by the Topology's tiling (0 to q^3 - 1)
    self.children = {i: None for i in range(self.topology.N_tiles)}

  def set_child(self, tile_id: int, subtopology: 'GemmTopology', sub_exec_type: ExecutionType = ExecutionType.PARALLEL):
    """
    Attaches a strategy (Sub-Topology) to a specific tile of the current level.
    """
    # 1. Validate Range (Is this tile ID valid for me?)
    if tile_id not in self.children:
        raise IndexError(f"Tile ID {tile_id} is out of bounds for parent topology with {self.topology.N_tiles} tiles.")

    # 2. Validate Shape Consistency (The Physics Check)
    # The shape of the specific tile in Parent must match the Total Shape of the Child
    parent_tile_shape = self.topology.get_tile_shape(tile_id)
    child_total_shape = subtopology.shape

    if np.any(np.array(parent_tile_shape) != np.array(child_total_shape)):
        raise ValueError(f"Shape Mismatch at Tile {tile_id}!\n"
                         f"Parent Tile Shape: {parent_tile_shape}\n"
                         f"Child Total Shape: {child_total_shape}\n"
                         f"A strategy must perfectly fill the tile it occupies.")

    # 3. Create and Link the Node
    # We wrap the topology in a Node to allow further recursion down the line
    self.children[tile_id] = GemmNode(subtopology, sub_exec_type)

  def __repr__(self):
      # Count how many children are actually defined (not None)
      defined_children = sum(1 for c in self.children.values() if c is not None)
      return f"<GemmNode ({self.execution_type.name}) | {self.topology} | Defined Children: {defined_children}/{self.topology.N_tiles}>"

