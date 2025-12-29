from typing import Tuple, List, Iterator, Union, Any
import itertools
import numpy as np
import math
from enum import Enum, auto
from .biject import Bijection
from .gemmtop import ExecutionType, GemmNode, GemmTopology

class SliceIR_node: 
  def __init__(self, rank, shape, slice, depth_id, global_offset, execution_policy):
    self.rank = rank  
    self.shape = shape 
    self.slice = slice 
    self.depth_id = depth_id 
    self.global_offset = global_offset
    self.execution_policy = execution_policy
    self.children = [] 
    self.parent = None

  def _fmt_slice(self, s):
    """Helper: converts slice(0, 4, None) -> '0:4'"""
    if s is None: return ":"
    start = s.start if s.start is not None else ""
    stop = s.stop if s.stop is not None else ""
    step = f":{s.step}" if s.step is not None else ""
    return f"{start}:{stop}{step}"

  def __repr__(self):
    slice_str = "[" + ", ".join(self._fmt_slice(s) for s in self.slice) + "]"
    S = (int(self.shape[0]), int(self.shape[1]), int(self.shape[2]))
    # Added Rank to the repr for clarity
    return (f"<IR_Node d={self.depth_id} | r={self.rank} | {self.execution_policy.name} | "
            f"shape={S} | slice={slice_str} | Children={len(self.children)}>")

class SliceIR_tree: 
  def __init__(self, root_gemm_node: GemmNode): 
    root_shape = root_gemm_node.topology.shape 
    try:
      root_slice = root_gemm_node.get_owned_slice_rel_parent()
    except AttributeError:
      root_slice = tuple(slice(0, s) for s in root_shape)

    # Initial global offset is (0,0,0) for the root
    self.root = self._build_recursive(
      node=root_gemm_node, 
      rank=0,
      shape=root_shape, 
      slc=root_slice, 
      global_offset=(0, 0, 0),
      depth=0
    )

  def _build_recursive(self, node: GemmNode, rank, shape, slc, global_offset, depth) -> SliceIR_node:
    ir_node = SliceIR_node(rank, shape, slc, depth, global_offset, node.execution_type)
    
    temp_children = []
    n_tiles = node.topology.N_tiles 
    
    for c in range(n_tiles): 
      child_shape = node.topology.get_tile_shape(c)
      child_slice = node.topology.get_tile_slice(c)
      child_rank = node.topology.get_rank(c) # Get the execution rank
      
      # Calculate absolute offset: Parent Global + Child Relative Start
      child_offset = (
        global_offset[0] + (child_slice[0].start or 0),
        global_offset[1] + (child_slice[1].start or 0),
        global_offset[2] + (child_slice[2].start or 0)
      )

      if 0 in child_shape: 
        assert node.children[c] is None, \
          f"Tile {c} has 0-volume {child_shape} but has an attached strategy."
        continue 

      child_gemm_node = node.children[c]

      if child_gemm_node is not None:
        child_ir = self._build_recursive(
          node=child_gemm_node, 
          rank=child_rank, # Pass rank down
          shape=child_shape, 
          slc=child_slice, 
          global_offset=child_offset, # Pass offset down
          depth=depth + 1
        )
        child_ir.parent = ir_node
        temp_children.append(child_ir)

    temp_children.sort(key=lambda x: x.rank)
    ir_node.children = temp_children

    return ir_node
  
def generate_dot_source(ir_tree) -> str:
  """
  Generates a Graphviz DOT string from a SliceIR_tree.
  """
  lines = [
      "digraph SliceIR {",
      "    rankdir=TB;",                # Top-to-Bottom layout
      "    splines=polyline;",          # Straighter lines for a technical look
      "    node [shape=record, fontname=\"Courier\", style=filled, fillcolor=\"#f8f9fa\"];",
      "    edge [fontname=\"Courier\"];"
  ]

  def _escape(s):
      """Escape special characters for DOT labels"""
      return str(s).replace("{", "\\{").replace("}", "\\}").replace("<", "\\<").replace(">", "\\>")

  def visit(node):
      node_id = f"node_{id(node)}"
      
      # Color coding logic (Optional: Customize based on ExecutionPolicy)
      # Global = Blue-ish, Thread = Green-ish, Block = Orange-ish?
      # For now, we stick to clean neutral formatting.
      
      # Formatting the 'slice' string nicely using the class's own logic if available,
      # or a raw repr if not.
      slice_repr = _escape("[" + ", ".join(node._fmt_slice(s) for s in node.slice) + "]")
      shape_repr = _escape((int(node.shape[0]),int(node.shape[1]),int(node.shape[2])))
      global_offset_repr = _escape((int(node.global_offset[0]), int(node.global_offset[1]), int(node.global_offset[2])))
      # The Record Label:
      # { Rank: 0 | Depth: 0 | ExecutionPolicy }
      # { Shape: (M,N,K) | Slice: [0:4, :] | Offset: (0,0,0) }
      label = (
          f"{{ Rank:{node.rank} | Depth:{node.depth_id} | child_exec:{node.execution_policy.name} }} | "
          f"{{ Shape:{shape_repr} | Slice:{slice_repr} | Offset:{global_offset_repr} }}"
      )
      
      lines.append(f'    {node_id} [label="{label}"];')
      
      # Draw edges to children
      for child in node.children:
          child_id = f"node_{id(child)}"
          lines.append(f'    {node_id} -> {child_id};')
          visit(child)

  if ir_tree.root:
      visit(ir_tree.root)
  
  lines.append("}")
  return "\n".join(lines)

# --- Usage Example ---
