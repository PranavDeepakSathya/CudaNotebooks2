from typing import Tuple, List, Iterator, Union, Any
import itertools
import numpy as np
import math
from enum import Enum, auto


class Bijection: 
  def __init__(self, length, bits, base, shift, scale, step): 

    assert math.gcd(scale, length) == 1, "Scale must be coprime to length for bijectivity"
    assert shift >= 0 
    
    self.length = length
    self.scale = scale
    self.step = step
    self.shift = shift
    self.mask = ((1 << bits) - 1) << base

    if bits > 0:
        is_power_of_two = (length > 0) and ((length & (length - 1)) == 0)
        if not is_power_of_two:
            raise ValueError(f"CRITICAL: Cannot apply bit-swizzle on non-power-of-2 length {length}. Collision imminent.")

  def __call__(self, x):
    swizzled = x ^ ((x >> self.shift) & self.mask)
    return (self.scale * (swizzled + self.step)) % self.length