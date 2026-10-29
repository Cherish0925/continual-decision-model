"""Distinct integer proportions and their lowest-material representatives."""
from functools import reduce
from math import gcd
from experiments.discoveryworld.collect_chemistry import integer_recipes


def primitive_recipes(component_count,max_parts=6,*,include_leave=False):
    """Enumerate unique nonempty ratios; optionally append the zero door action."""
    result=[counts for counts in integer_recipes(component_count,max_parts) if reduce(gcd,counts)==1]
    if include_leave:result.append((0,)*component_count)
    return result
