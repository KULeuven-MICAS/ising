import random
import typing
from collections.abc import Callable, Generator
from ising.utils.convert import GraphLike, convert_to_graph
from ising.typing import Vartype
from ising.model import BinaryQuadraticModel

__all__ = ['generate_bqm', 'SubsetType']

SubsetType = typing.Literal['all', 'quadratic']

def generate_bqm(
        graph: GraphLike,
        vartype: Vartype,
        subset: SubsetType = 'all',
        bias_generator: None | Callable | Generator = None
        ) -> BinaryQuadraticModel:
    nodes, edges = convert_to_graph(graph)

    if bias_generator is None:
        f = random.random
    elif isinstance(bias_generator, Callable):
        f = bias_generator
    elif isinstance(bias_generator, Generator):
        def f():
            return next(bias_generator)
    else:
        raise ValueError(f'bias_generator is neither Callable nor a Generator: {bias_generator}')

    if subset == 'all':
        linear = { v: f() for v in nodes }
        quadratic = { e: f() for e in edges }
    elif subset == 'quadratic':
        linear = { v: 0 for v in nodes }
        quadratic = { e: f() for e in edges }
    else:
        raise ValueError(f'Subset-option {subset} unkown')

    return BinaryQuadraticModel(linear, quadratic, 0, vartype)
