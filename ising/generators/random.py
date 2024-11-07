import random
from ising.utils.convert import GraphLike
from ising.typing import Vartype, Bias
from ising.generators.generator import generate_bqm, SubsetType
from ising.model import BinaryQuadraticModel

__all__ = ['uniform', 'randint']

def uniform(graph: GraphLike, vartype: Vartype, subset: SubsetType = 'all', low: Bias = 0, high: Bias = 1, seed = None) -> BinaryQuadraticModel:
    random.seed(seed)
    def gen():
        while True:
            yield random.uniform(low, high)
    return generate_bqm(graph, vartype, subset, gen())

def randint(graph: GraphLike, vartype: Vartype, subset: SubsetType = 'all', low: int = 0, high: int = 1, seed = None) -> BinaryQuadraticModel:
    random.seed(seed)
    def gen():
        while True:
            yield random.randint(low, high)
    return generate_bqm(graph, vartype, subset, gen())
