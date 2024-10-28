from __future__ import annotations
from collections.abc import Mapping, Collection, Sequence
from pathlib import Path
import numpy as np
from bqm.model.typing import Variable, Bias, Vartype

# Note: We can build a class upon this, which enforces the use of n-bit weights.
class BinaryQuadraticModel(object):
    """Encodes a binary quadratic model.

    Attributes:
        linear (dict[Variable, Bias]):
            A dictionary mapping variables in the BQM to its linear bias. 
            Biasless variables have bias zero.

        quadratic (dict[frozenset[Variable, Variable], Bias]):
            A dictionary mapping unordered collections of 2 variables (edges/pairs in
            the BQM) to their quadratic biases. Omitted edges are implicitly zero.
            Every variable present here should have an entry in self.linear.

        offset (Bias): 
            Offset of the BQM. This constant becomes meaningless in optimization solving
            but is a necessary part for proper QUBO-Ising mapping.

        vartype (Vartype):
            Encoding type of the BQM, either Vartype.SPIN or Vartype.BINARY.

    """

    def __init__(
            self,
            linear: Mapping[Variable, Bias],
            quadratic: Mapping[Collection[Variable, Variable], Bias],
            offset: Bias,
            vartype: Vartype
            ):
        self.linear: dict[Variable, Bias] = {}
        self.quadratic: dict[frozenset[Variable, Variable], Bias] = {}
        self.offset: Bias = offset
        self.vartype: Vartype = vartype

        self.add_variables_from(linear)
        self.add_interactions_from(quadratic)

    def __repr__(self) -> str:
        return f'BinaryQuadraticModel({self.linear}, {self.quadratic}, {self.offset}, {self.vartype})'

    def __len__(self) -> int:
        return self.num_variables

    @property
    def num_variables(self) -> int:
        """The number of variables in the BQM."""
        return len(self.linear)

    @property
    def num_interactions(self) -> int:
        """The number of nonzero interactions in the BQM."""
        return len(self.quadratic)

    def set_offset(self, offset: Bias) -> None:
        """Set the offset of the BQM."""
        self.offset = offset

    def set_variable(self, v: Variable, bias: Bias, vartype: Vartype|None = None) -> None:
        """Set a variable of the BQM.
        If the variable already exists, overwrite its bias, else, create a new variable.
        """
        if vartype is not None and vartype is not self.vartype:
            if vartype is Vartype.SPIN and self.vartype is Vartype.BINARY:
                bias *= -2
            elif vartype is Vartype.BINARY and self.vartype is Vartype.SPIN:
                bias *= -1/2
            else:
                raise ValueError(f'Vartype unknown: {Vartype}')
        self.linear[v] = bias

    def set_variables_from(self, linear: Mapping[Variable, Bias], vartype: Vartype|None = None):
        """Set variables of the BQM."""
        for v, bias in linear.items():
            self.set_variable(v, bias, vartype)

    def set_interaction(self, u: Variable, v: Variable, bias: Bias, vartype: Vartype|None = None):
        """Set an interaction of the BQM.
        Set/Overwrite the coupling term between variables u and v of the BQM.
        If variables u and/or v do not exists, create them first (with linear bias 0).
        """
        if u == v:
            raise ValueError(f'Self-coupling interactions such as ({u},{v}) are not allowed')
        for var in (u, v):
            if var not in self.linear:
                self.linear[var] = 0
        if vartype is not None and vartype is not self.vartype:
            if vartype is Vartype.SPIN and self.vartype is Vartype.BINARY:
                self.linear[u] += 2 * bias
                self.linear[v] += 2 * bias
                self.offset += -bias
                bias *= -4
            elif vartype is Vartype.BINARY and self.vartype is Vartype.SPIN:
                self.linear[u] += -1/4 * bias
                self.linear[v] += -1/4 * bias
                self.offset += 1/4 * bias
                bias *= -1/4
            else:
                raise ValueError(f'Vartype unknown: {Vartype}')
        self.quadratic[frozenset(u, v)] = bias

    def set_interactions_from(self, quadratic: Mapping[Collection[Variable, Variable], Bias], vartype: Vartype|None = None):
        """Set interactions of the BQM."""
        for (u, v), bias in quadratic.items():
            self.set_interaction(u, v, bias, vartype)

    def remove_variable(self, v: Variable):
        if v not in self.linear:
            return
        del self.linear[v]
        for e in self.quadratic:
            if v in e:
                del self.quadratic[e]
        
    def remove_interaction(self, e: Collection[Variable, Variable]):
        e = frozenset(e)
        if e in self.quadratic:
            del self.quadratic[e]

    def scale(self, scalar: Bias):
        for v in self.linear:
            self.linear[v] *= scalar
        for e in self.quadratic:
            self.quadratic[e] *= scalar

    def change_vartype(self, vartype: Vartype):
        if vartype is self.vartype:
            return
        if vartype is Vartype.SPIN and self.vartype is Vartype.BINARY:
            self.linear, self.quadratic, self.offset = self.binary_to_spin(self.linear, self.quadratic, self.offset)
        elif vartype is Vartype.BINARY and self.vartype is Vartype.SPIN:
            self.linear, self.quadratic, self.offset = self.spin_to_binary(self.linear, self.quadratic, self.offset)
        else:
            raise ValueError(f'Vartype unknown: {Vartype}')

    @staticmethod
    def spin_to_binary(linear: dict[Variable, Bias], quadratic: dict[frozenset[Variable, Variable], Bias], offset: Bias):
        linear = { v : 1/4 * sum([ bias for (e, bias) in quadratic.items() if v in e ]) - 1/2 * bias for (v, bias) in linear.items() }
        quadratic = { e : -1/4 * bias for (e, bias) in quadratic.items() }
        offset = 1/4 * sum(quadratic.values()) + 1/2 * sum(linear.values())
        return linear, quadratic, offset

    @staticmethod
    def binary_to_spin(linear: dict[Variable, Bias], quadratic: dict[frozenset[Variable, Variable], Bias], offset: Bias):
        linear = { v : 2 * sum([ bias for (e, bias) in quadratic.items() if v in e ]) - 2 * bias for (v, bias) in linear.items() }
        quadratic = { e : -4 * bias for (e, bias) in quadratic.items() }
        offset = - sum(quadratic.values()) + sum(linear.values())
        return linear, quadratic, offset

    def eval(self, sample: Mapping[Variable, bool]) -> Bias:
        if self.vartype is Vartype.SPIN:
            t = { True: -1, False: 1 }
            q_sign, l_sign = -1, -1
        if self.vartype is Vartype.BINARY:
            t = { True: 1, False: 0 }
            q_sign, l_sign = 1, 1
        else:
            raise ValueError(f'Vartype unknown: {Vartype}')
        quadratic = sum([ bias * t[sample[v]] * t[sample[u]] for (v, u), bias in self.quadratic.items() ])
        linear = sum([ bias * t[sample[v]] for (v, bias) in self.linear.items() ])
        return q_sign * quadratic + l_sign * linear + self.offset

    def copy(self) -> BinaryQuadraticModel:
        return BinaryQuadraticModel(self.linear, self.quadratic, self.offset, self.vartype)

    def to_qubo(self, variable_order: Sequence[Variable]|None = None) -> tuple[np.ndarray, Bias]:
        self.change_vartype(Vartype.BINARY)
        Q = np.zeros((self.num_variables)*2, dtype=float)
        if variable_order is None:
            idx = { v : i for i, v in enumerate(self.linear) } # essentially random order
        else:
            idx = { v : i for i, v in enumerate(variable_order) }
        try:
            for v, bias in self.linear:
                Q[idx[v], idx[v]] = bias
            for (u, v), bias in self.quadratic:
                iu, iv = idx[u], idx[v]
                if iu < iv:
                    Q[iu, iv] = bias
                else:
                    Q[iv, iu] = bias
        except KeyError:
            raise ValueError(f'variable {v} missing from variable_order')
        #Q[np.tril_indices_from(Q, k=-1)] = np.triu(Q, k=1)
        return Q, self.offset

    @classmethod
    def from_qubo(cls, Q: np.ndarray, offset: Bias = 0.0, variable_order: Sequence[Variable]|None = None) -> BinaryQuadraticModel:
        if Q.ndim != 2:
            raise ValueError()
        if Q.shape[0] != Q.shape[1]:
            raise ValueError()
        if Q.shape[0] < 1:
            raise ValueError()
        if variable_order is None:
            variable_order = list(range(Q.shape[0]))
        try:
            bqm = cls({}, {}, offset, Vartype.BINARY)
            it = np.nditer(Q, flags=['multi_index'])
            for bias in it:
                row, col = it.multi_index
                if row == col:
                    bqm.set_variable(variable_order[row], bias)
                elif row > col:
                    bqm.set_interaction(variable_order[row], variable_order[col], bias)
                else:
                    continue
        except IndexError:
            raise ValueError()
        return bqm

    def to_ising(self, variable_order: Sequence[Variable]|None = None) -> tuple[np.ndarray, np.ndarray, Bias]:
        self.change_vartype(Vartype.SPIN)
        h = np.zeros((self.num_variables), dtype=float)
        J = np.zeros((self.num_variables)*2, dtype=float)
        if variable_order is None:
            idx = { v : i for i, v in enumerate(self.linear) } # essentially random order
        else:
            idx = { v : i for i, v in enumerate(variable_order) }
        try:
            for v, bias in self.linear:
                h[idx[v]] = bias
            for (u, v), bias in self.quadratic:
                iu, iv = idx[u], idx[v]
                if iu < iv:
                    J[iu, iv] = bias
                else:
                    J[iv, iu] = bias
        except KeyError:
            raise ValueError(f'variable {v} missing from variable_order')
        return h

    @classmethod
    def from_ising(cls, h: np.ndarray, J: np.ndarray, offset: Bias = 0.0, variable_order: Sequence[Variable]|None = None) -> BinaryQuadraticModel:
        if h.ndim != 1:
            raise ValueError()
        if h.size < 1:
            raise ValueError()
        if J.ndim != 2:
            raise ValueError()
        if J.shape[0] != J.shape[1]:
            raise ValueError()
        if J.shape[0] != h.size:
            raise ValueError()
        if variable_order is None:
            variable_order = list(range(h.size))
        try:
            bqm = cls({}, {}, offset, Vartype.SPIN)
            for i, bias in np.ndenumerate(h):
                    bqm.set_variable(variable_order[i], bias)
            it = np.nditer(J, flags=['multi_index'])
            for bias in it:
                i, j = it.multi_index
                if i > j:
                    bqm.set_interaction(variable_order[i], variable_order[j], bias)
                else:
                    continue
        except IndexError:
            raise ValueError()
        return bqm

    def to_file(self, file: Path):
        raise NotImplementedError()

    @classmethod
    def from_file(cls, file: Path):
        raise NotImplementedError()
