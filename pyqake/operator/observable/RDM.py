from pyqake.operator.Source import creation_annihilations, Pauli_Decompose
from qiskit.quantum_info import SparsePauliOp
from functools import reduce
import numpy as np
import itertools
import math


def generate(N_orb, mapping='jordan_wigner', Degree=1):
    #  [Input]   N_orb : Number of spatial orbitals
    #  [Input] mapping : Mapping algorithm from fermionic into qubit
    #  [Input]  Degree : {Degree}-Body RDM will be created
    # [Output]  Ops_Re : List of real part of RDM component operators. Operators are listed to use numpy.reshape easily
    # [Output]  Ops_Im : List of imaginary part of RDM component operators. Operators are listed to use numpy.reshape easily
    # [Output]   binom : List of binomial coefficient generated for efficient reshaping
    # Initialize
    C, D    = creation_annihilations(2 * N_orb, mapping = mapping)
    orb_lst = list(range(N_orb))
    ind_lst = list(range(Degree))
    Ops_Re  = []
    Ops_Im  = []
    binom   = [math.comb(Degree, i) for i in range(Degree + 1)]

    # Generate RDM Operators
    for N_RDM_beta in range(Degree + 1):
        for beta_case in itertools.combinations(ind_lst, N_RDM_beta):
            spin_shifter    = np.zeros(2 * Degree)
            for beta_ind in beta_case:
                spin_shifter[2 * beta_ind]     = N_orb
                spin_shifter[2 * beta_ind + 1] = N_orb
            for rdm_case in itertools.product(orb_lst, repeat = 2 * Degree):
                rdm_case_array  = np.array(rdm_case)
                Op_Re, Op_Im    = Pauli_Decompose(_generate(C, D, rdm_case_array + spin_shifter, Degree))
                Ops_Re.append(Op_Re)
                Ops_Im.append(Op_Im)
    return Ops_Re, Ops_Im, binom


def _generate(C, D, Indices, Degree):
    reordered_indices   = [int(a) for a in Indices[0::2]] + [int(b) for b in Indices[1::2]]
    def mult_ops(A, B):
        return A @ B
    T_before_product    = [C[a] for a in reordered_indices[:Degree]] + [D[b] for b in reversed(reordered_indices[Degree:])]
    T                   = reduce(mult_ops, T_before_product)
    return SparsePauliOp.simplify(T)