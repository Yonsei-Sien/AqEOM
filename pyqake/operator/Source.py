from qiskit.quantum_info import SparsePauliOp
import numpy as np


def identity(n):
    #  [Input] n : Number of qubit
    # [Output]   : n-qubit identity operator
    return SparsePauliOp.from_list([("I" * n, 1)])


def creation_annihilation(n, p, mapping="jordan_wigner"): 
    #  [Input]       n : Number of qubit
    #  [Input]       p : Index of qubit for creation && annihilation operator
    #  [Input] mapping : Mapping algorithm from fermionic into qubit
    # [Output]         : pth creation and annihilation operator for n-qubit space
    C = None
    if mapping == "jordan_wigner":
        if p == 0:
            l, r = "I" * (n - 1), ""
        elif p == n - 1:
            l, r = "", "Z" * (n - 1)
        else:
            l, r = "I" * (n - p - 1), "Z" * p
        C = SparsePauliOp.from_list([(l + "X" + r, 0.5), (l + "Y" + r, -0.5j)])
    elif mapping == "parity":
        if p == 0:
            l, r = "X" * (n - 1), ""
            C = SparsePauliOp.from_list([(l + "X" + r, 0.5), (l + "Y" + r, -0.5j)])
        elif p == 1:
            l, r = "X" * (n - 2), ""
            C = SparsePauliOp.from_list([(l + "XZ" + r, 0.5), (l + "YI" + r, -0.5j)])
        elif p == n - 1:
            l, r = "", "I" * (n - 2)
            C = SparsePauliOp.from_list([(l + "XZ" + r, 0.5), (l + "YI" + r, -0.5j)])
        else:
            l, r = "X" * (n - p - 1), "I" * (p - 1)
            C = SparsePauliOp.from_list([(l + "XZ" + r, 0.5), (l + "YI" + r, -0.5j)])
    #elif mapping == "bravyi_kitaev":

    else:
        raise ValueError("Unsupported mapping.")
    return C, C.conjugate().transpose()


def creation_annihilations(n, mapping="jordan_wigner"): 
    #  [Input]       n : Number of qubit
    #  [Input] mapping : Mapping algorithm from fermionic into qubit
    # [Output]         : List of creation and annihilation operators for n-qubit space
    Cs = []
    if mapping == "jordan_wigner":
        for p in range(n):
            if p == 0:
                l, r = "I" * (n - 1), ""
            elif p == n - 1:
                l, r = "", "Z" * (n - 1)
            else:
                l, r = "I" * (n - p - 1), "Z" * p
            C = SparsePauliOp.from_list([(l + "X" + r, 0.5), (l + "Y" + r, -0.5j)])
            Cs.append(C)
    elif mapping == "parity":
        for p in range(n):
            if p == 0:
                l, r = "X" * (n - 1), ""
                C = SparsePauliOp.from_list([(l + "X" + r, 0.5), (l + "Y" + r, -0.5j)])
                Cs.append(C)
            elif p == 1:
                l, r = "X" * (n - 2), ""
                C = SparsePauliOp.from_list([(l + "XZ" + r, 0.5), (l + "YI" + r, -0.5j)])
                Cs.append(C)
            elif p == n - 1:
                l, r = "", "I" * (n - 2)
                C = SparsePauliOp.from_list([(l + "XZ" + r, 0.5), (l + "YI" + r, -0.5j)])
                Cs.append(C)
            else:
                l, r = "X" * (n - p - 1), "I" * (p - 1)
                C = SparsePauliOp.from_list([(l + "XZ" + r, 0.5), (l + "YI" + r, -0.5j)])
                Cs.append(C)
    #elif mapping == "bravyi_kitaev":
    else:
        raise ValueError("Unsupported mapping.")
    As = [C.conjugate().transpose() for C in Cs]
    return Cs, As


########################<<< Commutators
def get_commutator(A, B, fermionic=False): 
    #  [Input] A : Operator
    #  [Input] B : Operator
    # [Output]   : [A, B]
    com = A@B + B@A if fermionic else A@B - B@A
    com = com.simplify()
    return com


def get_commutators(A, Bs, is_A_list=False, fermionic=False): 
    #  [Input]        A  : Operators
    #  [Input]        Bs : List of Operators
    #  [Input] is_A_list : Whether A is a single operator or list of operatos
    # [Output]           : [A, B0], [A, B1], ..., [A, Bn]
    if is_A_list:
        commutators = []
        for aa in A:
            commutators += [get_commutator(aa, B, fermionic) for B in Bs]
        return commutators

    else:
        commutators = [get_commutator(A, B, fermionic) for B in Bs]
        return commutators


def get_double_commutator(A, B, C): 
    #  [Input] A : Operator
    #  [Input] B : Operator
    #  [Input] C : Operator
    # [Output]   : [A, B, C] = 0.5([[A, B], C] + [A, [B, C]])
    com_AB  = get_commutator(A, B)
    com_BC  = get_commutator(B, C)
    com     = 0.5 * (get_commutator(com_AB, C) + get_commutator(A, com_BC))
    com     = com.simplify()
    return com


def get_double_commutators(As, B, Cs): 
    #  [Input] As : List of Operators
    #  [Input]  B : Operator
    #  [Input] Cs : List of Operators
    # [Output]    : [A0, B, C0], [A0, B, C1], ..., [A0, B, Cn], [A1, B, C0], ..., [An, B, Cn]
    commutators = []
    for A in As:
        commutators += [get_double_commutator(A, B, C) for C in Cs]
    return commutators


########################<<< Pauli Operator Decomposer into Re (=Direct Hermitian) Op.s and Im (=Op.s w/ Im. Coef.s) Op.s
def Pauli_Decompose(Pauli_op):
    #  [Input] Pauli_op : Series of Pauli operators with mixed operators w/ Re. and Im. coeff.s
    # [Output] Pauli_Re : Series of Pauli operators which its expectation value can be directly used
    # [Output] Pauli_Im : Series of Pauli operators which its expectation value can be used after multiplied 1.0j
    # <Pauli_op> = <Pauli_Re> + 1.0j * <Pauli_Im>
    Pauli_Re =  0.5  * (Pauli_op + Pauli_op.conjugate().transpose())
    Pauli_Im = -0.5j * (Pauli_op - Pauli_op.conjugate().transpose())
    return Pauli_Re.simplify(), Pauli_Im.simplify()


def Pauli_Decomposes(Pauli_ops):
    #  [Input] Pauli_ops : List of series of Pauli operators with mixed operators w/ Re. and Im. coeff.s
    # [Output] Pauli_Res : List of series of Pauli operators which its expectation value can be directly used
    # [Output] Pauli_Ims : List of series of Pauli operators which its expectation value can be used after multiplied 1.0j
    # <Pauli_op>[index] = <Pauli_Re>[index] + 1.0j * <Pauli_Im>[index]
    Pauli_Res   = []
    Pauli_Ims   = []
    for Pauli_op in Pauli_ops:
        Pauli_Re, Pauli_Im  = Pauli_Decompose(Pauli_op)
        Pauli_Res.append(Pauli_Re)
        Pauli_Ims.append(Pauli_Im)
    
    return Pauli_Res, Pauli_Ims


########################<<< Zero Operator Filter
def zero_operator_filter(Ops, n):
    #  [Input]          Ops : List of Operators
    #  [Input]            n : Number of qubits
    # [Output] Filtered_Ops : Operators with no Zero Operator
    # [Output] Non_Zero_Obs : Numpy array with binary values (0 for original position for Zero-Op.s and vice versa)
    Filtered_Ops    = []
    Non_zero_Obs    = np.zeros(len(Ops))
    Zero_Op         = 0 * identity(n)
    for ind,op in enumerate(Ops):
        if SparsePauliOp.simplify(op) != Zero_Op:
            Filtered_Ops.append(SparsePauliOp.simplify(op))
            Non_zero_Obs[ind] = 1
    return Filtered_Ops, Non_zero_Obs
