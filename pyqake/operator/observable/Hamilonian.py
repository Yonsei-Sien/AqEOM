from pyqake.lib import Cholesky_Decomposition
from pyqake.operator import Source
import numpy as np


def generate(classic_object, mapping='jordan_wigner', cd_acc=1e-6):
    #  [Input] classic_object : transpiled object from Transpiler.PySCFTranspiler
    #  [Input]        mapping : Mapping algorithm from fermionic into qubit
    #  [Input]         cd_acc : Cholesky Decomposition Accuracy
    # [Output]                : Second quantized Hamiltonian mapped into qubit space

    # 1. Initialize variables to use
    ecore   = classic_object.e0_core    # Float
    h1e     = classic_object.mo_h1e     # 2D matrix or matrices for alpha and beta spins
    h2e     = classic_object.mo_h2e     # 4D tensor
    N_orb   = classic_object.N_orb      # Number of Spatial Orbitals 

    if not(classic_object.is_openshell): # Closed Shell        
        # 2. Make creation & annihilation operators with Pauli operators
        C, D = Source.creation_annihilations(2 * N_orb, mapping=mapping)
        Exc = []
        for p in range(N_orb):
            Excp = [C[p] @ D[p] + C[N_orb + p] @ D[N_orb + p]]
            for r in range(p + 1, N_orb):
                Excp.append(
                    C[p] @ D[r]
                    + C[N_orb + p] @ D[N_orb + r]
                    + C[r] @ D[p]
                    + C[N_orb + r] @ D[N_orb + p]
                )
            Exc.append(Excp)

        # 3. Do low-rank decomposition of the 2e Hamiltonian for operator optimization
        # By fermionic property, 2-body term can be decomposed into multiply of 1 body term with certain accuracy
        Lop, ng = Cholesky_Decomposition.run(h2e, cd_acc)
        t1e = h1e - 0.5 * np.einsum("pxxr->pr", h2e) # decomposed 2e term's S + original 1e term

        # 4. Sum all operators with coefficients we've got above
        # 0-body term
        H = ecore * Source.identity(2 * N_orb)

        # one-body term
        for p in range(N_orb):
            for r in range(p, N_orb):
                H += t1e[p, r] * Exc[p][r - p] # r-p follows the order of the operator list

        # two-body term
        for g in range(ng):
            Lg = 0 * Source.identity(2 * N_orb)
            for p in range(N_orb):
                for r in range(p, N_orb):
                    Lg += Lop[p, r, g] * Exc[p][r - p]
            H += 0.5 * Lg @ Lg

        return H.simplify()
        
    else: # Open Shell
        h1e_a   = h1e[0]
        h1e_b   = h1e[1]

        # 2. Make creation & annihilation operators with Pauli operators
        C, D = Source.creation_annihilations(2 * N_orb, mapping=mapping)

        # 3. Sum all operators with coefficients we've got above
        H = ecore * Source.identity(2 * N_orb)

        # one-body term
        for p in range(N_orb):
            for r in range(N_orb):
                    H += h1e_a[p, r] * C[p] @ D[r]
                    H += h1e_b[p, r] * C[N_orb + p] @ D[N_orb + r]

        # two-body term ~ pq: spin1 | rs: spin2
        for p in range(N_orb):
            for q in range(N_orb):
                for r in range(N_orb):
                    for s in range(N_orb):
                        H += 0.5 * h2e[p, q, r, s] * C[p] @ C[r] @ D[s] @ D[q]
                        H += 0.5 * h2e[N_orb + p, N_orb + q, N_orb + r, N_orb + s] * C[N_orb + p] @ C[N_orb + r] @ D[N_orb + s] @ D[N_orb + q]
                        H += 0.5 * h2e[p, q, N_orb + r, N_orb + s] * C[p] @ C[N_orb + r] @ D[N_orb + s] @ D[q]
                        H += 0.5 * h2e[N_orb + p, N_orb + q, r, s] * C[N_orb + p] @ C[r] @ D[s] @ D[N_orb + q]

        return H.simplify()