from qiskit.quantum_info import SparsePauliOp
from pyqake.operator import Source
from functools import reduce
import itertools


def generate(N_orb, N_alpha, N_beta, N_excitation, mapping='jordan_wigner', restricted=True, spin_symm=True, hermitian=True):
    #  [Input]        N_orb : Number of spatial orbitals
    #  [Input]      N_alpha : Number of alpha electrons
    #  [Input]       N_beta : Number of beta electrons
    #  [Input] N_excitation : Number of excitation order (single=1, double=2, ...)
    #  [Input]      mapping : Mapping algorithm from fermionic into qubit
    #  [Input]   restricted : Whether excitations are allowed 
    #                          True) occ. to virt. (e.g. UCC) 
    #                         False) for every orb. to orb. (e.g. ADAPT-VQE)
    #  [Input]    spin_symm : Whether electrons excited 
    #                          True) spin-symmetrically
    #                         False) or not
    #  [Input]    hermitian : Whether excitation operators are 
    #                          True) sum of their -{adjoint} 
    #                         False) or not
    # [Output]              : List of excitation operators and List of indices

    if spin_symm and N_alpha == N_beta: # Based on Spin Symmetry
        # Only Occ. to Virt. with Spin Symmetry
        if restricted: 
            # Check whether N_excitation is reasonable
            max_alpha_excitation_allowed    = min(N_alpha, N_orb - N_alpha)
            max_excitation_allow            = max_alpha_excitation_allowed * 2
            if N_excitation > max_excitation_allow:
                raise ValueError("N_excitation is out of range")
            
            # Initialize
            Operators           = []
            Op_indices          = []
            C, D                = Source.creation_annihilations(2*N_orb, mapping=mapping)

            alpha_occ           = list(range(N_alpha))
            alpha_virt          = list(range(N_alpha, N_orb))
            beta_occ            = list(range(N_beta))
            beta_virt           = list(range(N_beta, N_orb))

            # Excitations
            for N_excited_alpha in range(max_alpha_excitation_allowed + 1):
                N_excited_beta = N_excitation - N_excited_alpha
                # If this number of alpha and beta excitation is reasonable
                if N_excited_beta >= 0 and N_excited_beta <= max_alpha_excitation_allowed and N_excited_alpha >= N_excited_beta: 
                    # We select N_excitation possible excitations from alpha and beta
                    alpha_froms = itertools.combinations(alpha_occ, N_excited_alpha)
                    alpha_tos   = itertools.combinations(alpha_virt, N_excited_alpha)
                    beta_froms  = itertools.combinations(beta_occ, N_excited_beta)
                    beta_tos    = itertools.combinations(beta_virt, N_excited_beta)

                    for alpha_from, alpha_to, beta_from, beta_to in itertools.product(alpha_froms, alpha_tos, beta_froms, beta_tos):
                        if N_excited_alpha == N_excited_beta:
                            alpha_config = (alpha_from, alpha_to)
                            beta_config  = (beta_from, beta_to)
                            if alpha_config < beta_config:
                                continue
                        T         , T_ind           = _generate(C, D, N_orb, alpha_from, alpha_to, beta_from, beta_to)
                        T_opposite, T_opposite_ind  = _generate(C, D, N_orb, beta_from, beta_to, alpha_from, alpha_to)
                        if hermitian:
                            Operators.append(SparsePauliOp.simplify(T + T_opposite - T.conjugate().transpose() - T_opposite.conjugate().transpose()))
                            Op_indices.append(f"{T_ind[0] + '//' + T_ind[1]} (+) {T_opposite_ind[0] + '//' + T_opposite_ind[1]} (-) " \
                                            +f"{T_ind[1] + '//' + T_ind[0]} (-) {T_opposite_ind[1] + '//' + T_opposite_ind[0]}")
                        else:
                            Operators.append(SparsePauliOp.simplify(T + T_opposite))
                            Op_indices.append(f"{T_ind[0] + '//' + T_ind[1]} (+) {T_opposite_ind[0] + '//' + T_opposite_ind[1]}")
            return Operators, Op_indices
        
        # All possible excitations with Spin Symmetry
        else: 
            # Initialize
            Operators   = []
            Op_indices  = []
            C, D        = Source.creation_annihilations(2*N_orb, mapping=mapping)
            orb_total   = list(range(N_orb))

            # Excitations
            for N_excited_alpha in range(N_excitation + 1):
                N_excited_beta = N_excitation - N_excited_alpha
                if N_excited_alpha >= N_excited_beta:
                    # We select N_excitation possible excitations from alpha and beta
                    alpha_froms = itertools.product(orb_total, repeat=N_excited_alpha)
                    alpha_tos   = itertools.product(orb_total, repeat=N_excited_alpha)
                    beta_froms  = itertools.product(orb_total, repeat=N_excited_beta)
                    beta_tos    = itertools.product(orb_total, repeat=N_excited_beta)

                    for alpha_from, alpha_to, beta_from, beta_to in itertools.product(alpha_froms, alpha_tos, beta_froms, beta_tos):
                        T         , T_ind           = _generate(C, D, N_orb, alpha_from, alpha_to, beta_from, beta_to)
                        T_opposite, T_opposite_ind  = _generate(C, D, N_orb, beta_from, beta_to, alpha_from, alpha_to)
                        if hermitian:
                            Operators.append(SparsePauliOp.simplify(T + T_opposite - T.conjugate().transpose() - T_opposite.conjugate().transpose()))
                            Op_indices.append(f"{T_ind[0] + '//' + T_ind[1]} (+) {T_opposite_ind[0] + '//' + T_opposite_ind[1]} (-) " \
                                            +f"{T_ind[1] + '//' + T_ind[0]} (-) {T_opposite_ind[1] + '//' + T_opposite_ind[0]}")
                        else:
                            Operators.append(SparsePauliOp.simplify(T + T_opposite))
                            Op_indices.append(f"{T_ind[0] + '//' + T_ind[1]} (+) {T_opposite_ind[0] + '//' + T_opposite_ind[1]}")
            return Operators, Op_indices
        
    # No Spin Symmetry
    else: 
        # Only Occ. to Virt. without Spin Symmetry
        if restricted: 
            # Check whether N_excitation is reasonable
            max_alpha_excitation_allowed    = min(N_alpha, N_orb - N_alpha)
            max_beta_excitation_allowed     = min(N_beta, N_orb - N_beta)
            max_excitation_allow            = max_alpha_excitation_allowed + max_beta_excitation_allowed
            if N_excitation > max_excitation_allow:
                raise ValueError("N_excitation is out of range")
            
            # Initialize
            Operators   = []
            Op_indices  = []
            C, D        = Source.creation_annihilations(2*N_orb, mapping=mapping)

            alpha_occ   = list(range(N_alpha))
            alpha_virt  = list(range(N_alpha, N_orb))
            beta_occ    = list(range(N_beta))
            beta_virt   = list(range(N_beta, N_orb))

            # Excitations
            for N_excited_alpha in range(max_alpha_excitation_allowed + 1):
                N_excited_beta = N_excitation - N_excited_alpha
                # If this number of alpha and beta excitation is reasonable
                if N_excited_beta >= 0 and N_excited_beta <= max_beta_excitation_allowed: 
                    # We select N_excitation possible excitations from alpha and beta
                    alpha_froms = itertools.combinations(alpha_occ, N_excited_alpha)
                    alpha_tos   = itertools.combinations(alpha_virt, N_excited_alpha)
                    beta_froms  = itertools.combinations(beta_occ, N_excited_beta)
                    beta_tos    = itertools.combinations(beta_virt, N_excited_beta)

                    for alpha_from, alpha_to, beta_from, beta_to in itertools.product(alpha_froms, alpha_tos, beta_froms, beta_tos):
                        T         , T_ind           = _generate(C, D, N_orb, alpha_from, alpha_to, beta_from, beta_to)
                        if hermitian:
                            Operators.append(SparsePauliOp.simplify(T - T.conjugate().transpose()))
                            Op_indices.append(f"{T_ind[0] + '//' + T_ind[1]} (-) {T_ind[1] + '//' + T_ind[0]}")
                        else:
                            Operators.append(T)
                            Op_indices.append(f"{T_ind[0] + '//' + T_ind[1]}")
            return Operators, Op_indices
        
        # All possible excitations without Spin Symmetry
        else: 
            # Initialize
            Operators   = []
            Op_indices  = []
            C, D        = Source.creation_annihilations(2*N_orb, mapping=mapping)
            orb_total   = list(range(N_orb))

            # Excitations
            for N_excited_alpha in range(N_excitation + 1):
                N_excited_beta = N_excitation - N_excited_alpha
                # We select N_excitation possible excitations from alpha and beta
                alpha_froms = itertools.product(orb_total, repeat=N_excited_alpha)
                alpha_tos   = itertools.product(orb_total, repeat=N_excited_alpha)
                beta_froms  = itertools.product(orb_total, repeat=N_excited_beta)
                beta_tos    = itertools.product(orb_total, repeat=N_excited_beta)

                for alpha_from, alpha_to, beta_from, beta_to in itertools.product(alpha_froms, alpha_tos, beta_froms, beta_tos):
                    T         , T_ind           = _generate(C, D, N_orb, alpha_from, alpha_to, beta_from, beta_to)
                    if hermitian:
                        Operators.append(SparsePauliOp.simplify(T - T.conjugate().transpose()))
                        Op_indices.append(f"{T_ind[0] + '//' + T_ind[1]} (-) {T_ind[1] + '//' + T_ind[0]}")
                    else:
                        Operators.append(T)
                        Op_indices.append(f"{T_ind[0] + '//' + T_ind[1]}")
                    if N_excited_alpha == N_excited_beta:
                        T_opposite, T_opposite_ind  = _generate(C, D, N_orb, beta_from, beta_to, alpha_from, alpha_to)
                        if hermitian:
                            Operators.append(SparsePauliOp.simplify(T - T.conjugate().transpose()))
                            Op_indices.append(f"{T_opposite_ind[0] + '//' + T_opposite_ind[1]} (-) {T_opposite_ind[1] + '//' + T_opposite_ind[0]}")
                        else:
                            Operators.append(T)
                            Op_indices.append(f"{T_opposite_ind[0] + '//' + T_opposite_ind[1]}")
            return Operators, Op_indices


def _generate(C, D, N_orb, alpha_from, alpha_to, beta_from, beta_to):
    def mult_ops(A, B):
        return A @ B

    T_before_product    = [C[a] for a in alpha_to]          \
                        + [C[N_orb + b] for b in beta_to]   \
                        + [D[N_orb + b] for b in beta_from] \
                        + [D[a] for a in alpha_from]
    T_ind_lst_C         = [f"a{a}" for a in alpha_to]       \
                        + [f"b{b}" for b in beta_to]
    T_ind_lst_D         = [f"b{b}" for b in beta_from]      \
                        + [f"a{a}" for a in alpha_from]
    T                   = reduce(mult_ops, T_before_product)
    T_ind               = ['/'.join(T_ind_lst_C), '/'.join(T_ind_lst_D)]
    return T, T_ind


def generates(N_orb, N_alpha, N_beta, ex_code='sd', mapping='jordan_wigner', restricted=True, spin_symm=True, hermitian=True):
    #  [Input]        N_orb : Number of spatial orbitals
    #  [Input]      N_alpha : Number of alpha electrons
    #  [Input]       N_beta : Number of beta electrons
    #  [Input]      ex_code : Excitation codes (sdtqph for 1 to 6)
    #  [Input]      mapping : Mapping algorithm from fermionic into qubit
    #  [Input]   restricted : Whether excitations are allowed 
    #                          True) occ. to virt. (e.g. UCC) 
    #                         False) for every orb. to orb. (e.g. ADAPT-VQE)
    #  [Input]    spin_symm : Whether electrons excited 
    #                          True) spin-symmetrically
    #                         False) or not
    #  [Input]    hermitian : Whether excitation operators are 
    #                          True) sum of their -{adjoint} 
    #                         False) or not
    # [Output]              : List of excitation operators and List of indices

    # Initialize
    ex_code         = ex_code.lower().strip()
    ex_translate    = {'s': 1, 'd': 2, 't': 3, 'q': 4, 'p': 5, 'h': 6}
    Operators       = []
    Op_indices      = []

    # Generate Operators
    for excitation in list(ex_translate.keys()):
        if excitation in ex_code:
            Ops, Inds   = generate(N_orb, N_alpha, N_beta, ex_translate[excitation], mapping, restricted, spin_symm, hermitian)
            Operators  += Ops
            Op_indices += Inds

    return Operators, Op_indices