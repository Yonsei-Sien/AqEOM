from pyqake.operator import Source
import numpy as np
    

def generate(classic_object, mapping='jordan_wigner'):
    #  [Input]   classic_object : transpiled object from Transpiler.PySCFTranspiler
    #  [Input]          mapping : Mapping algorithm from fermionic into qubit
    # [Output] S+S-, S-S+, SzSz : Components of spin square eigenvalue
    #
    # Operator Generator for <S^2> = <0.5 * (S+S- + S-S+) + Sz^2>
    # classic_object is transpiled object from Transpiler.classic_object
    # mo_coeff is 2D matrix or matrices for alpha and beta spins
    N_orb       = classic_object.N_orb
    S_ao        = classic_object.ao_ovlp
    ao_mo_a     = classic_object.ao_mo_coeff[0] if classic_object.is_openshell else classic_object.ao_mo_coeff
    ao_mo_b     = classic_object.ao_mo_coeff[1] if classic_object.is_openshell else classic_object.ao_mo_coeff
    S_mo        = np.einsum('ia,jb,ij->ab', ao_mo_a, ao_mo_b, S_ao)
    SS_mo       = np.einsum('pq,rs->pqrs', S_mo, S_mo)
    SS_2_body   = SS_mo
    SS_1_body   = np.zeros_like(S_mo)
    SS_constant = 0
    if classic_object.active:
        SS_2_body, SS_1_body, SS_constant   = classic_object.apply_active_space(SS_mo, N_orb, classic_object.core_orb, classic_object.active_orb)

    # 2. Make creation & annihilation operators with Pauli operators
    C, D        = Source.creation_annihilations(2 * N_orb, mapping=mapping)

    # 3. Make Spin Square Operator Components
    SpSm        = 0 * Source.identity(2 * N_orb) # S_+ @ S_-
    SmSp        = 0 * Source.identity(2 * N_orb) # S_+ @ S_-
    for orb_1 in range(N_orb):
        for orb_2 in range(N_orb):
            SpSm   += (SS_1_body[orb_1, orb_2]) * C[orb_1] @ D[N_orb + orb_2]
            SmSp   += (SS_1_body[orb_2, orb_1]) * C[N_orb + orb_1] @ D[orb_2]
            for orb_3 in range(N_orb):
                for orb_4 in range(N_orb):
                    SpSm   += (SS_2_body[orb_1, orb_2, orb_4, orb_3]) * C[orb_1] @ D[N_orb + orb_2] @ C[N_orb + orb_3] @ D[orb_4]
                    SmSp   += (SS_2_body[orb_2, orb_1, orb_3, orb_4]) * C[N_orb + orb_1] @ D[orb_2] @ C[orb_3] @ D[N_orb + orb_4]

    Sz          = 0 * Source.identity(2 * N_orb) # S_z
    for orb in range(N_orb):
        Sz     += C[orb] @ D[orb] - C[N_orb + orb] @ D[N_orb + orb]

    SS_constant = len(classic_object.core_orb[0]) - SS_constant if classic_object.active else 0
    return SpSm.simplify(), SmSp.simplify(), (Sz @ Sz).simplify(), SS_constant