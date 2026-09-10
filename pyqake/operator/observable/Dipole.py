from qiskit.quantum_info import SparsePauliOp
from ase.data import atomic_numbers
from pyqake.operator import Source
import numpy as np


def gen_core_dipole(classic_object):
    #  [Input] classic_object : Transpiled Object for 2nd Quantization (e.g. Transpile.PySCFTranspiler)
    # [Output]                  Core electrons' dipole moments. Each name stands for the direction

    # Initialize
    ao_r            = classic_object.call('1e_r')
    mo_r_a_full     = np.einsum('ia,jb,xij->xab', classic_object.ao_mo_coeff[0], classic_object.ao_mo_coeff[0], ao_r) if classic_object.is_openshell else \
                      np.einsum('ia,jb,xij->xab', classic_object.ao_mo_coeff   , classic_object.ao_mo_coeff   , ao_r)
    mo_r_b_full     = np.einsum('ia,jb,xij->xab', classic_object.ao_mo_coeff[1], classic_object.ao_mo_coeff[1], ao_r) if classic_object.is_openshell else \
                      np.einsum('ia,jb,xij->xab', classic_object.ao_mo_coeff   , classic_object.ao_mo_coeff   , ao_r)
    r_x, r_y, r_z   = [0, 0, 0]
    if classic_object.active:
        full_N_orb  = classic_object.ao_mo_coeff.shape[1]
        if classic_object.is_openshell:
            _, r_x  = classic_object.apply_active_space(np.array([mo_r_a_full[0], mo_r_b_full[0]]), full_N_orb, classic_object.core_orb, classic_object.active_orb) 
            _, r_y  = classic_object.apply_active_space(np.array([mo_r_a_full[1], mo_r_b_full[1]]), full_N_orb, classic_object.core_orb, classic_object.active_orb) 
            _, r_z  = classic_object.apply_active_space(np.array([mo_r_a_full[2], mo_r_b_full[2]]), full_N_orb, classic_object.core_orb, classic_object.active_orb)
        else:
            _, r_x  = classic_object.apply_active_space(mo_r_a_full[0], full_N_orb, classic_object.core_orb, classic_object.active_orb) 
            _, r_y  = classic_object.apply_active_space(mo_r_a_full[1], full_N_orb, classic_object.core_orb, classic_object.active_orb) 
            _, r_z  = classic_object.apply_active_space(mo_r_a_full[2], full_N_orb, classic_object.core_orb, classic_object.active_orb)

    return np.array([r_x, r_y, r_z])


def generate(classic_object, mapping='jordan_wigner', ignore_core=False):
    #  [Input] classic_object : Transpiled Object for 2nd Quantization (e.g. Transpile.PySCFTranspiler)
    #  [Input]        mapping : Mapping Algorithm
    #  [Input]    ignore_core : Whether ignore the dipole from core electron(s)
    # [Output]     Dipole_nuc : Dipole moment vector for nuclear
    # [Output]        X, Y, Z : Eigenvalue operators for electron configuration. Each name stands for the direction

    # Initialize
    ao_r        = classic_object.call('1e_r')
    N_orb       = classic_object.N_orb
    C, D        = Source.creation_annihilations(2 * N_orb, mapping=mapping)
    mo_r_a_full = np.einsum('ia,jb,xij->xab', classic_object.ao_mo_coeff[0], classic_object.ao_mo_coeff[0], ao_r) if classic_object.is_openshell else \
                  np.einsum('ia,jb,xij->xab', classic_object.ao_mo_coeff   , classic_object.ao_mo_coeff   , ao_r)
    mo_r_b_full = np.einsum('ia,jb,xij->xab', classic_object.ao_mo_coeff[1], classic_object.ao_mo_coeff[1], ao_r) if classic_object.is_openshell else \
                  np.einsum('ia,jb,xij->xab', classic_object.ao_mo_coeff   , classic_object.ao_mo_coeff   , ao_r)
    mo_r_a      = None
    mo_r_b      = None
    r_x, r_y, r_z   = [0, 0, 0]
    if classic_object.active:
        full_N_orb  = classic_object.ao_mo_coeff.shape[1]
        ac_mo_r_a   = np.zeros((3, N_orb, N_orb))
        ac_mo_r_b   = np.zeros_like(ac_mo_r_a)
        if classic_object.is_openshell:
            [ac_mo_r_a[0], ac_mo_r_b[0]], r_x   = classic_object.apply_active_space(np.array([mo_r_a_full[0], mo_r_b_full[0]]), full_N_orb, classic_object.core_orb, classic_object.active_orb) 
            [ac_mo_r_a[1], ac_mo_r_b[1]], r_y   = classic_object.apply_active_space(np.array([mo_r_a_full[1], mo_r_b_full[1]]), full_N_orb, classic_object.core_orb, classic_object.active_orb) 
            [ac_mo_r_a[2], ac_mo_r_b[2]], r_z   = classic_object.apply_active_space(np.array([mo_r_a_full[2], mo_r_b_full[2]]), full_N_orb, classic_object.core_orb, classic_object.active_orb)
        else:
            ac_mo_r_a[0], r_x  = classic_object.apply_active_space(mo_r_a_full[0], full_N_orb, classic_object.core_orb, classic_object.active_orb) 
            ac_mo_r_a[1], r_y  = classic_object.apply_active_space(mo_r_a_full[1], full_N_orb, classic_object.core_orb, classic_object.active_orb) 
            ac_mo_r_a[2], r_z  = classic_object.apply_active_space(mo_r_a_full[2], full_N_orb, classic_object.core_orb, classic_object.active_orb)
            ac_mo_r_b[0]       = ac_mo_r_a[0]
            ac_mo_r_b[1]       = ac_mo_r_a[1]
            ac_mo_r_b[2]       = ac_mo_r_a[2]
        mo_r_a  = ac_mo_r_a
        mo_r_b  = ac_mo_r_b
    else:
        mo_r_a  = mo_r_a_full
        mo_r_b  = mo_r_b_full

    # Dipole vector for Nuclear
    Dipole_nuc  = np.zeros(3)
    atom_symbol = classic_object.call('atom_symbol')
    atom_coords = classic_object.call('atom_coords')
    for atom_ind in range(classic_object.N_atom):
        atom_Z          = atomic_numbers[atom_symbol(atom_ind)]
        Dipole_nuc[0]  += atom_Z * atom_coords[atom_ind][0]
        Dipole_nuc[1]  += atom_Z * atom_coords[atom_ind][1]
        Dipole_nuc[2]  += atom_Z * atom_coords[atom_ind][2]

    # Generate operators
    X       = 0 * Source.identity(2 * N_orb) if ignore_core else r_x * Source.identity(2 * N_orb) # Dipole Operator for X-Direction
    Y       = 0 * Source.identity(2 * N_orb) if ignore_core else r_y * Source.identity(2 * N_orb) # Dipole Operator for Y-Direction
    Z       = 0 * Source.identity(2 * N_orb) if ignore_core else r_z * Source.identity(2 * N_orb) # Dipole Operator for Z-Direction
    for p in range(N_orb):
        for r in range(N_orb):
            X += mo_r_a[0, p, r] * C[p] @ D[r]
            X += mo_r_b[0, p, r] * C[N_orb + p] @ D[N_orb + r]
            Y += mo_r_a[1, p, r] * C[p] @ D[r]
            Y += mo_r_b[1, p, r] * C[N_orb + p] @ D[N_orb + r]
            Z += mo_r_a[2, p, r] * C[p] @ D[r]
            Z += mo_r_b[2, p, r] * C[N_orb + p] @ D[N_orb + r]

    return Dipole_nuc, X, Y, Z


def run(mf_qc, Estimator, L=None, R=None, mapping='jordan_wigner', ignore_core=False):
    # [Input]       mf_qc : QC object such as UCC.UCC
    # [Input]   Estimator : Qiskit estimator object
    # [Input]        L, R : Operators for excitation in case of TDM. <0|L_O_dagger @ Dipole @ R_O|0>
    # [Input]     mapping : Mapping Algorithm
    # [Input] ignore_core : Whether ignore the dipole from core electron(s)
    # Initialize
    identity = Source.identity(2 * mf_qc.classic_object.N_orb)
    if L == None:
        L = identity
    if R == None:
        R = identity
    L = L.conjugate().transpose()

    # Generate Dipole Moment Operators
    Dipole_nuc, X, Y, Z         = generate(classic_object = mf_qc.classic_object, 
                                           mapping        = mapping,
                                           ignore_core    = ignore_core)
    D_X                         = SparsePauliOp.simplify(L @ X @ R)
    D_Y                         = SparsePauliOp.simplify(L @ Y @ R)
    D_Z                         = SparsePauliOp.simplify(L @ Z @ R)

    # Compute Dipole Moments
    DM_Ops                      = [D_X, D_Y, D_Z]
    DM_Re, DM_Im                = Source.Pauli_Decomposes(DM_Ops)
    Filtered_DM_Ops, eval_DM    = Source.zero_operator_filter(DM_Re + DM_Im, 2 * mf_qc.N_orb)
    eval_DM[eval_DM > 0]        = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, Filtered_DM_Ops, Estimator)
    Dipole_elec                 = eval_DM[:3] + 1.0j * eval_DM[3:]

    return Dipole_nuc, Dipole_elec

