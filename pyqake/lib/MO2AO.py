import numpy as np


def run(RDM, mo_coeff):
    #  [Input]      RDM : Reduced Density Matrix with MO indices
    #  [Input] mo_coeff : MO coefficient with AO indices
    # [Output]      RDM : Reduced Density Matrix with AO indices
    # If we get MO-based RDM, sometimes it is needed to change into AO representation
    is_split_RDM        = len(RDM.shape) - 2
    is_split_mo_coeff   = len(mo_coeff.shape) - 2

    if is_split_RDM:
        if is_split_mo_coeff:
            D1   = np.einsum('ij,pi,qj->pq', RDM[0], mo_coeff[0], mo_coeff[0])
            D2   = np.einsum('ij,pi,qj->pq', RDM[1], mo_coeff[1], mo_coeff[1])
            return np.array([D1, D2])
        else:
            D1   = np.einsum('ij,pi,qj->pq', RDM[0], mo_coeff, mo_coeff)
            D2   = np.einsum('ij,pi,qj->pq', RDM[1], mo_coeff, mo_coeff)
            return np.array([D1, D2])
    else:
        if is_split_mo_coeff:
            D1   = np.einsum('ij,pi,qj->pq', RDM / 2, mo_coeff[0], mo_coeff[0])
            D2   = np.einsum('ij,pi,qj->pq', RDM / 2, mo_coeff[1], mo_coeff[1])
            return np.array([D1, D2])
        else:
            D0   = np.einsum('ij,pi,qj->pq', RDM, mo_coeff, mo_coeff)
            return D0


def restore_full_space(RDM, classic_object):
    if not(classic_object.active):
        return RDM
    else:
        N_basis             = classic_object.ao_mo_coeff.shape[1]
        is_RDM_openshell    = len(RDM.shape) - 2
        if is_RDM_openshell:
            restored_RDM                                                                        = np.zeros((2, N_basis, N_basis))
            restored_RDM[0][np.ix_(classic_object.core_orb[0], classic_object.core_orb[0])]     = np.eye(len(classic_object.core_orb[0]))
            restored_RDM[1][np.ix_(classic_object.core_orb[1], classic_object.core_orb[1])]     = np.eye(len(classic_object.core_orb[1]))
            restored_RDM[0][np.ix_(classic_object.active_orb[0], classic_object.active_orb[0])] = RDM[0]
            restored_RDM[1][np.ix_(classic_object.active_orb[1], classic_object.active_orb[1])] = RDM[1]
            return restored_RDM
        else:
            restored_RDM                                                                        = np.zeros((N_basis, N_basis))
            restored_RDM[np.ix_(classic_object.core_orb[0], classic_object.core_orb[0])]        = 2 * np.eye(len(classic_object.core_orb[0]))
            restored_RDM[np.ix_(classic_object.active_orb[0], classic_object.active_orb[0])]    = RDM
            return restored_RDM
            