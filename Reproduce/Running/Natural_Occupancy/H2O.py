from pyqake.lib.Transpile import PySCFTranspiler
from pyqake.lib import MO2AO
from pyscf import scf
from scipy.optimize import linear_sum_assignment
import numpy as np
import sys

###=============================================<<< Setup

# N_ss_states : Number of states to be investigated the spin square
#    N_states : Number of states to be quantity-evaluated

# !!! Please handle active space and mol manually !!!

dist        = np.round(float(sys.argv[1]),1) # Bond distance
master_dir  = '***'
system      = 'H2O'

N_ss_states = 8
N_states    = 3

###=============================================<<< Function

# natural_orbitals    : Hermitian part only, since eigh silently drops the anti-Hermitian residue
# match_to_reference  : One-to-one assignment maximizing |<ref|vec>|^2, so that every method
#                       reports the occupation of the same orbital as the CASCI reference

def natural_orbitals(rdm):
    herm            = 0.5 * (rdm + rdm.conj().T)
    return np.linalg.eigh(herm)

def match_to_reference(occ, vec, ref_vec):
    overlap         = np.abs(ref_vec.conj().T @ vec) ** 2
    rows, cols      = linear_sum_assignment(-overlap)
    matched         = np.empty(cols.shape[0], dtype=int)
    matched[rows]   = cols
    return occ[matched]

###=============================================<<< Calculation

# 1. HF
mol, mf_chk = scf.chkfile.load_scf(f"{master_dir}/Data/{system}/Chkfile/scf_{dist}")
mf          = scf.RHF(mol)
mf.__dict__.update(mf_chk)
Transfiled      = PySCFTranspiler(mf, 6, (4, 4)) # !!! Active Space Here !!!
print("HF Done")

# 2. Spin Square
ss_AqEOM        = np.load(f"{master_dir}/Data/{system}/Spin_Square/{dist}_fqEOM.npy")
Spin_match      = np.where(ss_AqEOM < 1)[0]
N_target_state  = Spin_match.shape[0]
N_states        = min(N_states, N_target_state)
Spin_match      = Spin_match[:N_states]

# 3. Natural occupation
NO_CAS          = np.zeros((N_states, mol.nao))
NO_qEOM         = np.zeros((N_states, mol.nao))
NO_AqEOM        = np.zeros((N_states, mol.nao))
for ex in range(N_states):
    RDM_CAS             = np.load(f"{master_dir}/Data/{system}/Density/CASCI_{ex}_{dist}.npy")
    RDM_qEOM            = np.load(f"{master_dir}/Data/{system}/Density/qEOM_{Spin_match[ex]}_{dist}.npy")
    RDM_fqEOM           = np.load(f"{master_dir}/Data/{system}/Density/fqEOM_{Spin_match[ex]}_{dist}.npy")
    res_MO_RDM_CAS      = MO2AO.restore_full_space(RDM_CAS, Transfiled)
    res_MO_RDM_qEOM     = MO2AO.restore_full_space(RDM_qEOM, Transfiled)
    res_MO_RDM_fqEOM    = MO2AO.restore_full_space(RDM_fqEOM, Transfiled)
    checker             = True
    while checker:
        # CASCI eigenvector is the reference basis, its ordering labels every method
        occ_CAS, vec_CAS    = natural_orbitals(res_MO_RDM_CAS)
        occ_qEOM, vec_qEOM  = natural_orbitals(res_MO_RDM_qEOM)
        occ_fqEOM,vec_fqEOM = natural_orbitals(res_MO_RDM_fqEOM)
        NO_CAS[ex]          = occ_CAS
        NO_qEOM[ex]         = match_to_reference(occ_qEOM, vec_qEOM, vec_CAS)
        NO_AqEOM[ex]        = match_to_reference(occ_fqEOM, vec_fqEOM, vec_CAS)
        if NO_CAS.max() <= 2 and NO_qEOM.max() <= 2 and NO_AqEOM.max() <= 2:
            checker = False

# 4. Comparison
print(f"\n=== Natural occupation @ {dist} A (CASCI eigenvector basis) ===")
for ex in range(N_states):
    print(f"[State {ex}] CASCI_{ex} <-> qEOM/AqEOM_{Spin_match[ex]}")
    print("     orb" + f"{'CAS':>13s}{'qEOM':>13s}{'AqEOM':>13s}{'qEOM-CAS':>13s}{'AqEOM-CAS':>13s}")
    for orb in range(mol.nao):
        print(f"    {orb:4d}{NO_CAS[ex,orb]:13.6f}{NO_qEOM[ex,orb]:13.6f}{NO_AqEOM[ex,orb]:13.6f}"
              f"{NO_qEOM[ex,orb]-NO_CAS[ex,orb]:+13.2e}{NO_AqEOM[ex,orb]-NO_CAS[ex,orb]:+13.2e}")
    print(f"    {'sum':>4s}{NO_CAS[ex].sum():13.6f}{NO_qEOM[ex].sum():13.6f}{NO_AqEOM[ex].sum():13.6f}")
print(f"Max deviation from CASCI : qEOM {np.abs(NO_qEOM-NO_CAS).max():.2e},"
      f" AqEOM {np.abs(NO_AqEOM-NO_CAS).max():.2e}")

# 5. Save
np.save(f"{master_dir}/Data/{system}/Natural_Occupancy/CASCI_{dist}", NO_CAS)
np.save(f"{master_dir}/Data/{system}/Natural_Occupancy/qEOM_CASCI_MATCH_{dist}", NO_qEOM)
np.save(f"{master_dir}/Data/{system}/Natural_Occupancy/fqEOM_CASCI_MATCH_{dist}", NO_AqEOM)

