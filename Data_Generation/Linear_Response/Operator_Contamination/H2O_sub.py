from qiskit.primitives import StatevectorEstimator as Estimator
from pyqake.lib.Transpile import PySCFTranspiler
from pyqake.operator.observable import Dipole
from pyqake.operator import Source
from pyqake.ansatz.UCC import UCC
from pyqake.ansatz import qEOM
from pyscf import gto, scf
import numpy as np
import sys


dist    = np.round(float(sys.argv[1]),1)
N_state = 6
Contam  = np.zeros((N_state, 3))

print(f"\n>>>>>> {dist} AA Start")

# HF
mol, mf_chk             = scf.chkfile.load_scf(f"/Data/H2O/Chkfile/scf_{dist}")
mf                      = scf.RHF(mol)
mf.__dict__.update(mf_chk)

# QC
ucc_amp                 = np.load(f"/Data/H2O/Res/UCC_{dist}.npy")
mf_qc                   = UCC(classic_object = PySCFTranspiler(mf, active_space=6, active_e=(4, 4)), mapping='jordan_wigner', spin_symm=True, amplitudes=ucc_amp)
mf_qc.build()
print("UCCSD Done")

# qEOM
mf_qEOM                 = qEOM.qEOM(mf_qc)
mf_qEOM.e               = np.load(f"/Data/H2O/qEOM/E_{dist}.npy")
mf_qEOM.XY              = np.load(f"/Data/H2O/qEOM/XY_{dist}.npy")
mf_qEOM._gen_ops()
print("qEOM Done")

# Hermitian and Anti-Hermtian
for ex in range(N_state):
    O               = mf_qEOM.gen_state_transfer_op(excitation_level = ex)    
    H               = (O + O.conjugate().transpose()).to_matrix() / 2
    A               = (O - O.conjugate().transpose()).to_matrix() / 2
    Contam[ex, 0]   = np.linalg.norm(O.to_matrix())
    Contam[ex, 1]   = np.linalg.norm(H)
    Contam[ex, 2]   = np.linalg.norm(A)

np.save(f"/Data/H2O/Operator_Contamination/{dist}", Contam)
