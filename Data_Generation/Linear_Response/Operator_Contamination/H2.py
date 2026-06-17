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
N_state = 4
Contam  = np.zeros((N_state, 3))

print(f"\n>>>>>> {dist} AA Start")

# HF
mol                     = gto.M(atom=f'H 0 0 0; H 0 0 {dist}', basis='sto3g', spin=0, charge=0)
mf                      = scf.RHF(mol)
mf.kernel()
print("HF Done")

# QC
mf_qc                   = UCC(classic_object = PySCFTranspiler(mf), mapping='jordan_wigner', spin_symm=True)
mf_qc.build()
mf_qc.run(Estimator())
print("UCCSD Done")

# qEOM
mf_qEOM                 = qEOM.qEOM(mf_qc)
e, XY                   = mf_qEOM.run(Estimator(), ex_code='sd')
print("qEOM Done")

# Hermitian and Anti-Hermtian
for ex in range(N_state):
    O               = mf_qEOM.gen_state_transfer_op(excitation_level = ex)    
    H               = (O + O.conjugate().transpose()).to_matrix() / 2
    A               = (O - O.conjugate().transpose()).to_matrix() / 2
    Contam[ex, 0]   = np.linalg.norm(O.to_matrix())
    Contam[ex, 1]   = np.linalg.norm(H)
    Contam[ex, 2]   = np.linalg.norm(A)

np.save(f"/Data/H2/Operator_Contamination/{dist}", Contam)
