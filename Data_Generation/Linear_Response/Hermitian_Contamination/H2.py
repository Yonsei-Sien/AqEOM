from qiskit.primitives import StatevectorEstimator as Estimator
from pyqake.lib.Transpile import PySCFTranspiler
from pyqake.operator.observable import Dipole
from pyqake.operator import Source
from pyqake.ansatz.UCC import UCC
from pyqake.ansatz import qEOM
from pyscf import gto, scf
import numpy as np
import sys


dist                = np.round(float(sys.argv[1]),1)
N_state             = 4
VAC_qEOM            = np.zeros((N_state, N_state))
VAC_fqEOM           = np.zeros((N_state, N_state))

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
    O = mf_qEOM.gen_state_transfer_op(excitation_level = ex)    
    Op_Re, Op_Im                    = Source.Pauli_Decomposes([O+O.conjugate().transpose(), O-O.conjugate().transpose()])
    Filtered_Ops, eval_excited      = Source.zero_operator_filter(Op_Re + Op_Im, 2 * mf_qc.N_orb)
    eval_excited[eval_excited > 0]  = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, Filtered_Ops, Estimator())
    Evaluated_values                = eval_excited[:2] + 1.0j * eval_excited[2:]
    VAC_qEOM[ex]                    = Evaluated_values[0]
    VAC_fqEOM[ex]                   = Evaluated_values[1]

np.save(f"/Data/H2/Hermitian_Contamination/{dist}_qEOM" , VAC_qEOM)
np.save(f"/Data/H2/Hermitian_Contamination/{dist}_fqEOM", VAC_fqEOM)