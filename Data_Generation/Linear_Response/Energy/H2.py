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
energy_qEOM         = np.zeros(N_state)
energy_fqEOM        = np.zeros(N_state)
energy_qEOM_diag    = np.zeros(N_state)

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
energy_qEOM[0]          = mf_qc.energy
energy_fqEOM[0]         = mf_qc.energy
energy_qEOM_diag[0]     = mf_qc.energy
print("UCCSD Done")

# qEOM
mf_qEOM                 = qEOM.qEOM(mf_qc)
e, XY                   = mf_qEOM.run(Estimator(), ex_code='sd')
energy_qEOM_diag[1:]    = np.sort(e[e > 0])[:N_state-1] + mf_qc.energy
print("qEOM Done")

# qEOM
for ex in range(N_state - 1):
    O                               = mf_qEOM.gen_state_transfer_op(excitation_level = ex + 1)                                                   
    Effective_H                     = O.conjugate().transpose() @ mf_qc.H @ O
    Denominator                     = O.conjugate().transpose() @ O
    Op_Re, Op_Im                    = Source.Pauli_Decomposes([Effective_H, Denominator])
    Filtered_Ops, eval_excited      = Source.zero_operator_filter(Op_Re + Op_Im, 2 * mf_qc.N_orb)
    eval_excited[eval_excited > 0]  = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, Filtered_Ops, Estimator())
    Evaluated_values                = eval_excited[:2] + 1.0j * eval_excited[2:]
    energy_qEOM[ex + 1]             = (Evaluated_values[0] / Evaluated_values[1]).real

# fqEOM
for ex in range(N_state - 1):
    fO  = mf_qEOM.gen_filtered_state_transfer_op(excitation_level   = ex + 1,
                                             Normalize          = True,
                                             Estimator          = Estimator())
                                                                    
    Effective_fH                    = fO.conjugate().transpose() @ mf_qc.H @ fO
    Op_Re, Op_Im                    = Source.Pauli_Decompose(Effective_fH)
    Filtered_Ops, eval_excited      = Source.zero_operator_filter([Op_Re, Op_Im], 2 * mf_qc.N_orb)
    eval_excited[eval_excited > 0]  = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, Filtered_Ops, Estimator())
    energy_fqEOM[ex + 1]            = (eval_excited[0] + 1.0j * eval_excited[1]).real

np.save(f"/Data/H2/Energy/{dist}_qEOM_Diag", energy_qEOM_diag)
np.save(f"/Data/H2/Energy/{dist}_qEOM"     , energy_qEOM)
np.save(f"/Data/H2/Energy/{dist}_fqEOM"    , energy_fqEOM)