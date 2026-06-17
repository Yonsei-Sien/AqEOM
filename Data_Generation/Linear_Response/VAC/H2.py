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

# qEOM
for Rex in range(N_state):
    R_O = mf_qEOM.gen_state_transfer_op(excitation_level = Rex)    
    for Lex in range(Rex + 1):
        L_O                             = mf_qEOM.gen_state_transfer_op(excitation_level = Lex)
        L_O_adjoint                     = L_O.conjugate().transpose()
        Op_VAC                          = L_O_adjoint @ R_O
        Op_Re, Op_Im                    = Source.Pauli_Decompose(Op_VAC)
        Filtered_Ops, eval_excited      = Source.zero_operator_filter([Op_Re, Op_Im], 2 * mf_qc.N_orb)
        eval_excited[eval_excited > 0]  = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, Filtered_Ops, Estimator())
        Evaluated_values                = eval_excited[0] + 1.0j * eval_excited[1]
        VAC_qEOM[Lex, Rex]              = Evaluated_values
        VAC_qEOM[Rex, Lex]              = np.conjugate(Evaluated_values)

# fqEOM
for Rex in range(N_state):
    RfO  = mf_qEOM.gen_filtered_state_transfer_op(excitation_level   = Rex,
                                                  Normalize          = True,
                                                  Estimator          = Estimator())  
    for Lex in range(Rex + 1):
        LfO  = mf_qEOM.gen_filtered_state_transfer_op(excitation_level   = Lex,
                                                      Normalize          = True,
                                                      Estimator          = Estimator())  
        LfO_adjoint                     = LfO.conjugate().transpose()
        Op_VAC                          = LfO_adjoint @ RfO
        Op_Re, Op_Im                    = Source.Pauli_Decompose(Op_VAC)
        Filtered_Ops, eval_excited      = Source.zero_operator_filter([Op_Re, Op_Im], 2 * mf_qc.N_orb)
        eval_excited[eval_excited > 0]  = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, Filtered_Ops, Estimator())
        Evaluated_values                = eval_excited[0] + 1.0j * eval_excited[1]
        VAC_fqEOM[Lex, Rex]             = Evaluated_values
        VAC_fqEOM[Rex, Lex]             = np.conjugate(Evaluated_values)

np.save(f"/Data/H2/VAC/{dist}_qEOM" , VAC_qEOM)
np.save(f"/Data/H2/VAC/{dist}_fqEOM", VAC_fqEOM)