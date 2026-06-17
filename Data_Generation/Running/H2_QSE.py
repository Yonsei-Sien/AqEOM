from qiskit.primitives import StatevectorEstimator as Estimator
from pyqake.lib.Transpile import PySCFTranspiler
from pyqake.operator.observable import Dipole
from pyqake.ansatz import UCC, qEOM, QSE
from pyqake.operator import Source
from pyscf import gto,scf,lib
import numpy as np
import sys


lib.num_threads(1)

dist = np.round(float(sys.argv[1]),1)

# 1. Classical Run
mol             = gto.M(atom   = f"H 0 0 0; H 0 0 {dist}", 
                        basis  = 'sto3g',
                        charge = 0,
                        spin   = 0)
mf              = scf.RHF(mol)
E_HF            = mf.kernel()
classic_object  = PySCFTranspiler(mf)

# 2. UCC
mf_ucc          = UCC.UCC(classic_object)
mf_ucc.build()
mf_ucc.run(Estimator())

# 3. QSE
mf_qse          = QSE.QSE(classic_object, amplitudes=mf_ucc.amplitudes)
mf_qse.build()
mf_qse.ansatz   = mf_ucc.ansatz
e_QSE, XY_QSE   = mf_qse.run(Estimator())

# 4. Restore QSE Operator and Obtain Energy
E_QSE   = np.zeros_like(e_QSE)
for ex in range(len(e_QSE)):
    O                               = mf_qse.gen_state_transfer_op(excitation_level = ex)                                                   
    Effective_H                     = O.conjugate().transpose() @ mf_qse.H @ O
    Denominator                     = O.conjugate().transpose() @ O
    Op_Re, Op_Im                    = Source.Pauli_Decomposes([Effective_H, Denominator])
    Filtered_Ops, eval_excited      = Source.zero_operator_filter(Op_Re + Op_Im, 2 * mf_qse.N_orb)
    eval_excited[eval_excited > 0]  = mf_qse.Estimator_costfunction(mf_qse.amplitudes, mf_qse.ansatz, Filtered_Ops, Estimator())
    Evaluated_values                = eval_excited[:2] + 1.0j * eval_excited[2:]
    E_QSE[ex]                       = (Evaluated_values[0] / Evaluated_values[1]).real

E_AQSE  = np.zeros_like(e_QSE)
for ex in range(len(e_QSE)):
    O                               = mf_qse.gen_filtered_state_transfer_op(excitation_level        = ex,
                                                                            Normalize               = True,
                                                                            Estimator               = Estimator())                                               
    Effective_H                     = O.conjugate().transpose() @ mf_qse.H @ O
    Op_Re, Op_Im                    = Source.Pauli_Decompose(Effective_H)
    Filtered_Ops, eval_excited      = Source.zero_operator_filter([Op_Re, Op_Im], 2 * mf_qse.N_orb)
    eval_excited[eval_excited > 0]  = mf_qse.Estimator_costfunction(mf_qse.amplitudes, mf_qse.ansatz, Filtered_Ops, Estimator())
    Evaluated_value                 = eval_excited[0] + 1.0j * eval_excited[1]
    E_AQSE[ex]                      = Evaluated_value.real

# 5. Obtain Dipole Moments via restored QSE Operator
D_QSE   = np.zeros((len(e_QSE), len(e_QSE), 3))
for N_R in range(len(e_QSE)):
    R_O = mf_qse.gen_state_transfer_op(excitation_level        = N_R)
    for N_L in range(len(e_QSE)):
        L_O = mf_qse.gen_state_transfer_op(excitation_level    = N_L)
        _, D_QSE[N_L, N_R, :]           = Dipole.run(mf_qc      = mf_qse, 
                                                     Estimator  = Estimator(),
                                                     L          = L_O,
                                                     R          = R_O,
                                                     mapping    = mf_qse.mapping)
        Op_Re, Op_Im                    = Source.Pauli_Decomposes([L_O.conjugate().transpose() @ L_O, R_O.conjugate().transpose() @ R_O])
        Filtered_Ops, eval_excited      = Source.zero_operator_filter(Op_Re + Op_Im, 2 * mf_qse.N_orb)
        eval_excited[eval_excited > 0]  = mf_qse.Estimator_costfunction(mf_qse.amplitudes, mf_qse.ansatz, Filtered_Ops, Estimator())
        Evaluated_values                = eval_excited[:2] + 1.0j * eval_excited[2:]
        D_QSE[N_L, N_R, :]             /= (Evaluated_values[0] * Evaluated_values[1]).real ** 0.5
print("QSE Dipole Done")

# 6. Obtain Dipole Moments via AQSE Operator
D_AQSE   = np.zeros((len(e_QSE), len(e_QSE), 3))
for N_R in range(len(e_QSE)):
    R_O = mf_qse.gen_filtered_state_transfer_op(excitation_level        = N_R,
                                                Normalize               = True,
                                                Estimator               = Estimator())
    for N_L in range(len(e_QSE)):
        L_O = mf_qse.gen_filtered_state_transfer_op(excitation_level    = N_L,
                                                    Normalize           = True,
                                                    Estimator           = Estimator())
        _, D_AQSE[N_L, N_R, :]          = Dipole.run(mf_qc      = mf_qse, 
                                                     Estimator  = Estimator(),
                                                     L          = L_O,
                                                     R          = R_O,
                                                     mapping    = mf_qse.mapping)
print("AQSE Dipole Done")

# Check Operator Contamination
V_QSE   = np.zeros((len(e_QSE), 3))
for ex in range(len(e_QSE)):
    O           = mf_qse.gen_state_transfer_op(excitation_level = ex)
    H           = O + O.conjugate().transpose()
    A           = O - O.conjugate().transpose()
    V_QSE[0]    = np.linalg.norm(O.to_matrix())
    V_QSE[1]    = np.linalg.norm(H.to_matrix())
    V_QSE[2]    = np.linalg.norm(A.to_matrix())

# Check State Overlap
S_QSE   = np.zeros((len(e_QSE), len(e_QSE)))
for N_R in range(len(e_QSE)):
    R_O = mf_qse.gen_state_transfer_op(excitation_level        = N_R)
    for N_L in range(len(e_QSE)):
        L_O = mf_qse.gen_state_transfer_op(excitation_level    = N_L)
        Op_Re, Op_Im                    = Source.Pauli_Decompose(L_O.conjugate().transpose() @ R_O)
        Filtered_Ops, eval_excited      = Source.zero_operator_filter([Op_Re, Op_Im], 2 * mf_qse.N_orb)
        eval_excited[eval_excited > 0]  = mf_qse.Estimator_costfunction(mf_qse.amplitudes, mf_qse.ansatz, Filtered_Ops, Estimator())
        S_QSE[N_L, N_R]                 = (eval_excited[0] + 1.0j * eval_excited[1]).real
print("QSE Overlap Done")

np.save(f"/Data/H2/Energy/{dist}_QSE" , E_QSE)
np.save(f"/Data/H2/Energy/{dist}_AQSE", E_AQSE)
np.save(f"/Data/H2/Dipole/{dist}_QSE" , D_QSE)
np.save(f"/Data/H2/Dipole/{dist}_AQSE", D_AQSE)
np.save(f"/Data/H2/Operator_Contamination/{dist}_QSE", V_QSE)
np.save(f"/Data/H2/VAC/{dist}_QSE", S_QSE)