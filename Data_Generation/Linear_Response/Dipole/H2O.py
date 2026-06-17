from qiskit.primitives import StatevectorEstimator as Estimator
from pyqake.lib.Transpile import PySCFTranspiler
from pyqake.operator.observable import Dipole
from pyqake.operator import Source
from pyqake.ansatz.UCC import UCC
from pyqake.ansatz import qEOM
from pyscf import gto, scf
import numpy as np
import sys


dist            = np.round(float(sys.argv[1]),1)
N_state         = 4
dipole_qEOM     = np.zeros((N_state, N_state, 3))
dipole_fqEOM    = np.zeros((N_state, N_state, 3))

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

# qEOM Dipole
for N_R in range(N_state):
    R_O = mf_qEOM.gen_state_transfer_op(excitation_level        = N_R)
    for N_L in range(N_R + 1):
        L_O = mf_qEOM.gen_state_transfer_op(excitation_level    = N_L)
        _, dipole_qEOM[N_L, N_R, :]     = Dipole.run(mf_qc      = mf_qc, 
                                                     Estimator  = Estimator(),
                                                     L          = L_O,
                                                     R          = R_O,
                                                     mapping    = mf_qc.mapping)
        Op_Re, Op_Im                    = Source.Pauli_Decomposes([L_O.conjugate().transpose() @ L_O, R_O.conjugate().transpose() @ R_O])
        Filtered_Ops, eval_excited      = Source.zero_operator_filter(Op_Re + Op_Im, 2 * mf_qc.N_orb)
        eval_excited[eval_excited > 0]  = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, Filtered_Ops, Estimator())
        Evaluated_values                = eval_excited[:2] + 1.0j * eval_excited[2:]
        dipole_qEOM[N_L, N_R, :]       /= (Evaluated_values[0] * Evaluated_values[1]).real ** 0.5
print("qEOM Dipole Done")

# fqEOM Dipole
for N_R in range(N_state):
    R_O = mf_qEOM.gen_filtered_state_transfer_op(excitation_level       = N_R,
                                                 Normalize              = True,
                                                 Estimator              = Estimator())
    for N_L in range(N_R + 1):
        L_O = mf_qEOM.gen_filtered_state_transfer_op(excitation_level   = N_L,
                                                     Normalize          = True,
                                                     Estimator          = Estimator())
        _, dipole_fqEOM[N_L, N_R, :] = Dipole.run(mf_qc                 = mf_qc, 
                                                  Estimator             = Estimator(),
                                                  L                     = L_O,
                                                  R                     = R_O,
                                                  mapping               = mf_qc.mapping)
print("fqEOM Dipole Done")

np.save(f"/Data/H2O/Dipole/{dist}_qEOM" , dipole_qEOM)
np.save(f"/Data/H2O/Dipole/{dist}_fqEOM", dipole_fqEOM)