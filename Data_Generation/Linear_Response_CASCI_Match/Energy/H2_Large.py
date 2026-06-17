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
N_state             = 10
CAS_match           = np.where(np.load(f"/Data/H2_Large/Spin_Square/{dist}_fqEOM.npy") < 1)[0][1:]
energy_qEOM         = np.zeros(N_state)
energy_fqEOM        = np.zeros(N_state)
energy_qEOM_diag    = np.zeros(N_state)

print(f"\n>>>>>> {dist} AA Start")

# HF
mol, mf_chk             = scf.chkfile.load_scf(f"/Data/H2_Large/Chkfile/scf_{dist}")
mf                      = scf.RHF(mol)
mf.__dict__.update(mf_chk)

# QC
ucc_amp                 = np.load(f"/Data/H2_Large/Res/UCC_{dist}.npy")
mf_qc                   = UCC(classic_object = PySCFTranspiler(mf), mapping='jordan_wigner', spin_symm=True, amplitudes=ucc_amp)
mf_qc.build()
mf_qc.energy            = mf_qc.energy_tot(Estimator())
print("UCCSD Done")

# qEOM
mf_qEOM                 = qEOM.qEOM(mf_qc)
mf_qEOM.e               = np.load(f"/Data/H2_Large/qEOM/E_{dist}.npy")
mf_qEOM.XY              = np.load(f"/Data/H2_Large/qEOM/XY_{dist}.npy")
mf_qEOM._gen_ops()
energy_qEOM[0]          = mf_qc.energy
energy_fqEOM[0]         = mf_qc.energy
energy_qEOM_diag[0]     = mf_qc.energy
energy_qEOM_diag[1:]    = np.sort(mf_qEOM.e[mf_qEOM.e > 0])[CAS_match - 1] + mf_qc.energy
print("qEOM Done")

# qEOM
for ex in range(N_state - 1):
    O                               = mf_qEOM.gen_state_transfer_op(excitation_level = CAS_match[ex])                                                   
    Effective_H                     = O.conjugate().transpose() @ mf_qc.H @ O
    Denominator                     = O.conjugate().transpose() @ O
    Op_Re, Op_Im                    = Source.Pauli_Decomposes([Effective_H, Denominator])
    Filtered_Ops, eval_excited      = Source.zero_operator_filter(Op_Re + Op_Im, 2 * mf_qc.N_orb)
    eval_excited[eval_excited > 0]  = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, Filtered_Ops, Estimator())
    Evaluated_values                = eval_excited[:2] + 1.0j * eval_excited[2:]
    energy_qEOM[ex + 1]             = (Evaluated_values[0] / Evaluated_values[1]).real

# fqEOM
for ex in range(N_state - 1):
    fO  = mf_qEOM.gen_filtered_state_transfer_op(excitation_level   = CAS_match[ex],
                                                 Normalize          = True,
                                                 Estimator          = Estimator())
                                                                    
    Effective_fH                    = fO.conjugate().transpose() @ mf_qc.H @ fO
    Op_Re, Op_Im                    = Source.Pauli_Decompose(Effective_fH)
    Filtered_Ops, eval_excited      = Source.zero_operator_filter([Op_Re, Op_Im], 2 * mf_qc.N_orb)
    eval_excited[eval_excited > 0]  = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, Filtered_Ops, Estimator())
    energy_fqEOM[ex + 1]            = (eval_excited[0] + 1.0j * eval_excited[1]).real

np.save(f"/Data/H2_Large/Energy/{dist}_CASCI_MATCH_qEOM_Diag", energy_qEOM_diag)
np.save(f"/Data/H2_Large/Energy/{dist}_CASCI_MATCH_qEOM"     , energy_qEOM)
np.save(f"/Data/H2_Large/Energy/{dist}_CASCI_MATCH_fqEOM"    , energy_fqEOM)