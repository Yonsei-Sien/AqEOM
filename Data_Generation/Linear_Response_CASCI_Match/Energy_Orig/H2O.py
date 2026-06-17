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
N_state             = 3
CAS_matches         = {0.5: [2, 3], 0.6: [1, 3], 0.7: [1, 3], 0.8: [1, 4], 0.9: [1, 4], 1.0: [1, 4], 1.1: [1, 4], 1.2: [1, 4], 1.3: [1, 4], 1.4: [1, 4], 1.5: [1, 4], 1.6: [1, 4]}
CAS_match           = CAS_matches[dist]
energy_qEOM         = np.zeros(N_state)
energy_fqEOM        = np.zeros(N_state)
energy_qEOM_diag    = np.zeros(N_state)

print(f"\n>>>>>> {dist} AA Start")

# HF
mol, mf_chk             = scf.chkfile.load_scf(f"/Data/H2O/Chkfile/scf_{dist}")
mf                      = scf.RHF(mol)
mf.__dict__.update(mf_chk)

# QC
ucc_amp                 = np.load(f"/Data/H2O/Res/UCC_{dist}.npy")
mf_qc                   = UCC(classic_object = PySCFTranspiler(mf, active_space=6, active_e=(4, 4)), mapping='jordan_wigner', spin_symm=True, amplitudes=ucc_amp)
mf_qc.build()
mf_qc.energy            = mf_qc.energy_tot(Estimator())
print("UCCSD Done")

# qEOM
mf_qEOM                 = qEOM.qEOM(mf_qc)
mf_qEOM.e               = np.load(f"/Data/H2O/qEOM/E_{dist}.npy")
mf_qEOM.XY              = np.load(f"/Data/H2O/qEOM/XY_{dist}.npy")
mf_qEOM._gen_ops()
energy_qEOM_diag[0]     = mf_qc.energy
energy_qEOM_diag[1:]    = np.sort(mf_qEOM.e[mf_qEOM.e > 0])[CAS_match] + mf_qc.energy
print("qEOM Done")

np.save(f"/Data/H2O/Energy/{dist}_CASCI_MATCH_qEOM_Diag", energy_qEOM_diag)