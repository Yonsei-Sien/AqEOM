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
CAS_match           = [2, 5, 6]
energy_qEOM_diag    = np.zeros(N_state)

print(f"\n>>>>>> {dist} AA Start")

# HF
mol, mf_chk             = scf.chkfile.load_scf(f"/Data/LiH/Chkfile/scf_{dist}")
mf                      = scf.RHF(mol)
mf.__dict__.update(mf_chk)

# QC
ucc_amp                 = np.load(f"/Data/LiH/Res/UCC_{dist}.npy")
mf_qc                   = UCC(classic_object = PySCFTranspiler(mf, active_space=5, active_e=(1, 1)), mapping='jordan_wigner', spin_symm=True, amplitudes=ucc_amp)
mf_qc.build()
mf_qc.energy            = mf_qc.energy_tot(Estimator())
print("UCCSD Done")

# qEOM
mf_qEOM                 = qEOM.qEOM(mf_qc)
mf_qEOM.e               = np.load(f"/Data/LiH/qEOM/E_{dist}.npy")
mf_qEOM.XY              = np.load(f"/Data/LiH/qEOM/XY_{dist}.npy")
mf_qEOM._gen_ops()
energy_qEOM_diag[0]     = mf_qc.energy
energy_qEOM_diag[1:]    = np.sort(mf_qEOM.e[mf_qEOM.e > 0])[[1, 4, 5]] + mf_qc.energy
print("qEOM Done")

np.save(f"/Data/LiH/Energy/{dist}_CASCI_MATCH_qEOM_Diag", energy_qEOM_diag)
