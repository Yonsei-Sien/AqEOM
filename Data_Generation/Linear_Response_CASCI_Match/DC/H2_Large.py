from pyqake.lib.Transpile import PySCFTranspiler
from pyqake.operator import Source
from pyqake.lib import MO2AO
from pyscf import gto, scf, dft
import numpy as np
import sys


dist            = np.round(float(sys.argv[1]),1)
N_state         = 10
CAS_match       = np.where(np.load(f"/Data/H2_Large/Spin_Square/{dist}_fqEOM.npy") < 1)[0]
energy_qEOM     = np.zeros((N_state))
energy_fqEOM    = np.zeros((N_state))

print(f"\n>>>>>> {dist} AA Start")

# HF
mol, mf_chk     = scf.chkfile.load_scf(f"/Data/H2_Large/Chkfile/scf_{dist}")
mf_hf           = scf.RHF(mol)
mf_hf.__dict__.update(mf_chk)
Trans           = PySCFTranspiler(mf_hf)

mf              = dft.RKS(mol)
mf.XC           = 'r2SCAN'
mf.grids.level  = 4

for ex in range(N_state):
    RDM_qEOM            = np.load(f"/Data/H2_Large/Density/qEOM_{CAS_match[ex]}_{dist}.npy")
    RDM_fqEOM           = np.load(f"/Data/H2_Large/Density/fqEOM_{CAS_match[ex]}_{dist}.npy")
    res_MO_RDM_qEOM     = MO2AO.restore_full_space(RDM_qEOM, Trans)
    res_MO_RDM_fqEOM    = MO2AO.restore_full_space(RDM_fqEOM, Trans)
    res_RDM_qEOM        = MO2AO.run(res_MO_RDM_qEOM, mf_hf.mo_coeff)
    res_RDM_fqEOM       = MO2AO.run(res_MO_RDM_fqEOM, mf_hf.mo_coeff)
    energy_qEOM[ex]     = mf.energy_tot(res_RDM_qEOM)
    energy_fqEOM[ex]    = mf.energy_tot(res_RDM_fqEOM)
print("Done")

np.save(f"/Data/H2_Large/Energy/{dist}_CASCI_MATCH_qEOM_r2SCAN" , energy_qEOM)
np.save(f"/Data/H2_Large/Energy/{dist}_CASCI_MATCH_fqEOM_r2SCAN", energy_fqEOM)