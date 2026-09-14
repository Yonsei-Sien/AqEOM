from pyqake.lib.Transpile import PySCFTranspiler
from pyqake.operator import Source
from pyqake.lib import MO2AO
from pyscf import gto, scf, dft
import numpy as np
import sys


master_dir  = '***'
system      = 'H2O'
XCs         = ['svwn', 'blyp', 'pbe', 'b97', 'r2SCAN', 'tpss', 'm06l', 'b3lyp', 'r2scan0', 'pbe0']

for xc in XCs:
    dist            = np.round(float(sys.argv[1]),1)
    N_states        = 3
    ss_AqEOM        = np.load(f"{master_dir}/Data/{system}/Spin_Square/{dist}_fqEOM.npy")

    Spin_match      = np.where(ss_AqEOM < 1)[0]
    N_target_state  = Spin_match.shape[0]
    N_states        = min(N_states, N_target_state)
    Spin_match      = Spin_match[:N_states]

    energy_CAS      = np.zeros((N_states))
    energy_qEOM     = np.zeros((N_states))
    energy_fqEOM    = np.zeros((N_states))

    print(f"\n>>>>>> {dist} AA Start")

    # HF
    mol, mf_chk     = scf.chkfile.load_scf(f"{master_dir}/Data/{system}/Chkfile/scf_{dist}")
    mf_hf           = scf.RHF(mol)
    mf_hf.__dict__.update(mf_chk)
    Trans           = PySCFTranspiler(mf_hf, 6, (4, 4))

    mf              = dft.RKS(mol)
    mf.xc           = xc
    mf.grids.level  = 4

    for ex in range(N_states):
        RDM_CAS             = np.load(f"{master_dir}/Data/{system}/Density/CASCI_{ex}_{dist}.npy")
        RDM_qEOM            = np.load(f"{master_dir}/Data/{system}/Density/qEOM_{Spin_match[ex]}_{dist}.npy")
        RDM_fqEOM           = np.load(f"{master_dir}/Data/{system}/Density/fqEOM_{Spin_match[ex]}_{dist}.npy")
        res_MO_RDM_CAS      = MO2AO.restore_full_space(RDM_CAS, Trans)
        res_MO_RDM_qEOM     = MO2AO.restore_full_space(RDM_qEOM, Trans)
        res_MO_RDM_fqEOM    = MO2AO.restore_full_space(RDM_fqEOM, Trans)
        res_RDM_CAS         = MO2AO.run(res_MO_RDM_CAS, mf_hf.mo_coeff)
        res_RDM_qEOM        = MO2AO.run(res_MO_RDM_qEOM, mf_hf.mo_coeff)
        res_RDM_fqEOM       = MO2AO.run(res_MO_RDM_fqEOM, mf_hf.mo_coeff)
        energy_CAS[ex]      = mf.energy_tot(res_RDM_CAS)
        energy_qEOM[ex]     = mf.energy_tot(res_RDM_qEOM)
        energy_fqEOM[ex]    = mf.energy_tot(res_RDM_fqEOM)
    print(f"{xc} Done")

    np.save(f"{master_dir}/Data/{system}/Energy/{dist}_CAS_{xc}"              , energy_CAS)
    np.save(f"{master_dir}/Data/{system}/Energy/{dist}_CASCI_MATCH_qEOM_{xc}" , energy_qEOM)
    np.save(f"{master_dir}/Data/{system}/Energy/{dist}_CASCI_MATCH_fqEOM_{xc}", energy_fqEOM)