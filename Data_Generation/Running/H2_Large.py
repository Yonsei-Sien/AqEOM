from qiskit.primitives import StatevectorEstimator as Estimator
from pyqake.lib.Transpile import PySCFTranspiler
from pyqake.ansatz import UCC, qEOM
from pyscf import gto,scf,lib
import numpy as np
import sys


lib.num_threads(1)

x = np.round(float(sys.argv[1]),1)

print(f"Distance {x} Angstrom Start")
mol             = gto.M(atom   = f"H 0 0 0; H 0 0 {x}", 
                        basis  = '6-31g',
                        charge = 0,
                        spin   = 0)
mf              = scf.RHF(mol)
mf.max_cycle    = 1000000
mf.chkfile      = f'/Data/H2_Large/Chkfile/scf_{x}'
mf.run()

uc_mf           = UCC.UCC(PySCFTranspiler(mf), ex_code='sd', cd_acc=1e-13, spin_symm=True)
uc_mf.build()
uc_mf.run(Estimator())

mf_qEOM_ee      = qEOM.qEOM(uc_mf, property='ee')
e_ee, XY_ee     = mf_qEOM_ee.run(Estimator(), ex_code='sd')

np.save(f"/Data/H2_Large/Density/{x}", mf.make_rdm1())
np.save(f"/Data/H2_Large/Res/UCC_{x}", uc_mf.amplitudes)
np.save(f"/Data/H2_Large/qEOM/E_{x}" , e_ee)
np.save(f"/Data/H2_Large/qEOM/XY_{x}", XY_ee)