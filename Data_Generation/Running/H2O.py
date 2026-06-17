from qiskit.primitives import StatevectorEstimator as Estimator
from pyqake.lib.Transpile import PySCFTranspiler
from pyqake.ansatz import UCC, qEOM
from pyscf import gto,scf,lib
import numpy as np
import sys


lib.num_threads(1)

x   = np.round(float(sys.argv[1]),1)
cs  = np.cos(np.pi * 104.5 / 360)
sn  = np.sin(np.pi * 104.5 / 360)

print(f"Distance {x} Angstrom Start")
mol             = gto.M(atom   = f'O 0 0 0; H {x * sn} 0 {x * cs}; H {-x * sn} 0 {x * cs}', 
                        basis  = 'sto3g',
                        charge = 0,
                        spin   = 0)
mf              = scf.RHF(mol)
mf.max_cycle    = 1000000
mf.chkfile      = f'/Data/H2O/Chkfile/scf_{x}'
mf.run()

uc_mf           = UCC.UCC(PySCFTranspiler(mf, 6, (4, 4)), ex_code='sd', cd_acc=1e-13, spin_symm=True, max_optimize=1000000)
uc_mf.build()
uc_mf.run(Estimator(), optimize_algorithm='COBYLA')

mf_qEOM         = qEOM.qEOM(uc_mf)
e, XY           = mf_qEOM.run(Estimator(), ex_code='sd')

np.save(f"/Data/H2O/Density/{x}", mf.make_rdm1())
np.save(f"/Data/H2O/Res/UCC_{x}", uc_mf.amplitudes)
np.save(f"/Data/H2O/qEOM/E_{x}" , e)
np.save(f"/Data/H2O/qEOM/XY_{x}", XY)