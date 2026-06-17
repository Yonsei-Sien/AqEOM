from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp
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

for ex in range(N_state):
    O   = mf_qEOM.gen_state_transfer_op(excitation_level = ex)
    def qEOM_transpiler(ansatz, Ops):
        transpiled_operator = [SparsePauliOp.simplify(O.conjugate().transpose() @ Op @ O) for Op in Ops]
        return ansatz, transpiled_operator
                                                                    
    RDM_qEOM                = mf_qc.make_rdm(Estimator(), Degree=1, Spin_trace=True, Spin_simplify=True, transpiler=qEOM_transpiler)
    Op_qEOM_denominator     = O.conjugate().transpose() @ O
    De_Re, De_Im            = Source.Pauli_Decompose(Op_qEOM_denominator)
    filt_de, eval_de        = Source.zero_operator_filter([De_Re, De_Im], mf_qc.N_orb * 2)
    eval_de[eval_de > 0]    = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, filt_de, Estimator())
    np.save(f"/Data/H2/Density/qEOM_{ex}_{dist}" , RDM_qEOM / (eval_de[0] + 1.0j * eval_de[1]).real)
print("qEOM RDM Done")

# fqEOM RDM
for ex in range(N_state):
    fO  = mf_qEOM.gen_filtered_state_transfer_op(excitation_level   = ex,
                                             Normalize          = True,
                                             Estimator          = Estimator())
    def fqEOM_transpiler(ansatz, Ops):
        transpiled_operator = [SparsePauliOp.simplify(fO.conjugate().transpose() @ Op @ fO) for Op in Ops]
        return ansatz, transpiled_operator
                                                                    
    RDM_fqEOM   = mf_qc.make_rdm(Estimator(), Degree=1, Spin_trace=True, Spin_simplify=True, transpiler=fqEOM_transpiler)
    np.save(f"/Data/H2/Density/fqEOM_{ex}_{dist}", RDM_fqEOM)
print("fqEOM RDM Done")