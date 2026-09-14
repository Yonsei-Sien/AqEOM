from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp
from pyqake.lib.Transpile import PySCFTranspiler
from pyqake.operator.observable import Dipole
from pyqake.ansatz import UCC, qEOM, QSE
from pyqake.operator import Source
from pyscf import gto,scf,lib
import numpy as np
import sys


###=============================================<<< Setup

# N_ss_states : Number of states to be investigated the spin square
#    N_states : Number of states to be quantity-evaluated

#   HF : Hartree-Fock
#  UCC : Unitary coupled cluster (here, singles and doubles)
# qEOM : Quantum equation-of-motion
#   SS : Spin Square
#    E : Energy, Direct expectation value
#    D : Dipole moments, Direct expectation value

# !!! Please handle active space and mol manually !!!

dist        = np.round(float(sys.argv[1]),1) # Bond distance
master_dir  = '***'
system      = 'H2_QSE'

N_ss_states = 8
N_states    = 3

do_HF       = True
do_UCC      = True
do_QSE      = True
do_SS       = True
do_E        = True
do_D        = True

###=============================================<<< Calculation

# 1. HF
mol             = None
mf              = None
if do_HF:
    mol         = gto.M(atom=f'H 0 0 0; H 0 0 {dist}', basis='sto3g', spin=0, charge=0)
    mf          = scf.RHF(mol)
    mf.chkfile  = f'{master_dir}/Data/{system}/Chkfile/scf_{dist}'
    mf.kernel()
else:
    mol, mf_chk = scf.chkfile.load_scf(f"{master_dir}/Data/{system}/Chkfile/scf_{dist}")
    mf          = scf.RHF(mol)
    mf.__dict__.update(mf_chk)
Transfiled      = PySCFTranspiler(mf) # !!! Active Space Here !!!
print("HF Done")

# 2. UCCSD
mf_ucc          = None
if do_UCC:
    mf_qc       = UCC.UCC(classic_object=Transfiled, spin_symm=True)
    mf_qc.build()
    mf_qc.run(Estimator())
    np.save(f"{master_dir}/Data/{system}/Res/UCC_{dist}", mf_qc.amplitudes)
else:
    ucc_amp     = np.load(f"{master_dir}/Data/{system}/Res/UCC_{dist}.npy")
    mf_qc       = UCC(classic_object=Transfiled, spin_symm=True, amplitudes=ucc_amp)
    mf_qc.build()
print("UCCSD Done")

# 3. QSE
mf_qse          = QSE.QSE(Transfiled, amplitudes=mf_qc.amplitudes)
mf_qse.build()
mf_qse.ansatz   = mf_qc.ansatz
if do_QSE:
    e_QSE, XY_QSE   = mf_qse.run(Estimator())
    np.save(f"{master_dir}/Data/{system}/QSE/E_{dist}" , e_QSE)
    np.save(f"{master_dir}/Data/{system}/QSE/XY_{dist}", XY_QSE)
else:
    mf_qse.e        = np.load(f"{master_dir}/Data/{system}/QSE/E_{dist}.npy")
    mf_qse.XY       = np.load(f"{master_dir}/Data/{system}/QSE/XY_{dist}.npy")

# 3-1. Modify State Number Variable - To avoid index error, not artificial work...
N_QSE_state     = (mf_qse.e.shape[0])
N_ss_states     = int(min(N_ss_states, N_QSE_state))

# 4. Spin Square
ss_QSE          = np.zeros(N_ss_states)
if do_SS:
    for ex in range(N_ss_states):
        O   = mf_qse.gen_state_transfer_op(excitation_level = ex)
        def qEOM_transpiler(ansatz, Ops):
            transpiled_operator = [SparsePauliOp.simplify(O.conjugate().transpose() @ Op @ O) for Op in Ops]
            return ansatz, transpiled_operator
        ss_QSE[ex]              = mf_qc.spin_square(Estimator(), transpiler=qEOM_transpiler)[0]
        Op_qEOM_denominator     = O.conjugate().transpose() @ O
        De_Re, De_Im            = Source.Pauli_Decompose(Op_qEOM_denominator)
        filt_de, eval_de        = Source.zero_operator_filter([De_Re, De_Im], mf_qc.N_orb * 2)
        eval_de[eval_de > 0]    = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, filt_de, Estimator())
        ss_QSE[ex]             /= (eval_de[0] + 1.0j * eval_de[1]).real
    print("QSE SS Done")
    np.save(f"{master_dir}/Data/{system}/Spin_Square/{dist}_QSE" , ss_QSE)
else:
    ss_QSE     = np.load(f"{master_dir}/Data/{system}/Spin_Square/{dist}_QSE.npy")

# 4-1. Modify State Number Variable - Select target spin state cases
Spin_match      = np.where(ss_QSE < 1)[0]
N_target_state  = Spin_match.shape[0]
N_states        = min(N_states, N_target_state)
Spin_match      = Spin_match[:N_states]

# 5. Energy
E_QSE   = np.zeros(N_states)
for ex in range(N_states):
    O                               = mf_qse.gen_state_transfer_op(excitation_level = Spin_match[ex])                                                   
    Effective_H                     = O.conjugate().transpose() @ mf_qse.H @ O
    Denominator                     = O.conjugate().transpose() @ O
    Op_Re, Op_Im                    = Source.Pauli_Decomposes([Effective_H, Denominator])
    Filtered_Ops, eval_excited      = Source.zero_operator_filter(Op_Re + Op_Im, 2 * mf_qse.N_orb)
    eval_excited[eval_excited > 0]  = mf_qse.Estimator_costfunction(mf_qse.amplitudes, mf_qse.ansatz, Filtered_Ops, Estimator())
    Evaluated_values                = eval_excited[:2] + 1.0j * eval_excited[2:]
    E_QSE[ex]                       = (Evaluated_values[0] / Evaluated_values[1]).real
print("QSE Energy Done")

# 6. Dipole moments
D_QSE   = np.zeros((N_states, N_states, 3))
for N_R in range(N_states):
    R_O = mf_qse.gen_state_transfer_op(excitation_level        = Spin_match[N_R])
    for N_L in range(N_states):
        L_O = mf_qse.gen_state_transfer_op(excitation_level    = Spin_match[N_L])
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

np.save(f"{master_dir}/Data/{system}/Energy/{dist}_QSE", E_QSE)
np.save(f"{master_dir}/Data/{system}/Dipole/{dist}_QSE", D_QSE)