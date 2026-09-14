from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp
from pyqake.lib.Transpile import PySCFTranspiler
from pyqake.operator.observable import Dipole
from pyqake.operator import Source
from pyqake.ansatz.UCC import UCC
from pyqake.ansatz import qEOM, QSE
from pyscf import gto, scf
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

dist        = 0.7
master_dir  = '***'
system      = 'H2_Noise'

N_ss_states = 8
N_states    = 3

do_HF       = True
do_UCC      = True
do_qEOM     = True
do_SS       = True
do_E        = True
do_D        = True

###=============================================<<< Noise Model

# Depolarizing error rates
noise_rank  = int(np.round(float(sys.argv[1]),1))
ERROR_1Q    = [1e-5, 1e-4, 5e-4, 1e-3][noise_rank]
ERROR_2Q    = [1e-4, 1e-3, 5e-3, 1e-2][noise_rank]
SHOTS       = [100000, 10000, 4096, 1000][noise_rank]

# Build noise model
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depolarizing_error(ERROR_1Q, 1), ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'x', 'y', 'z', 'h', 's', 't', 'id'])
noise_model.add_all_qubit_quantum_error(depolarizing_error(ERROR_2Q, 2), ['cx', 'cz', 'cy', 'swap', 'ecr'])

# Create noisy Estimator
noisy_estimator = AerEstimator(
    options={
        "backend_options": {
            "noise_model": noise_model,
        },
        "run_options": {
            "shots": SHOTS,
        },
    }
)

# Transpiler: decompose PauliEvolutionGate into basic gates for Aer compatibility
aer_backend = AerSimulator(noise_model=noise_model)
pm = generate_preset_pass_manager(optimization_level=1, backend=aer_backend)

def transpile_for_aer(ansatz, operators):
    """Transpile circuit to basic gates that Aer can execute."""
    transpiled_ansatz = pm.run(ansatz)
    return transpiled_ansatz, operators

###=============================================<<< Calculation

# 1. HF
mol             = None
mf              = None
if do_HF:
    mol         = gto.M(atom=f'H 0 0 0; H 0 0 0.7', basis='sto3g', spin=0, charge=0)
    mf          = scf.RHF(mol)
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
    mf_qc       = UCC(classic_object=Transfiled, spin_symm=True)
    mf_qc.build()
    mf_qc.run(noisy_estimator, transpiler=transpile_for_aer)
    np.save(f"{master_dir}/Data/{system}/Res/UCC_{dist}", mf_qc.amplitudes)
else:
    ucc_amp     = np.load(f"{master_dir}/Data/{system}/Res/UCC_{dist}.npy")
    mf_qc       = UCC(classic_object=Transfiled, spin_symm=True, amplitudes=ucc_amp)
    mf_qc.build()
print("UCCSD Done")

# 3. qEOM
mf_qEOM         = None
if do_qEOM:
    mf_qEOM     = qEOM.qEOM(mf_qc)
    e, XY       = mf_qEOM.run(noisy_estimator, ex_code='sd')
    np.save(f"{master_dir}/Data/{system}/qEOM/E_{dist}" , e)
    np.save(f"{master_dir}/Data/{system}/qEOM/XY_{dist}", XY)
else:
    mf_qEOM     = qEOM.qEOM(mf_qc)
    mf_qEOM.e   = np.load(f"{master_dir}/Data/{system}/qEOM/E_{dist}.npy")
    mf_qEOM.XY  = np.load(f"{master_dir}/Data/{system}/qEOM/XY_{dist}.npy")
    mf_qEOM._gen_ops()
print("qEOM Done")

# 3-1. Modify State Number Variable - To avoid index error, not artificial work...
N_qEOM_state    = (mf_qEOM.e.shape[0] / 2) + 1      # As the qEOM returns double of the excited state (excitation and de-excitation), +1 is GS
N_ss_states     = int(min(N_ss_states, N_qEOM_state))

# 4. Spin Square
ss_qEOM         = np.zeros(N_ss_states)
ss_AqEOM        = np.zeros(N_ss_states)
if do_SS:
    for ex in range(N_ss_states):
        O   = mf_qEOM.gen_state_transfer_op(excitation_level = ex)
        def qEOM_transpiler(ansatz, Ops):
            transpiled_operator = [SparsePauliOp.simplify(O.conjugate().transpose() @ Op @ O) for Op in Ops]
            return ansatz, transpiled_operator
        ss_qEOM[ex]             = mf_qc.spin_square(noisy_estimator, transpiler=qEOM_transpiler)[0]
        Op_qEOM_denominator     = O.conjugate().transpose() @ O
        De_Re, De_Im            = Source.Pauli_Decompose(Op_qEOM_denominator)
        filt_de, eval_de        = Source.zero_operator_filter([De_Re, De_Im], mf_qc.N_orb * 2)
        eval_de[eval_de > 0]    = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, filt_de, noisy_estimator)
        ss_qEOM[ex]            /= (eval_de[0] + 1.0j * eval_de[1]).real
    print("qEOM SS Done")

    for ex in range(N_ss_states):
        fO  = mf_qEOM.gen_filtered_state_transfer_op(excitation_level   = ex,
                                                Normalize          = True,
                                                Estimator          = noisy_estimator)
        def fqEOM_transpiler(ansatz, Ops):
            transpiled_operator = [SparsePauliOp.simplify(fO.conjugate().transpose() @ Op @ fO) for Op in Ops]
            return ansatz, transpiled_operator                                        
        ss_AqEOM[ex]    = mf_qc.spin_square(noisy_estimator, transpiler=fqEOM_transpiler)[0]
    print("AqEOM SS Done")

    np.save(f"{master_dir}/Data/{system}/Spin_Square/{dist}_qEOM" , ss_qEOM)
    np.save(f"{master_dir}/Data/{system}/Spin_Square/{dist}_fqEOM", ss_AqEOM)
else:
    ss_qEOM     = np.load(f"{master_dir}/Data/{system}/Spin_Square/{noise_rank}_qEOM.npy" , ss_qEOM)
    ss_AqEOM    = np.load(f"{master_dir}/Data/{system}/Spin_Square/{noise_rank}_fqEOM.npy", ss_AqEOM)

# 4-1. Modify State Number Variable - Select target spin state cases
Spin_match      = np.where(ss_AqEOM < 1)[0]
N_target_state  = Spin_match.shape[0]
N_states        = min(N_states, N_target_state)
Spin_match      = Spin_match[:N_states]

# 5. Energy
energy_qEOM         = np.zeros(N_states)
energy_fqEOM        = np.zeros(N_states)
energy_qEOM_diag    = np.zeros(N_states)
if do_E:
    energy_qEOM[0]          = mf_qc.energy
    energy_fqEOM[0]         = mf_qc.energy
    energy_qEOM_diag[0]     = mf_qc.energy
    energy_qEOM_diag[1:]    = np.sort(mf_qEOM.e[mf_qEOM.e > 0])[Spin_match[1:] - 1] + mf_qc.energy

    for ex in range(N_states - 1):
        O                               = mf_qEOM.gen_state_transfer_op(excitation_level = Spin_match[ex + 1])
        Effective_H                     = O.conjugate().transpose() @ mf_qc.H @ O
        Denominator                     = O.conjugate().transpose() @ O
        Op_Re, Op_Im                    = Source.Pauli_Decomposes([Effective_H, Denominator])
        Filtered_Ops, eval_excited      = Source.zero_operator_filter(Op_Re + Op_Im, 2 * mf_qc.N_orb)
        eval_excited[eval_excited > 0]  = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, Filtered_Ops, noisy_estimator)
        Evaluated_values                = eval_excited[:2] + 1.0j * eval_excited[2:]
        energy_qEOM[ex + 1]             = (Evaluated_values[0] / Evaluated_values[1]).real
    print("qEOM Energy Done")

    for ex in range(N_states - 1):
        fO  = mf_qEOM.gen_filtered_state_transfer_op(excitation_level   = Spin_match[ex + 1],
                                                     Normalize          = True,
                                                     Estimator          = noisy_estimator)
                                                                        
        Effective_fH                    = fO.conjugate().transpose() @ mf_qc.H @ fO
        Op_Re, Op_Im                    = Source.Pauli_Decompose(Effective_fH)
        Filtered_Ops, eval_excited      = Source.zero_operator_filter([Op_Re, Op_Im], 2 * mf_qc.N_orb)
        eval_excited[eval_excited > 0]  = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, Filtered_Ops, noisy_estimator)
        energy_fqEOM[ex + 1]            = (eval_excited[0] + 1.0j * eval_excited[1]).real
    print("AqEOM Energy Done")

    np.save(f"{master_dir}/Data/{system}/Energy/{noise_rank}_CASCI_MATCH_qEOM_Diag", energy_qEOM_diag)
    np.save(f"{master_dir}/Data/{system}/Energy/{noise_rank}_CASCI_MATCH_qEOM"     , energy_qEOM)
    np.save(f"{master_dir}/Data/{system}/Energy/{noise_rank}_CASCI_MATCH_fqEOM"    , energy_fqEOM)

# 6. Dipole moments
dipole_qEOM     = np.zeros((N_states, N_states, 3))
dipole_AqEOM    = np.zeros((N_states, N_states, 3))
if do_D:
    for N_R in range(N_states):
        R_O = mf_qEOM.gen_state_transfer_op(excitation_level        = Spin_match[N_R])
        for N_L in range(N_R + 1):
            L_O = mf_qEOM.gen_state_transfer_op(excitation_level    = Spin_match[N_L])
            _, dipole_qEOM[N_L, N_R, :]     = Dipole.run(mf_qc      = mf_qc, 
                                                         Estimator  = noisy_estimator,
                                                         L          = L_O,
                                                         R          = R_O,
                                                         mapping    = mf_qc.mapping)
            Op_Re, Op_Im                    = Source.Pauli_Decomposes([L_O.conjugate().transpose() @ L_O, R_O.conjugate().transpose() @ R_O])
            Filtered_Ops, eval_excited      = Source.zero_operator_filter(Op_Re + Op_Im, 2 * mf_qc.N_orb)
            eval_excited[eval_excited > 0]  = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, Filtered_Ops, noisy_estimator)
            Evaluated_values                = eval_excited[:2] + 1.0j * eval_excited[2:]
            dipole_qEOM[N_L, N_R, :]       /= (Evaluated_values[0] * Evaluated_values[1]).real ** 0.5
    print("qEOM Dipole Done")

    for N_R in range(N_states):
        R_O = mf_qEOM.gen_filtered_state_transfer_op(excitation_level       = Spin_match[N_R],
                                                    Normalize               = True,
                                                    Estimator               = noisy_estimator)
        for N_L in range(N_R + 1):
            L_O = mf_qEOM.gen_filtered_state_transfer_op(excitation_level   = Spin_match[N_L],
                                                        Normalize           = True,
                                                        Estimator           = noisy_estimator)
            _, dipole_AqEOM[N_L, N_R, :] = Dipole.run(mf_qc                 = mf_qc, 
                                                     Estimator              = noisy_estimator,
                                                     L                      = L_O,
                                                     R                      = R_O,
                                                     mapping                = mf_qc.mapping)
    print("AqEOM Dipole Done")
    np.save(f"{master_dir}/Data/{system}/Dipole/{noise_rank}_CASCI_MATCH_qEOM" , dipole_qEOM)
    np.save(f"{master_dir}/Data/{system}/Dipole/{noise_rank}_CASCI_MATCH_fqEOM", dipole_AqEOM)

# 7. QSE
mf_qse          = QSE.QSE(PySCFTranspiler(mf), amplitudes=mf_qc.amplitudes)
mf_qse.build()
mf_qse.ansatz   = mf_qc.ansatz
e_QSE, XY_QSE   = mf_qse.run(noisy_estimator)

# 4. Restore QSE Operator and Obtain Energy
E_QSE   = np.zeros_like(e_QSE)
for ex in range(N_states):
    O                               = mf_qse.gen_state_transfer_op(excitation_level = Spin_match[ex])                                                   
    Effective_H                     = O.conjugate().transpose() @ mf_qse.H @ O
    Denominator                     = O.conjugate().transpose() @ O
    Op_Re, Op_Im                    = Source.Pauli_Decomposes([Effective_H, Denominator])
    Filtered_Ops, eval_excited      = Source.zero_operator_filter(Op_Re + Op_Im, 2 * mf_qse.N_orb)
    eval_excited[eval_excited > 0]  = mf_qse.Estimator_costfunction(mf_qse.amplitudes, mf_qse.ansatz, Filtered_Ops, noisy_estimator)
    Evaluated_values                = eval_excited[:2] + 1.0j * eval_excited[2:]
    E_QSE[ex]                       = (Evaluated_values[0] / Evaluated_values[1]).real

# 5. Obtain Dipole Moments via restored QSE Operator
D_QSE   = np.zeros((N_states, N_states, 3))
for N_R in range(N_states):
    R_O = mf_qse.gen_state_transfer_op(excitation_level        = Spin_match[N_R])
    for N_L in range(N_states):
        L_O = mf_qse.gen_state_transfer_op(excitation_level    = Spin_match[N_L])
        _, D_QSE[N_L, N_R, :]           = Dipole.run(mf_qc      = mf_qse, 
                                                     Estimator  = noisy_estimator,
                                                     L          = L_O,
                                                     R          = R_O,
                                                     mapping    = mf_qse.mapping)
        Op_Re, Op_Im                    = Source.Pauli_Decomposes([L_O.conjugate().transpose() @ L_O, R_O.conjugate().transpose() @ R_O])
        Filtered_Ops, eval_excited      = Source.zero_operator_filter(Op_Re + Op_Im, 2 * mf_qse.N_orb)
        eval_excited[eval_excited > 0]  = mf_qse.Estimator_costfunction(mf_qse.amplitudes, mf_qse.ansatz, Filtered_Ops, noisy_estimator)
        Evaluated_values                = eval_excited[:2] + 1.0j * eval_excited[2:]
        D_QSE[N_L, N_R, :]             /= (Evaluated_values[0] * Evaluated_values[1]).real ** 0.5
print("QSE Dipole Done")

np.save(f"{master_dir}/Data/{system}/Energy/{noise_rank}_QSE" , E_QSE)
np.save(f"{master_dir}/Data/{system}/Dipole/{noise_rank}_QSE" , D_QSE)