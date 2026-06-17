from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from pyqake.lib.Transpile import PySCFTranspiler
from pyqake.operator.observable import Dipole
from pyqake.ansatz import UCC, qEOM, QSE
from pyqake.operator import Source
from pyscf import gto,scf,lib
import numpy as np
import sys


lib.num_threads(1)

# =====================< Noise Configuration >=====================
# Depolarizing error rates
noise_rank  = int(np.round(float(sys.argv[1]),1))
ERROR_1Q    = [1e-5, 1e-4, 5e-4, 1e-3][noise_rank]
ERROR_2Q    = [1e-4, 1e-3, 5e-3, 1e-2][noise_rank]
SHOTS       = [100000, 10000, 4096, 1000][noise_rank]
# =================================================================

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

x = 0.7

print(f"Distance {x} Angstrom Start")
print(f"Noise Model: 1Q depolarizing = {ERROR_1Q}, 2Q depolarizing = {ERROR_2Q}, Shots = {SHOTS}")

N_state             = 4
energy_qEOM         = np.zeros(N_state)
energy_fqEOM        = np.zeros(N_state)
energy_qEOM_diag    = np.zeros(N_state)
dipole_qEOM         = np.zeros((N_state, N_state, 3))
dipole_fqEOM        = np.zeros((N_state, N_state, 3))

mol                 = gto.M(atom   = f"H 0 0 0; H 0 0 {x}", 
                            basis  = 'sto3g',
                            charge = 0,
                            spin   = 0)
mf                  = scf.RHF(mol)
mf.chkfile          = f'/Data/H2_Noise/Chkfile/scf_{noise_rank}'
mf.max_cycle        = 1000000
mf.run()

uc_mf               = UCC.UCC(PySCFTranspiler(mf), ex_code='sd', cd_acc=1e-13, spin_symm=True)
uc_mf.build()
uc_mf.run(noisy_estimator, transpiler=transpile_for_aer)

energy_qEOM[0]          = uc_mf.energy
energy_fqEOM[0]         = uc_mf.energy
energy_qEOM_diag[0]     = uc_mf.energy

mf_qEOM                 = qEOM.qEOM(uc_mf, property='ee')
e, XY_ee                = mf_qEOM.run(noisy_estimator, ex_code='sd')
energy_qEOM_diag[1:]    = np.sort(e[e > 0])[:N_state-1] + uc_mf.energy

# qEOM
for ex in range(N_state - 1):
    O                               = mf_qEOM.gen_state_transfer_op(excitation_level = ex + 1)                                                   
    Effective_H                     = O.conjugate().transpose() @ uc_mf.H @ O
    Denominator                     = O.conjugate().transpose() @ O
    Op_Re, Op_Im                    = Source.Pauli_Decomposes([Effective_H, Denominator])
    Filtered_Ops, eval_excited      = Source.zero_operator_filter(Op_Re + Op_Im, 2 * uc_mf.N_orb)
    eval_excited[eval_excited > 0]  = uc_mf.Estimator_costfunction(uc_mf.amplitudes, uc_mf.ansatz, Filtered_Ops, noisy_estimator)
    Evaluated_values                = eval_excited[:2] + 1.0j * eval_excited[2:]
    energy_qEOM[ex + 1]             = (Evaluated_values[0] / Evaluated_values[1]).real
print("qEOM Energy Done")

# fqEOM
for ex in range(N_state - 1):
    fO  = mf_qEOM.gen_filtered_state_transfer_op(excitation_level   = ex + 1,
                                             Normalize          = True,
                                             Estimator          = noisy_estimator)
                                                                    
    Effective_fH                    = fO.conjugate().transpose() @ uc_mf.H @ fO
    Op_Re, Op_Im                    = Source.Pauli_Decompose(Effective_fH)
    Filtered_Ops, eval_excited      = Source.zero_operator_filter([Op_Re, Op_Im], 2 * uc_mf.N_orb)
    eval_excited[eval_excited > 0]  = uc_mf.Estimator_costfunction(uc_mf.amplitudes, uc_mf.ansatz, Filtered_Ops, noisy_estimator)
    energy_fqEOM[ex + 1]            = (eval_excited[0] + 1.0j * eval_excited[1]).real
print("fqEOM Energy Done")

# qEOM Dipole
for N_R in range(N_state):
    R_O = mf_qEOM.gen_state_transfer_op(excitation_level        = N_R)
    for N_L in range(N_R + 1):
        L_O = mf_qEOM.gen_state_transfer_op(excitation_level    = N_L)
        _, dipole_qEOM[N_L, N_R, :]     = Dipole.run(mf_qc      = uc_mf, 
                                                     Estimator  = noisy_estimator, 
                                                     L          = L_O,
                                                     R          = R_O,
                                                     mapping    = uc_mf.mapping)
        Op_Re, Op_Im                    = Source.Pauli_Decomposes([L_O.conjugate().transpose() @ L_O, R_O.conjugate().transpose() @ R_O])
        Filtered_Ops, eval_excited      = Source.zero_operator_filter(Op_Re + Op_Im, 2 * uc_mf.N_orb)
        eval_excited[eval_excited > 0]  = uc_mf.Estimator_costfunction(uc_mf.amplitudes, uc_mf.ansatz, Filtered_Ops, noisy_estimator)
        Evaluated_values                = eval_excited[:2] + 1.0j * eval_excited[2:]
        dipole_qEOM[N_L, N_R, :]       /= (Evaluated_values[0] * Evaluated_values[1]).real ** 0.5
print("qEOM Dipole Done")

# fqEOM Dipole
for N_R in range(N_state):
    R_O = mf_qEOM.gen_filtered_state_transfer_op(excitation_level       = N_R,
                                                 Normalize              = True,
                                                 Estimator              = noisy_estimator)
    for N_L in range(N_R + 1):
        L_O = mf_qEOM.gen_filtered_state_transfer_op(excitation_level   = N_L,
                                                     Normalize          = True,
                                                     Estimator          = noisy_estimator)
        _, dipole_fqEOM[N_L, N_R, :] = Dipole.run(mf_qc                 = uc_mf, 
                                                  Estimator             = noisy_estimator,
                                                  L                     = L_O,
                                                  R                     = R_O,
                                                  mapping               = uc_mf.mapping)
print("fqEOM Dipole Done")

np.save(f"/Data/H2_Noise/Density/{noise_rank}", mf.make_rdm1())
np.save(f"/Data/H2_Noise/Res/UCC_{noise_rank}", uc_mf.amplitudes)
np.save(f"/Data/H2_Noise/qEOM/E_{noise_rank}" , e)
np.save(f"/Data/H2_Noise/qEOM/XY_{noise_rank}", XY_ee)
np.save(f"/Data/H2_Noise/Energy/{noise_rank}_qEOM_Diag", energy_qEOM_diag)
np.save(f"/Data/H2_Noise/Energy/{noise_rank}_qEOM"     , energy_qEOM)
np.save(f"/Data/H2_Noise/Energy/{noise_rank}_fqEOM"    , energy_fqEOM)
np.save(f"/Data/H2_Noise/Dipole/{noise_rank}_qEOM" , dipole_qEOM)
np.save(f"/Data/H2_Noise/Dipole/{noise_rank}_fqEOM", dipole_fqEOM)


# 3. QSE
mf_qse          = QSE.QSE(PySCFTranspiler(mf), amplitudes=uc_mf.amplitudes)
mf_qse.build()
mf_qse.ansatz   = uc_mf.ansatz
e_QSE, XY_QSE   = mf_qse.run(noisy_estimator)

# 4. Restore QSE Operator and Obtain Energy
E_QSE   = np.zeros_like(e_QSE)
for ex in range(len(e_QSE)):
    O                               = mf_qse.gen_state_transfer_op(excitation_level = ex)                                                   
    Effective_H                     = O.conjugate().transpose() @ mf_qse.H @ O
    Denominator                     = O.conjugate().transpose() @ O
    Op_Re, Op_Im                    = Source.Pauli_Decomposes([Effective_H, Denominator])
    Filtered_Ops, eval_excited      = Source.zero_operator_filter(Op_Re + Op_Im, 2 * mf_qse.N_orb)
    eval_excited[eval_excited > 0]  = mf_qse.Estimator_costfunction(mf_qse.amplitudes, mf_qse.ansatz, Filtered_Ops, noisy_estimator)
    Evaluated_values                = eval_excited[:2] + 1.0j * eval_excited[2:]
    E_QSE[ex]                       = (Evaluated_values[0] / Evaluated_values[1]).real

# 5. Obtain Dipole Moments via restored QSE Operator
D_QSE   = np.zeros((len(e_QSE), len(e_QSE), 3))
for N_R in range(len(e_QSE)):
    R_O = mf_qse.gen_state_transfer_op(excitation_level        = N_R)
    for N_L in range(len(e_QSE)):
        L_O = mf_qse.gen_state_transfer_op(excitation_level    = N_L)
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

np.save(f"/Data/H2_Noise/Energy/{noise_rank}_QSE" , E_QSE)
np.save(f"/Data/H2_Noise/Dipole/{noise_rank}_QSE" , D_QSE)
