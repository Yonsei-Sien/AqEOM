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

###=============================================<<< Setup

# N_ss_states : Number of states to be investigated the spin square
#    N_states : Number of states to be quantity-evaluated

#   HF : Hartree-Fock
#  UCC : Unitary coupled cluster (here, singles and doubles)
# qEOM : Quantum equation-of-motion
#   SS : Spin Square
#    E : Energy, Direct expectation value
#    D : Dipole moments, Direct expectation value
#    V : Vacuum annihilation condition, Direct expectation value
#  RDM : Reduced density matrix

# !!! Please handle active space and mol manually !!!

dist        = np.round(float(sys.argv[1]),1) # Bond distance
master_dir  = '***'
system      = 'H2O'

N_ss_states = 8
N_states    = 3

do_HF       = False
do_UCC      = False
do_qEOM     = False
do_SS       = False
do_E        = False
do_D        = False
do_V        = True
do_RDM      = False

###=============================================<<< Calculation

# 1. HF
mol             = None
mf              = None
cs              = np.cos(np.pi * 104.5 / 360)
sn              = np.sin(np.pi * 104.5 / 360)
if do_HF:
    mol         = gto.M(atom   = f'O 0 0 0; H {dist * sn} 0 {dist * cs}; H {-dist * sn} 0 {dist * cs}', 
                        basis  = 'sto3g',
                        charge = 0,
                        spin   = 0)
    mf          = scf.RHF(mol)
    mf.chkfile  = f'{master_dir}/Data/{system}/Chkfile/scf_{dist}'
    mf.kernel()
else:
    mol, mf_chk = scf.chkfile.load_scf(f"{master_dir}/Data/{system}/Chkfile/scf_{dist}")
    mf          = scf.RHF(mol)
    mf.__dict__.update(mf_chk)
Transfiled      = PySCFTranspiler(mf, 6, (4, 4)) # !!! Active Space Here !!!
print("HF Done")

# 2. UCCSD
mf_ucc          = None
if do_UCC:
    mf_qc       = UCC(classic_object=Transfiled, spin_symm=True)
    mf_qc.build()
    mf_qc.run(Estimator())
    np.save(f"{master_dir}/Data/{system}/Density/{dist}", mf.make_rdm1())
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
    e, XY       = mf_qEOM.run(Estimator(), ex_code='sd')
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
        ss_qEOM[ex]             = mf_qc.spin_square(Estimator(), transpiler=qEOM_transpiler)[0]
        Op_qEOM_denominator     = O.conjugate().transpose() @ O
        De_Re, De_Im            = Source.Pauli_Decompose(Op_qEOM_denominator)
        filt_de, eval_de        = Source.zero_operator_filter([De_Re, De_Im], mf_qc.N_orb * 2)
        eval_de[eval_de > 0]    = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, filt_de, Estimator())
        ss_qEOM[ex]            /= (eval_de[0] + 1.0j * eval_de[1]).real
    print("qEOM SS Done")

    for ex in range(N_ss_states):
        fO  = mf_qEOM.gen_filtered_state_transfer_op(excitation_level   = ex,
                                                Normalize          = True,
                                                Estimator          = Estimator())
        def fqEOM_transpiler(ansatz, Ops):
            transpiled_operator = [SparsePauliOp.simplify(fO.conjugate().transpose() @ Op @ fO) for Op in Ops]
            return ansatz, transpiled_operator                                        
        ss_AqEOM[ex]    = mf_qc.spin_square(Estimator(), transpiler=fqEOM_transpiler)[0]
    print("AqEOM SS Done")

    np.save(f"{master_dir}/Data/{system}/Spin_Square/{dist}_qEOM" , ss_qEOM)
    np.save(f"{master_dir}/Data/{system}/Spin_Square/{dist}_fqEOM", ss_AqEOM)
else:
    ss_qEOM     = np.load(f"{master_dir}/Data/{system}/Spin_Square/{dist}_qEOM.npy")
    ss_AqEOM    = np.load(f"{master_dir}/Data/{system}/Spin_Square/{dist}_fqEOM.npy")

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
        eval_excited[eval_excited > 0]  = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, Filtered_Ops, Estimator())
        Evaluated_values                = eval_excited[:2] + 1.0j * eval_excited[2:]
        energy_qEOM[ex + 1]             = (Evaluated_values[0] / Evaluated_values[1]).real
    print("qEOM Energy Done")

    for ex in range(N_states - 1):
        fO  = mf_qEOM.gen_filtered_state_transfer_op(excitation_level   = Spin_match[ex + 1],
                                                     Normalize          = True,
                                                     Estimator          = Estimator())
                                                                        
        Effective_fH                    = fO.conjugate().transpose() @ mf_qc.H @ fO
        Op_Re, Op_Im                    = Source.Pauli_Decompose(Effective_fH)
        Filtered_Ops, eval_excited      = Source.zero_operator_filter([Op_Re, Op_Im], 2 * mf_qc.N_orb)
        eval_excited[eval_excited > 0]  = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, Filtered_Ops, Estimator())
        energy_fqEOM[ex + 1]            = (eval_excited[0] + 1.0j * eval_excited[1]).real
    print("AqEOM Energy Done")

    np.save(f"{master_dir}/Data/{system}/Energy/{dist}_CASCI_MATCH_qEOM_Diag", energy_qEOM_diag)
    np.save(f"{master_dir}/Data/{system}/Energy/{dist}_CASCI_MATCH_qEOM"     , energy_qEOM)
    np.save(f"{master_dir}/Data/{system}/Energy/{dist}_CASCI_MATCH_fqEOM"    , energy_fqEOM)

# 6. Dipole moments
dipole_qEOM     = np.zeros((N_states, N_states, 3))
dipole_AqEOM    = np.zeros((N_states, N_states, 3))
if do_D:
    for N_R in range(N_states):
        R_O = mf_qEOM.gen_state_transfer_op(excitation_level        = Spin_match[N_R])
        for N_L in range(N_R + 1):
            L_O = mf_qEOM.gen_state_transfer_op(excitation_level    = Spin_match[N_L])
            _, dipole_qEOM[N_L, N_R, :]     = Dipole.run(mf_qc      = mf_qc, 
                                                         Estimator  = Estimator(),
                                                         L          = L_O,
                                                         R          = R_O,
                                                         mapping    = mf_qc.mapping)
            Op_Re, Op_Im                    = Source.Pauli_Decomposes([L_O.conjugate().transpose() @ L_O, R_O.conjugate().transpose() @ R_O])
            Filtered_Ops, eval_excited      = Source.zero_operator_filter(Op_Re + Op_Im, 2 * mf_qc.N_orb)
            eval_excited[eval_excited > 0]  = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, Filtered_Ops, Estimator())
            Evaluated_values                = eval_excited[:2] + 1.0j * eval_excited[2:]
            dipole_qEOM[N_L, N_R, :]       /= (Evaluated_values[0] * Evaluated_values[1]).real ** 0.5
    print("qEOM Dipole Done")

    for N_R in range(N_states):
        R_O = mf_qEOM.gen_filtered_state_transfer_op(excitation_level       = Spin_match[N_R],
                                                    Normalize               = True,
                                                    Estimator               = Estimator())
        for N_L in range(N_R + 1):
            L_O = mf_qEOM.gen_filtered_state_transfer_op(excitation_level   = Spin_match[N_L],
                                                        Normalize           = True,
                                                        Estimator           = Estimator())
            _, dipole_AqEOM[N_L, N_R, :] = Dipole.run(mf_qc                 = mf_qc, 
                                                     Estimator              = Estimator(),
                                                     L                      = L_O,
                                                     R                      = R_O,
                                                     mapping                = mf_qc.mapping)
    print("AqEOM Dipole Done")
    np.save(f"{master_dir}/Data/{system}/Dipole/{dist}_CASCI_MATCH_qEOM" , dipole_qEOM)
    np.save(f"{master_dir}/Data/{system}/Dipole/{dist}_CASCI_MATCH_fqEOM", dipole_AqEOM)

# 7. Vacuum annihilation condition
VAC_qEOM    = np.zeros((N_states, N_states), dtype=complex)
VAC_AqEOM   = np.zeros((N_states, N_states), dtype=complex)
if do_V:
    for Rex in range(N_states):
        R_O = mf_qEOM.gen_state_transfer_op(excitation_level = Spin_match[Rex])    
        for Lex in range(Rex + 1):
            L_O                             = mf_qEOM.gen_state_transfer_op(excitation_level = Spin_match[Lex])
            L_O_adjoint                     = L_O.conjugate().transpose()
            Op_VAC                          = L_O_adjoint @ R_O
            Op_Re, Op_Im                    = Source.Pauli_Decompose(Op_VAC)
            Filtered_Ops, eval_excited      = Source.zero_operator_filter([Op_Re, Op_Im], 2 * mf_qc.N_orb)
            eval_excited[eval_excited > 0]  = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, Filtered_Ops, Estimator())
            Evaluated_values                = eval_excited[0] + 1.0j * eval_excited[1]
            VAC_qEOM[Lex, Rex]              = Evaluated_values
            VAC_qEOM[Rex, Lex]              = np.conjugate(Evaluated_values)
    print("qEOM VAC Done")

    for Rex in range(N_states):
        RfO  = mf_qEOM.gen_filtered_state_transfer_op(excitation_level      = Spin_match[Rex],
                                                      Normalize             = True,
                                                      Estimator             = Estimator())  
        for Lex in range(Rex + 1):
            LfO  = mf_qEOM.gen_filtered_state_transfer_op(excitation_level  = Spin_match[Lex],
                                                        Normalize           = True,
                                                        Estimator           = Estimator())  
            LfO_adjoint                     = LfO.conjugate().transpose()
            Op_VAC                          = LfO_adjoint @ RfO
            Op_Re, Op_Im                    = Source.Pauli_Decompose(Op_VAC)
            Filtered_Ops, eval_excited      = Source.zero_operator_filter([Op_Re, Op_Im], 2 * mf_qc.N_orb)
            eval_excited[eval_excited > 0]  = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, Filtered_Ops, Estimator())
            Evaluated_values                = eval_excited[0] + 1.0j * eval_excited[1]
            VAC_AqEOM[Lex, Rex]             = Evaluated_values
            VAC_AqEOM[Rex, Lex]             = np.conjugate(Evaluated_values)
    print("AqEOM VAC Done")

    np.save(f"{master_dir}/Data/{system}/VAC/{dist}_CASCI_MATCH_qEOM" , VAC_qEOM)
    np.save(f"{master_dir}/Data/{system}/VAC/{dist}_CASCI_MATCH_fqEOM", VAC_AqEOM)

# 8. Reduced density matrix
if do_RDM:
    for ex in range(N_states):
        O   = mf_qEOM.gen_state_transfer_op(excitation_level = Spin_match[ex])
        def qEOM_transpiler(ansatz, Ops):
            transpiled_operator = [SparsePauliOp.simplify(O.conjugate().transpose() @ Op @ O) for Op in Ops]
            return ansatz, transpiled_operator
                                                                        
        RDM_qEOM                = mf_qc.make_rdm(Estimator(), Degree=1, Spin_trace=True, Spin_simplify=True, transpiler=qEOM_transpiler if ex else None)
        Op_qEOM_denominator     = O.conjugate().transpose() @ O
        De_Re, De_Im            = Source.Pauli_Decompose(Op_qEOM_denominator)
        filt_de, eval_de        = Source.zero_operator_filter([De_Re, De_Im], mf_qc.N_orb * 2)
        eval_de[eval_de > 0]    = mf_qc.Estimator_costfunction(mf_qc.amplitudes, mf_qc.ansatz, filt_de, Estimator())
        np.save(f"{master_dir}/Data/{system}/Density/qEOM_{Spin_match[ex]}_{dist}" , RDM_qEOM / (eval_de[0] + 1.0j * eval_de[1]).real)
    print("qEOM RDM Done")

    for ex in range(N_states):
        fO  = mf_qEOM.gen_filtered_state_transfer_op(excitation_level   = Spin_match[ex],
                                                     Normalize          = True,
                                                     Estimator          = Estimator())
        def AqEOM_transpiler(ansatz, Ops):
            transpiled_operator = [SparsePauliOp.simplify(fO.conjugate().transpose() @ Op @ fO) for Op in Ops]
            return ansatz, transpiled_operator
                                                                        
        RDM_AqEOM   = mf_qc.make_rdm(Estimator(), Degree=1, Spin_trace=True, Spin_simplify=True, transpiler=AqEOM_transpiler if ex else None)
        np.save(f"{master_dir}/Data/{system}/Density/fqEOM_{Spin_match[ex]}_{dist}", RDM_AqEOM)
    print("AqEOM RDM Done")

# 9. Hermitian Contamination
Contam  = np.zeros((N_states, 3)) # O, H, A --> C = H/O
C_H     = np.zeros(N_states)
for ex in range(N_states):
    O               = mf_qEOM.gen_state_transfer_op(excitation_level = Spin_match[ex])    
    H               = (O + O.conjugate().transpose()).to_matrix() / 2
    A               = (O - O.conjugate().transpose()).to_matrix() / 2
    Contam[ex, 0]   = np.linalg.norm(O.to_matrix())
    Contam[ex, 1]   = np.linalg.norm(H)
    Contam[ex, 2]   = np.linalg.norm(A)
    C_H[ex]         = Contam[ex, 1] / Contam[ex, 0]
print("Hermitian Contamination Done")
np.save(f"{master_dir}/Data/{system}/Hermitian_Contamination/{dist}", Contam)
np.save(f"{master_dir}/Data/{system}/Hermitian_Contamination/C_{dist}", C_H)