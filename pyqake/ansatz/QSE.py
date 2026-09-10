from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit
from pyqake.operator.observable import RDM, Hamilonian, Spin_Square
from pyqake.operator.gate import Excitation
from pyqake.operator import Source
from pyqake.ansatz import Initial
from pyqake.lib import Transpile
import numpy as np
import scipy
import time


class QSE:
    def __init__(self        , classic_object         ,
                 ex_code='sd', mapping='jordan_wigner', 
                 cd_acc=1e-6 , max_optimize=200000    , amplitudes=[], spin_symm=False):
        # Input Values
        self.classic_object         = classic_object    # Transpiled Object for 2nd Quantization (e.g. Transpile.PySCFTranspiler)
        self.ex_code                = ex_code           # Excitation Codes
        self.mapping                = mapping           # Mapping Algorithm
        self.cd_acc                 = cd_acc            # Cholesty Decomposition Accuracy
        self.max_optimize           = max_optimize      # Maximum Iteration for Parameter Optimization
        self.amplitudes             = amplitudes        # Amplitudes of Excitation Evolvers

        # Saving Values
        self.H                      = None              # 2nd Quantized Hamiltonian
        self.ansatz                 = None              # Ansatz Quantum Circuit
        self.energy                 = 0                 # Energy at Last Cycle
        self.op_pool                = None              # Excitation Operator Pool
        self.secular_ops            = None              # List of Operators for components of Secular Equation Matrices

        # Auto-Loaded Values
        self.N_orb                  = self.classic_object.N_orb                             # Number of Spacial Orbital
        self.N_alpha                = self.classic_object.N_alpha                           # Number of Alpha-Spin Electron
        self.N_beta                 = self.classic_object.N_beta                            # Number of Beta-Spin Electron
        self.spin_symm              = (spin_symm and not(self.classic_object.is_openshell)) # Excitation Spin Symmetry 
        self.Estimator_costfunction = Transpile.QiskitCostfunction.Estimator_Costfunction   # Cost Function for Estimator


    def gen_ops(self, H):
        # 1. Generate list of excitation operators that are used for subspace expansion
        self.op_pool, _ = Excitation.generates(self.N_orb, self.N_alpha, self.N_beta, self.ex_code, self.mapping, True, self.spin_symm, True) 
        self.op_pool.append(Source.identity(self.classic_object.N_orb * 2))

        # 2. Generate F and S operator pool to make generalized eigenproblem with expectation values of these.
        op_F                = []
        op_S                = []
        for Lex in range(len(self.op_pool)):
            for Rex in range(len(self.op_pool)):
                op_F.append(self.op_pool[Lex].conjugate().transpose() @ H @ self.op_pool[Rex])
                op_S.append(self.op_pool[Lex].conjugate().transpose()   @   self.op_pool[Rex])
        F_Re, F_Im          = Source.Pauli_Decomposes(op_F)
        S_Re, S_Im          = Source.Pauli_Decomposes(op_S)
        self.secular_ops    = F_Re + F_Im + S_Re + S_Im


    def gen_ansatz(self, HF_initial=True):
        #  [Input] HF_initial : Whether the state has HF initial configuration
        # [Output]     ansatz : Reference Quantum circuit
        ansatz          = Initial.HF(self.N_orb, self.N_alpha, self.N_beta, self.mapping) if HF_initial else QuantumCircuit(2 * self.N_orb)
        return ansatz
    

    def build(self):
        # Build quantum circuit, excitation operators, and Hamiltonian for calculation
        # H = self.H
        # |psi_0> = self.ansatz
        self.H          = Hamilonian.generate(self.classic_object, self.mapping, self.cd_acc)
        self.ansatz     = self.gen_ansatz()
        self.gen_ops(self.H)
        print(f"QSE Build Done")


    def run(self, Estimator, ex_code='sd', transpiler=None):
        # [Input]  Estimator : Qiskit estimator object
        # [Input]    ex_code : Excitation codes (sdtqph for 1 to 6)
        # [Input] transpiler : Function for ansatz or operators operated between creation of the ansatz and operation. e.g.) passmanager, transpile
        # [Input]  parameter : If you need parameter, use it
        # 1. Initialize with generating operators
        time_start                                  = time.time()
        secular_ops_filtered, secular_obs_filtered  = Source.zero_operator_filter(self.secular_ops, 2 * self.classic_object.N_orb)
        time_initialize                             = time.time()
        print(  "=========================================================="
             f"\n      Excitation Code : {ex_code.upper()}"
             f"\n        Spin Symmetry : {self.spin_symm}"
             f"\n           # Operator : {len(self.secular_ops)}"
             f"\n # Measuring Operator : {len(secular_ops_filtered)}"
             f"\n  Initialization Time : {time_initialize - time_start:.3f}s"
              "\n==========================================================")
        
        # 2. Measure Operators
        if transpiler != None:
            self.mf_qc.ansatz, secular_ops_filtered     = transpiler(self.classic_object.ansatz, secular_ops_filtered)
        eval_secular_ops                                = self.Estimator_costfunction(self.amplitudes, self.ansatz, secular_ops_filtered, Estimator, 0)
        secular_obs_filtered[secular_obs_filtered > 0]  = eval_secular_ops

        time_eval = time.time()
        print(f"QSE Quantum Computing Evaluation Done | Computation Time: {time_eval - time_initialize:.3f}s")

        # 3. Reshape Evaluated Values into Matrices
        N_space                 = int((len(secular_obs_filtered) // 4) ** 0.5) 
        eval_chunk              = np.split(secular_obs_filtered, 4)
        F                       = (eval_chunk[0].reshape(N_space, N_space) + 1.0j * eval_chunk[1].reshape(N_space, N_space))
        S                       = (eval_chunk[2].reshape(N_space, N_space) + 1.0j * eval_chunk[3].reshape(N_space, N_space))

        # 4. Solve Eigenproblem
        e, XY                   = scipy.linalg.eigh(F, S)
        e_print                 = np.sort(e)
        self.e                  = e
        self.XY                 = XY

        # 5. Print Output
        time_fin                = time.time()
        output_lst              = []
        for e_ind, e_0n in enumerate(e_print):
            output_lst.append(f"E({e_ind+1}) : {e_0n.real:.4f} (+{(e_0n-e_print[e_ind-1]).real:.4f})") if e_ind != 0 else \
            output_lst.append(f"E({e_ind+1}) : {e_0n.real:.4f}")
        print(f"QSE Finished Successfully ({time_fin-time_eval:.3f}s)\n[Unit: Hartree]\n" + '\n'.join(output_lst))
        return e, XY
    

    def gen_state_transfer_op(self, excitation_level=1, e=[], XY=[], op_pool=[]):
        #  [Input]    excitation_level : Order of excitation from ground state
        #  [Input]                   e : Excited energy vector obtained from QSE
        #  [Input]                  XY : Excited operator coefficient tensor obtained from QSE
        #  [Input]             op_pool : Operator pool used in QSE formalism
        # [Output] Excitation_operator : Excitation Operator
        # Restore excitation operator from solved QSE results
        # Initialize
        if len(e) == 0:
            e = self.e
        if len(XY) == 0:
            XY = self.XY
        if len(op_pool) == 0:
            op_pool = self.op_pool

        # If excitation level is out of range
        if excitation_level < 0 or excitation_level > len(e) + 1:
            print(f"Electron Excitation Operator : 0 --(0 Hartree)--> 0")
            return Source.identity(op_pool[0].num_qubits)

        # Sort e and XY
        excitation_indices          = np.argsort(np.real(e))
        excitation_energy           = np.real(e)[excitation_indices]
        excitation_operator_coeff   = XY[:, excitation_indices]

        # Get Excitation Energy and Excitaiton Operator Coefficient Vector
        target_excitation_E                 = excitation_energy[excitation_level]
        target_excitation_operator_coeff    = excitation_operator_coeff[:, excitation_level]

        # Generate Excitation Operator
        Excitation_operator         = 0 * Source.identity(op_pool[0].num_qubits)
        for operator_ind, operator in enumerate(op_pool):
            Excitation_operator    += target_excitation_operator_coeff[operator_ind] * operator

        # Output
        print(f"Electron Excitation Operator : 0 --({target_excitation_E:.4f} Hartree)--> {excitation_level}")
        return SparsePauliOp.simplify(Excitation_operator)
    

    def gen_filtered_state_transfer_op(self, excitation_level=1, e=[], XY=[], op_pool=[], Normalize=False, Estimator=None, state_transfer_op=None):
        #  [Input]    excitation_level : Order of excitation from ground state
        #  [Input]                   e : Excited energy vector obtained from QSE
        #  [Input]                  XY : Excited operator coefficient tensor obtained from QSE
        #  [Input]             op_pool : Operator pool used in QSE formalism
        #  [Input]           Normalize : Whether do operator normalization or not
        #  [Input]           Estimator : Qiskit estimator object
        # [Output] Excitation_operator : Filtered Excitation Operator + Normalized if wanted
        # Restore excitation operator from solved QSE results
        # Initialize
        if len(e) == 0:
            e = self.e
        if len(XY) == 0:
            XY = self.XY
        if len(op_pool) == 0:
            op_pool = self.op_pool

        # If excitation level is out of range
        if excitation_level < 1 or excitation_level > len(e) + 1:
            return Source.identity(op_pool[0].num_qubits)
        
        # Get original QSE excitation operator
        QSE_excitation_operator    = self.gen_state_transfer_op(excitation_level=excitation_level, e=e, XY=XY, op_pool=op_pool) if state_transfer_op is None else state_transfer_op
        Filtered_Excitation_Op      = SparsePauliOp.simplify(QSE_excitation_operator \
                                                             - QSE_excitation_operator.conjugate().transpose())
        
        # Do normalization or not
        if Normalize:
            Normalizer                  = SparsePauliOp.simplify(Filtered_Excitation_Op.conjugate().transpose() @ Filtered_Excitation_Op)
            Op_Re, Op_Im                = Source.Pauli_Decompose(Normalizer)
            Filtered_Ops, eval_Norm     = Source.zero_operator_filter([Op_Re, Op_Im], 2 * self.classic_object.N_orb)
            eval_Norm[eval_Norm > 0]    = self.Estimator_costfunction(self.amplitudes, self.ansatz, Filtered_Ops, Estimator)
            Norm_coeff                  = (eval_Norm[0] + 1.0j * eval_Norm[1]).real
            Normalized_Excitation_Op    = Filtered_Excitation_Op / (Norm_coeff ** 0.5)
            return Normalized_Excitation_Op
        else:
            return Filtered_Excitation_Op