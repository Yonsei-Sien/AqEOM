from qiskit.quantum_info import SparsePauliOp
from pyqake.operator.gate import Excitation, EA, IP
from pyqake.operator import Source
from pyqake.lib import Transpile
import numpy as np
import scipy
import time


property_cases  = {'ee': 'Electron Excitation',
                   'ea': 'Electron Affinity',
                   'ip': 'Ionization Potential'}


class qEOM:
    def __init__(self, mf_qc, property='ee', spin_symm=False):
        # Input Values
        self.mf_qc                  = mf_qc                                         # QC object such as UCC.UCC
        self.property               = property                                      # Property wants to see, EE | EA | IP
        self.spin_symm              = (spin_symm and \
                                       not(self.mf_qc.classic_object.is_openshell)) # Excitation Spin Symmetry 

        # Saving Values
        self.op_pool                = None      # Operators for configuration excitation
        self.secular_ops            = None      # List of Operators for components of Secular Equation Matrices
        self.evals                  = None      # Evaluated values for self.secular_ops
        self.e                      = None      # (De)Excitation energies after qEOM
        self.XY                     = None      # (De)Excitation operator coefficient tensor
        
        # Auto-Loading Values
        self.Estimator_costfunction = Transpile.QiskitCostfunction.Estimator_Costfunction   # Cost Function for Estimator


    def _gen_ops(self, ex_code='sd'):
        # [Input] ex_code : Excitation codes (sdtqph for 1 to 6)
        # Make operator pool only. Useful when only want to restore excitation operator with saved XY tensor.
        op_pool = None
        if self.property.lower() == 'ea':
            op_pool, _ = EA.generates(self.mf_qc.N_orb, 
                                      self.mf_qc.N_alpha, 
                                      self.mf_qc.N_beta, 
                                      ex_code,
                                      self.mf_qc.mapping, 
                                      True,
                                      self.spin_symm,
                                      False)        
        elif self.property.lower() == 'ip':
            op_pool, _ = IP.generates(self.mf_qc.N_orb, 
                                      self.mf_qc.N_alpha, 
                                      self.mf_qc.N_beta, 
                                      ex_code,
                                      self.mf_qc.mapping, 
                                      True,
                                      self.spin_symm,
                                      False)
        else:
            op_pool, _ = Excitation.generates(self.mf_qc.N_orb, 
                                            self.mf_qc.N_alpha, 
                                            self.mf_qc.N_beta, 
                                            ex_code,
                                            self.mf_qc.mapping, 
                                            True,
                                            self.spin_symm,
                                            False)       
        self.op_pool    = op_pool


    def gen_ops(self, ex_code='sd'):
        # [Input] ex_code : Excitation codes (sdtqph for 1 to 6)
        # 1. General Excitation Operators
        op_pool = None
        if self.property.lower() == 'ea':
            op_pool, _ = EA.generates(self.mf_qc.N_orb, 
                                      self.mf_qc.N_alpha, 
                                      self.mf_qc.N_beta, 
                                      ex_code,
                                      self.mf_qc.mapping, 
                                      True,
                                      self.spin_symm,
                                      False)  
        elif self.property.lower() == 'ip':
            op_pool, _ = IP.generates(self.mf_qc.N_orb, 
                                      self.mf_qc.N_alpha, 
                                      self.mf_qc.N_beta, 
                                      ex_code,
                                      self.mf_qc.mapping, 
                                      True,
                                      self.spin_symm,
                                      False)
        else:
            op_pool, _ = Excitation.generates(self.mf_qc.N_orb, 
                                              self.mf_qc.N_alpha, 
                                              self.mf_qc.N_beta, 
                                              ex_code,
                                              self.mf_qc.mapping, 
                                              True,
                                              self.spin_symm,
                                              False)    
        op_pool_adjoint = [op.conjugate().transpose() for op in op_pool]
        self.op_pool    = op_pool

        # 2. Operators to Evaluate for Making Matrices
        # Signs are different compares to original paper's. Please consider that.
        M   = Source.get_double_commutators(op_pool_adjoint, self.mf_qc.H, op_pool)
        Q   = Source.get_double_commutators(op_pool_adjoint, self.mf_qc.H, op_pool_adjoint)
        V   = Source.get_commutators(op_pool_adjoint, op_pool, True)
        W   = Source.get_commutators(op_pool_adjoint, op_pool_adjoint, True)

        # 3. Decompose Pauli Operators to be Hermitian (Decompose by the dimension of coefficients... whether imaginary or real...)
        M_Re, M_Im          = Source.Pauli_Decomposes(M)
        Q_Re, Q_Im          = Source.Pauli_Decomposes(Q)
        V_Re, V_Im          = Source.Pauli_Decomposes(V)
        W_Re, W_Im          = Source.Pauli_Decomposes(W)
        self.secular_ops    = M_Re + M_Im + Q_Re + Q_Im + V_Re + V_Im + W_Re + W_Im 


    def run(self, Estimator, ex_code='sd', transpiler=None):
        # [Input]  Estimator : Qiskit estimator object
        # [Input]    ex_code : Excitation codes (sdtqph for 1 to 6)
        # [Input] transpiler : Function for ansatz or operators operated between creation of the ansatz and operation. e.g.) passmanager, transpile
        # 1. Initialize with generating operators
        time_start                                  = time.time()
        self.gen_ops(ex_code)
        secular_ops_filtered, secular_obs_filtered  = Source.zero_operator_filter(self.secular_ops, 2 * self.mf_qc.N_orb)
        time_initialize                             = time.time()
        print(  "=========================================================="
             f"\n    Physical Property : {property_cases[self.property.lower()]}"
             f"\n      Excitation Code : {ex_code.upper()}"
             f"\n        Spin Symmetry : {self.spin_symm}"
             f"\n           # Operator : {len(self.secular_ops)}"
             f"\n # Measuring Operator : {len(secular_ops_filtered)}"
             f"\n  Initialization Time : {time_initialize - time_start:.3f}s"
              "\n==========================================================")

        # 2. Measure Operators
        if transpiler != None:
            self.mf_qc.ansatz, secular_ops_filtered     = transpiler(self.mf_qc.ansatz, secular_ops_filtered)
        eval_secular_ops                                = self.Estimator_costfunction(self.mf_qc.amplitudes, self.mf_qc.ansatz, secular_ops_filtered, Estimator, 0)
        secular_obs_filtered[secular_obs_filtered > 0]  = eval_secular_ops

        time_eval = time.time()
        print(f"qEOM Quantum Computing Evaluation Done | Computation Time: {time_eval - time_initialize:.3f}s")

        # 3. Reshape Evaluated Values into Matrices
        N_space                 = int((len(secular_obs_filtered) // 8) ** 0.5) 
        eval_chunk              = np.split(secular_obs_filtered, 8)
        M                       = (eval_chunk[0].reshape(N_space, N_space) + 1.0j * eval_chunk[1].reshape(N_space, N_space))
        Q                       = (eval_chunk[2].reshape(N_space, N_space) + 1.0j * eval_chunk[3].reshape(N_space, N_space)) * (-1)
        V                       = (eval_chunk[4].reshape(N_space, N_space) + 1.0j * eval_chunk[5].reshape(N_space, N_space))
        W                       = (eval_chunk[6].reshape(N_space, N_space) + 1.0j * eval_chunk[7].reshape(N_space, N_space)) * (-1)
        self.evals              = [M, Q, V, W]

        # 4. Make Matrices into Secular Equation Form
        S_L                     = np.zeros((2 * N_space, 2 * N_space), dtype=complex)
        S_L[:N_space, :N_space] = M
        S_L[:N_space, N_space:] = Q
        S_L[N_space:, :N_space] = Q.conjugate()
        S_L[N_space:, N_space:] = M.conjugate()

        S_R                     = np.zeros((2 * N_space, 2 * N_space), dtype=complex)
        S_R[:N_space, :N_space] = V
        S_R[:N_space, N_space:] = W
        S_R[N_space:, :N_space] = W.conjugate() * (-1)
        S_R[N_space:, N_space:] = V.conjugate() * (-1)

        # 5. Solve Eigenproblem
        e, XY                   = scipy.linalg.eig(S_L, S_R)
        e_print                 = np.sort(e[e>0])
        self.e                  = e
        self.XY                 = XY

        # 6. Print Output
        time_fin                = time.time()
        output_lst              = []
        for e_ind, e_0n in enumerate(e_print):
            output_lst.append(f"E({e_ind+1}) : {e_0n.real:.4f} (+{(e_0n-e_print[e_ind-1]).real:.4f})") if e_ind != 0 else \
            output_lst.append(f"E({e_ind+1}) : {e_0n.real:.4f}")
        print(f"qEOM Finished Successfully ({time_fin-time_eval:.3f}s)\n[Unit: Hartree]\n" + '\n'.join(output_lst))
        return e, XY
    

    def gen_state_transfer_op(self, excitation_level=1, e=[], XY=[], op_pool=[]):
        #  [Input]    excitation_level : Order of excitation from ground state
        #  [Input]                   e : Excited energy vector obtained from qEOM
        #  [Input]                  XY : Excited operator coefficient tensor obtained from qEOM
        #  [Input]             op_pool : Operator pool used in qEOM formalism
        # [Output] Excitation_operator : Excitation Operator
        # Restore excitation operator from solved qEOM results
        # Initialize
        if len(e) == 0:
            e = self.e
        if len(XY) == 0:
            XY = self.XY
        if len(op_pool) == 0:
            op_pool = self.op_pool

        # If excitation level is out of range
        if excitation_level < 1 or excitation_level > len(e) + 1:
            print(f"{property_cases[self.property.lower()]} Operator : 0 --(0 Hartree)--> 0")
            return Source.identity(op_pool[0].num_qubits)

        # Sort e and XY
        _excitation_indices         = np.where(np.real(e) > 0)[0]
        _excitation_energy          = np.real(e[_excitation_indices])
        _excitation_operator_coeff  = XY[:, _excitation_indices]
        excitation_indices          = np.argsort(_excitation_energy)
        excitation_energy           = _excitation_energy[excitation_indices]
        excitation_operator_coeff   = _excitation_operator_coeff[:, excitation_indices]

        # Get Excitation Energy and Excitaiton Operator Coefficient Vector
        target_excitation_E                 = excitation_energy[excitation_level - 1]
        target_excitation_operator_coeff    = excitation_operator_coeff[:, excitation_level - 1]

        # Generate Excitation Operator
        Excitation_operator         = 0 * Source.identity(op_pool[0].num_qubits)
        for operator_ind, operator in enumerate(op_pool):
            Excitation_operator    += target_excitation_operator_coeff[operator_ind] * operator -\
                                      target_excitation_operator_coeff[operator_ind + len(op_pool)] * operator.conjugate().transpose()

        # Output
        print(f"{property_cases[self.property.lower()]} Operator : 0 --({target_excitation_E:.4f} Hartree)--> {excitation_level}")
        return SparsePauliOp.simplify(Excitation_operator)
    

    def _gen_state_transfer_op(self, excitation_level=1, e=[], XY=[], op_pool=[]):
        #  [Input]    excitation_level : Order of excitation from ground state
        #  [Input]                   e : Excited energy vector obtained from qEOM
        #  [Input]                  XY : Excited operator coefficient tensor obtained from qEOM
        #  [Input]             op_pool : Operator pool used in qEOM formalism
        # [Output] Excitation_operator : Excitation Operator
        # Restore excitation operator from solved qEOM results
        # Initialize
        if len(e) == 0:
            e = self.e
        if len(XY) == 0:
            XY = self.XY
        if len(op_pool) == 0:
            op_pool = self.op_pool

        # If excitation level is out of range
        if excitation_level == 0 or abs(excitation_level) > int(len(e) // 2):
            print(f"{property_cases[self.property.lower()]} Operator : 0 --(0 Hartree)--> 0")
            return Source.identity(op_pool[0].num_qubits)

        # Sort e and XY
        excitation_indices          = np.argsort(e.real)
        excitation_energy           = e.real[excitation_indices]
        excitation_operator_coeff   = XY[:, excitation_indices]

        # Get Excitation Energy and Excitaiton Operator Coefficient Vector
        state_level_direction               = 0 if excitation_level < 0 else -1
        target_excitation_E                 = excitation_energy[excitation_level + int(len(excitation_energy) // 2) + state_level_direction]
        target_excitation_operator_coeff    = excitation_operator_coeff[:, excitation_level + int(len(excitation_energy) // 2) + state_level_direction]

        # Generate Excitation Operator
        Excitation_operator         = 0 * Source.identity(op_pool[0].num_qubits)
        for operator_ind, operator in enumerate(op_pool):
            Excitation_operator    += target_excitation_operator_coeff[operator_ind] * operator -\
                                      target_excitation_operator_coeff[operator_ind + len(op_pool)] * operator.conjugate().transpose()

        # Output
        print(f"{property_cases[self.property.lower()]} Operator : 0 --({target_excitation_E:.4f} Hartree)--> {excitation_level}")
        return SparsePauliOp.simplify(Excitation_operator)
    

    def gen_filtered_state_transfer_op(self, excitation_level=1, e=[], XY=[], op_pool=[], Normalize=False, Estimator=None, state_transfer_op=None):
        #  [Input]    excitation_level : Order of excitation from ground state
        #  [Input]                   e : Excited energy vector obtained from qEOM
        #  [Input]                  XY : Excited operator coefficient tensor obtained from qEOM
        #  [Input]             op_pool : Operator pool used in qEOM formalism
        #  [Input]           Normalize : Whether do operator normalization or not
        #  [Input]           Estimator : Qiskit estimator object
        # [Output] Excitation_operator : Filtered Excitation Operator + Normalized if wanted
        # Restore excitation operator from solved qEOM results
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
        
        # Get original qEOM excitation operator
        qEOM_excitation_operator    = self.gen_state_transfer_op(excitation_level=excitation_level, e=e, XY=XY, op_pool=op_pool) if state_transfer_op is None else state_transfer_op
        Filtered_Excitation_Op      = SparsePauliOp.simplify(qEOM_excitation_operator \
                                                             - qEOM_excitation_operator.conjugate().transpose())
        
        # Do normalization or not
        if Normalize:
            Normalizer                  = SparsePauliOp.simplify(Filtered_Excitation_Op.conjugate().transpose() @ Filtered_Excitation_Op)
            Op_Re, Op_Im                = Source.Pauli_Decompose(Normalizer)
            Filtered_Ops, eval_Norm     = Source.zero_operator_filter([Op_Re, Op_Im], 2 * self.mf_qc.N_orb)
            eval_Norm[eval_Norm > 0]    = self.mf_qc.Estimator_costfunction(self.mf_qc.amplitudes, self.mf_qc.ansatz, Filtered_Ops, Estimator)
            Norm_coeff                  = (eval_Norm[0] + 1.0j * eval_Norm[1]).real
            Normalized_Excitation_Op    = Filtered_Excitation_Op / (Norm_coeff ** 0.5)
            return Normalized_Excitation_Op
        else:
            return Filtered_Excitation_Op