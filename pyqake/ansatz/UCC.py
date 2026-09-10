from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import ParameterVector, QuantumCircuit
from pyqake.operator.observable import RDM, Hamilonian, Spin_Square
from pyqake.operator.gate import Excitation
from pyqake.operator import Source
from pyqake.ansatz import Initial
from pyqake.lib import Transpile
from scipy.optimize import minimize
import numpy as np
import time
import json


mapper_code = {'jordan_wigner': 'Jordan-Wigner',
               'parity'       : 'Parity',
               'bravyi_kitaev': 'Bravyi-Kitaev'}


class UCC:
    def __init__(self        , classic_object         ,
                 ex_code='sd', mapping='jordan_wigner', 
                 cd_acc=1e-6 , max_optimize=200000    , amplitudes=[], spin_symm=True):
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
        self.op_ind                 = None              # Excitation Operator Indices
        self.op_pool                = None              # Excitation Operator Pool

        # Auto-Loaded Values
        self.N_amplitudes           = 0                                                     # Number of Amplitudes
        self.N_orb                  = self.classic_object.N_orb                             # Number of Spacial Orbital
        self.N_alpha                = self.classic_object.N_alpha                           # Number of Alpha-Spin Electron
        self.N_beta                 = self.classic_object.N_beta                            # Number of Beta-Spin Electron
        self.spin_symm              = (spin_symm and not(self.classic_object.is_openshell)) # Excitation Spin Symmetry 
        self.Estimator_costfunction = Transpile.QiskitCostfunction.Estimator_Costfunction   # Cost Function for Estimator


    def gen_ops(self):
        # Generate list of excitation operators that makes CC excitation operator
        # U = exp(-T(t))
        # T = {single_excitation} + {double excitation} + ...
        #   = sum_i(theta_i * creation_ik * annihilation_il) + sum_j(theta_j * creation_jk * creation_jl * annihilation_jm * annihilation_jn) + ...
        self.op_pool, self.op_ind   = Excitation.generates(self.N_orb, self.N_alpha, self.N_beta, self.ex_code, self.mapping, True, self.spin_symm, True) 
        self.N_amplitudes           = len(self.op_pool) # Number of amplitudes
    

    def gen_ansatz(self, op_pool, HF_initial=True):
        #  [Input]    op_pool : List of operators to make into exponential excitation operator
        #  [Input] HF_initial : Whether the state has HF initial configuration
        # [Output]     ansatz : Quantum circuit with UCC ansatz made with op_pool
        # Generate ansatz operator U with given excitation operators
        # Why multiply 1.0i? : as Qiskit provides exp(-iAt), by multiplying imaginary, can change into proper form
        # U = exp(-T(t))
        # T = sum_i(theta_i * creation_ik * annihilation_il) 
        ansatz          = Initial.HF(self.N_orb, self.N_alpha, self.N_beta, self.mapping) if HF_initial else QuantumCircuit(2 * self.N_orb)
        cc_amps         = ParameterVector("CC Amp.", len(op_pool))
        for ind, op in enumerate(op_pool):
            op.coeffs  *= 1.0j
            op_evolver  = PauliEvolutionGate(op, time=cc_amps[ind])
            ansatz.append(op_evolver, range(2 * self.N_orb))
        return ansatz
    

    def build(self):
        # Build quantum circuit, excitation operators, and Hamiltonian for calculation
        # E = <psi|H|psi> = <psi_0|U_dag H U|psi_0>
        # H = self.H
        # U|psi_0> = self.ansatz
        self.H          = Hamilonian.generate(self.classic_object, self.mapping, self.cd_acc)
        self.gen_ops()
        self.ansatz     = self.gen_ansatz(self.op_pool)
        print(f"UCC Build Done")


    def run(self, Estimator, optimize_algorithm='COBYLA', transpiler=None):
        # [Input]          Estimator : Qiskit estimator object
        # [Input] optimize_algorithm : Optimize algorithm for scipy minimizer
        # [Input]         transpiler : Function for ansatz or operators operated between creation of the ansatz and operation. e.g.) passmanager, transpile
        # Do ansatz parameter optimization based on energy minimization
        # argmin_t(E(t)) = argmin_t(<psi(t)|H|psi(t)>)
        time_start = time.time()
        print( "============================< UCC >============================\n"
              f"        Mapping Algorithm : {mapper_code[self.mapping]}\n"
              f"          Excitation Code : {self.ex_code.upper()}\n"
              f"            C.D. Accuracy : {self.cd_acc:.2e}\n"
               " ---------------------------------------------------------------\n"
              f"     Number of Amplitudes : {self.N_amplitudes}\n"
              f"       Initial Conditions : {'True' if len(self.amplitudes) == self.N_amplitudes else 'False'}\n"
              f"       Optimize Algorithm : {optimize_algorithm}\n"
              f" Max Optimizing Iteration : {self.max_optimize}\n"
               "===============================================================\n")
        
        # 1st: Initialize excitation amplitudes if there's no initial input
        # As initial state has no excitation, initial excitation amplitudes are 0
        cc_amps = np.zeros(self.N_amplitudes)
        if len(self.amplitudes) != 0:
            if len(self.amplitudes) == self.N_amplitudes:
                cc_amps         = self.amplitudes
            else:
                print("Initial Amplitude Application Failed. Reset Amplitudes")

        # 2nd: For some cases, you need to do some python operation to Hamiltonian and quantum circuit such as passmanager
        if transpiler != None:
            self.ansatz, self.H = transpiler(self.ansatz, self.H)

        # 3rd: Minimize E by changing excitation amplitudes
        res             = minimize(self.Estimator_costfunction, 
                                   cc_amps, 
                                   args     = (self.ansatz, self.H, Estimator, 0), 
                                   method   = optimize_algorithm, 
                                   options  = {'maxiter': self.max_optimize})
        self.energy     = getattr(res,'fun')
        self.amplitudes = getattr(res,'x').tolist()

        time_end = time.time()
        print(f"=====================< UCC Done ({time_end - time_start:.3f}s) >=====================\n"
              f"   Convergence : {getattr(res,'success')}\n"
              f"        Energy : {self.energy} Hartree\n")
        return self.energy
    

    def energy_tot(self, Estimator, transpiler=None):
        # [Input]  Estimator : Qiskit estimator object
        # [Input] transpiler : Function for ansatz or operators operated between creation of the ansatz and operation. e.g.) passmanager, transpile
        # Do energy calculation with operators and ansatz in this object 
        # You can change any operators or excitation amplitudes freely
        if transpiler != None:
            self.ansatz, self.H = transpiler(self.ansatz, self.H)

        # Check whether initial amplitude is empty
        if len(self.amplitudes) == 0:
            self.amplitudes = np.zeros(self.N_amplitudes)

        energy = self.Estimator_costfunction(self.amplitudes, self.ansatz, self.H, Estimator, 0)
        return energy


    def probability_distribution(self, Sampler, Shot=100000, dir=None, transpiler=None):
        # Evaluate probability distribution of electron excitation
        # It will give dictionary with {str(configuration_binary):probability}
        if transpiler != None:
            self.ansatz, self.H = transpiler(self.ansatz, self.H)
        probability_dists = Transpile.QiskitCostfunction.Sampler_Costfunction(self.amplitudes, self.ansatz, Sampler, Shot)

        states = {}
        for key in probability_dists:
            bin_key         = str(bin(int(key))[2:].zfill(self.ansatz.num_qubits))
            states[bin_key] = probability_dists[key]

        if dir != None:
            with open(f"{dir}.json", 'w') as f:
                json.dump(states, f, indent=4)
            print('Save Done')
        return probability_dists
    

    def make_rdm(self, Estimator, Degree=1, Spin_trace=True, Spin_simplify=True, transpiler=None):
        # [Input]     Estimator : Qiskit estimator object
        # [Input]        Degree : {Degree}-Body RDM will be created
        # [Input]    Spin_trace : Whether spin component of RDM is traced or not
        # [Input] Spin_simplify : Whether wants simplified not-spin-traced RDM or all possible alpha-beta permutations (Option for not Spin traced)
        #                         e.g.) for 1-RDM, (aa, ab, bb) if spin-simplified else (aa, ab, ba ,bb)
        # [Input]    transpiler : Function for ansatz or operators operated between creation of the ansatz and operation. e.g.) passmanager, transpile
        # 1st: Generate Operators
        RDM_op_Re, RDM_op_Im, RDM_binom = RDM.generate(self.N_orb, self.mapping, Degree)
        RDM_operators                   = RDM_op_Re + RDM_op_Im
        RDM_Op_filterd, RDM_components  = Source.zero_operator_filter(RDM_operators, 2 * self.N_orb)

        # 2nd: For some cases, you need to do some python operation to Hamiltonian and quantum circuit such as passmanager
        if transpiler != None:
            self.ansatz, RDM_Op_filterd = transpiler(self.ansatz, RDM_Op_filterd)

        # 3rd: Evaluate Operators
        eval_RDM_components                 = self.Estimator_costfunction(self.amplitudes, self.ansatz, RDM_Op_filterd, Estimator, 0)

        # 4th Reshape Components into RDM shape
        RDM_components[RDM_components > 0]  = eval_RDM_components
        RDM_shape                           = [self.N_orb] * (2 * Degree)
        RDM_not_traced                      = np.zeros([sum(RDM_binom)] + RDM_shape, dtype=complex)
        RDM_components_chunk                = np.split(RDM_components, 2 * sum(RDM_binom)) # 2 for Re. and Im.
        for beta_case in range(sum(RDM_binom)):
            RDM_not_traced[beta_case]   = RDM_components_chunk[beta_case].reshape(RDM_shape) +\
                                          1.0j * RDM_components_chunk[beta_case + sum(RDM_binom)].reshape(RDM_shape)
        if Spin_trace:
            RDM_traced  = np.zeros(RDM_shape, dtype=complex)
            for beta_case in range(sum(RDM_binom)):
                RDM_traced += RDM_not_traced[beta_case]
            return RDM_traced
        else:
            if Spin_simplify:
                RDM_spin_simplified = np.zeros([Degree + 1] + RDM_shape, dtype=complex)
                for spin_case in range(Degree + 1):
                    RDM_spin_simplified[spin_case]  = RDM_not_traced[sum(RDM_binom[:spin_case])]
                return RDM_spin_simplified
            else:
                return RDM_not_traced
    

    def spin_square(self, Estimator, transpiler=None):
        #  [Input]  Estimator : Qiskit estimator object
        #  [Input] transpiler : Function for ansatz or operators operated between creation of the ansatz and operation. e.g.) passmanager, transpile
        # [Output]    Eval_SS : <S^2> = <0.5 * (S+S- + S-S+) + Sz^2>
        # [Output]  Eval_SpSm : S+S-
        # [Output]  Eval_SmSp : S-S+
        # [Output]  Eval_SzSz : Sz^2
        # 1. Generate Operators
        SpSm, SmSp, SzSz, S0    = Spin_Square.generate(self.classic_object, self.mapping)

        # 2. Evaluate Components (This evaluates not accurate spin operators. Needs little post-work.)
        if transpiler != None:
            self.ansatz, [SpSm, SmSp, SzSz] = transpiler(self.ansatz, [SpSm, SmSp, SzSz])

        SS_Re, SS_Im            = Source.Pauli_Decomposes([SpSm, SmSp, SzSz])
        Filt_SS, eval_ss        = Source.zero_operator_filter(SS_Re + SS_Im, self.N_orb * 2)
        eval_ss[eval_ss > 0]    = self.Estimator_costfunction(self.amplitudes, self.ansatz, Filt_SS, Estimator, 0)

        # 3. Calculate <S^2>
        [Eval_SpSm, Eval_SmSp, Eval_SzSz]   = eval_ss[:3] + 1.0j * eval_ss[3:]
        Eval_SS                             = 0.5 * (Eval_SpSm + Eval_SmSp) + 0.25 * (Eval_SzSz) + S0
        return Eval_SS, 0.5 * Eval_SpSm, 0.5 * Eval_SmSp, 0.25 * Eval_SzSz, S0