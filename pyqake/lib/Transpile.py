from pyscf import ao2mo
import numpy as np


class QiskitCostfunction:
    def Estimator_Costfunction(Parameter=[], Ansatz=None, H=None, Estimator=None, Result_type=0):
        #  [Input]   Parameter : No paramter set or Set of parameters (for single ansatz) or List of sets of parameters (for various ansatze)
        #  [Input]      Ansatz : Quantum circuit
        #  [Input]           H : Single operator or List of operators
        #  [Input] Result_type : 0: Env(=Expectation Values) only | 1: Stds only | 2: Both Env and Stds
        # [Output]        Envs : Expectation Value(s)
        # [Output]        Stds : Standard Deviation(s)
        # Cost Function for Estimator
        # Can be modified suitable to each circumstance
        # PUBS are calculated based on qiskit PUB rule
        PUBresult   = None
        try:
            PUBresult = Estimator.run([(Ansatz, H)]).result() if len(Parameter) == 0 else Estimator.run([(Ansatz, H, Parameter)]).result()
        except ValueError:
            PUBresult = Estimator.run([(Ansatz, H)]).result()
        Envs        = [PUBresult[i].data.evs for i in range(len(PUBresult))] if len(PUBresult) > 1 else PUBresult[0].data.evs
        Stds        = [PUBresult[i].data.stds for i in range(len(PUBresult))] if len(PUBresult) > 1 else PUBresult[0].data.stds

        if Result_type == 0:
            return Envs
        elif Result_type == 1:
            return Stds
        else:
            return Envs, Stds
        
    def Sampler_Costfunction(Parameter=[], Ansatz=None, Sampler=None, Shot=100000):
        # [Input]   Parameter : No paramter set or Set of parameters (for single ansatz) or List of sets of parameters (for various ansatze)
        # [Input]      Ansatz : Quantum circuit
        # [Input]     Sampler : Qiskit Sampler1
        # [Input]        Shot : Number of shots to sample
        # Cost Function for Sampler
        # Can be modified suitable to each circumstance
        # PUBS are calculated based on qiskit PUB rule
        PUBresult   = None
        try:
            PUBresult   = Sampler.run(Ansatz, shots=Shot).result() if len(Parameter) == 0 else Sampler.run(Ansatz, Parameter, shots=Shot).result()
        except ValueError:
            PUBresult   = Sampler.run(Ansatz, shots=Shot).result()
        Quasi       = PUBresult.quasi_dists[0]
        return Quasi


class PySCFTranspiler:
    def __init__(self, mf, active_space=None, active_e=None):
        # [Input] mf : Chkfile or scf.HF or dft.KS object
        # For mf, code automatically check whether it is chkfile or not

        # Basic Load (Several things + Ingredients of Hamiltonian)
        self.mf             = mf
        self.spin           = mf.mol.spin # 2S
        self.N_orb          = mf.mol.nao
        self.N_alpha        = mf.mol.nelec[0]
        self.N_beta         = mf.mol.nelec[1]
        self.N_atom         = mf.mol.natm
        self.e0_core        = mf.mol.energy_nuc()
        self.ao_ovlp        = mf.mol.intor('int1e_ovlp')
        self.ao_h1e         = mf.mol.intor('int1e_kin') + mf.mol.intor('int1e_nuc')
        self.active_space   = active_space
        self.active_e       = active_e

        self.active         = False
        if active_space is not None \
        and active_e is not None\
        and active_e[0] <= self.N_alpha \
        and active_e[1] <= self.N_beta:
            self.active = True

        try:
            self.ao_mo_coeff    = mf.mo_coeff
            self.ao_mo_occ      = mf.mo_occ
            self.is_openshell   = len(self.ao_mo_coeff.shape) - 2
        except AttributeError:
            self.ao_mo_coeff    = mf['mo_coeff']
            self.ao_mo_occ      = mf['mo_occ']
            self.is_openshell   = len(self.ao_mo_coeff.shape) - 2

        if not(self.is_openshell) \
        and self.active \
        and active_e[0] != active_e[1]:
            self.is_openshell = True
        
        if self.is_openshell:
            mo_coeff    = np.hstack((self.ao_mo_coeff[0], self.ao_mo_coeff[1]))
            mo_eri      = ao2mo.kernel(mf.mol, mo_coeff)
            self.mo_h1e = np.einsum('xia,xjb,ij->xab', self.ao_mo_coeff, self.ao_mo_coeff, self.ao_h1e)
            self.mo_h2e = ao2mo.restore(1, mo_eri, 2 * self.N_orb)
        else:
            mo_eri      = ao2mo.kernel(mf.mol, self.ao_mo_coeff)
            self.mo_h1e = np.einsum('ia,jb,ij->ab', self.ao_mo_coeff, self.ao_mo_coeff, self.ao_h1e)
            self.mo_h2e = ao2mo.restore(1, mo_eri, self.N_orb)

        if self.active:
            core_orb_alpha      = np.array(range(self.N_alpha - active_e[0]), dtype=int)
            core_orb_beta       = np.array(range(self.N_beta  - active_e[1]), dtype=int)
            active_orb_alpha    = np.array(range(self.N_alpha - active_e[0], self.N_alpha - active_e[0] + active_space), dtype=int)
            active_orb_beta     = np.array(range(self.N_beta  - active_e[1], self.N_beta  - active_e[1] + active_space), dtype=int)
            self.core_orb       = np.vstack([core_orb_alpha, core_orb_beta])
            self.active_orb     = np.vstack([active_orb_alpha, active_orb_beta])
            
            e2, e1_e2, e0_e2    = self.apply_active_space(self.mo_h2e, self.N_orb, self.core_orb, self.active_orb)
            e1, e0_e1           = self.apply_active_space(self.mo_h1e, self.N_orb, self.core_orb, self.active_orb)

            self.N_orb          = self.active_space
            self.N_alpha        = self.active_e[0]
            self.N_beta         = self.active_e[1]
            self.mo_h2e         = e2
            self.mo_h1e         = e1 + e1_e2
            self.e0_core        = self.e0_core + e0_e1 + e0_e2            

        print("=====================< PySCF Transpiler >=====================\n"
              f"      Multiplicity : {self.spin + 1}\n"
              f" # Spatial Orbital : {self.N_orb} {'(Active Space Applied)' if self.active else ''}\n"
              f"       # Electrons : Alpha({self.N_alpha}) | Beta({self.N_beta})\n"
              f"            # Atom : {self.N_atom}\n"
              f"            Core E : {self.e0_core:.4f} Hartree")
        print("="*62)
    
    def apply_active_space(self, A, N_orb, core_orb, active_orb):
        N_body  = int(len(A.shape) // 2) 
        if N_body == 1:
            if self.is_openshell:
                e0_a    = np.einsum('ii', A[0][np.ix_(core_orb[0], core_orb[0])])
                e0_b    = np.einsum('ii', A[1][np.ix_(core_orb[1], core_orb[1])])
                e1_a    = A[0][np.ix_(active_orb[0], active_orb[0])]
                e1_b    = A[1][np.ix_(active_orb[1], active_orb[1])]
                return np.array([e1_a, e1_b]), e0_a + e0_b
            else:
                e0  = 2 * np.einsum('ii', A[np.ix_(core_orb[1], core_orb[1])])
                e1  = A[np.ix_(active_orb[0], active_orb[0])]
                return e1, e0
            
        elif N_body == 2:
            if self.is_openshell:
                total_core_orb      = np.hstack([core_orb[0]  , core_orb[1] + N_orb])
                total_active_orb    = np.hstack([active_orb[0], active_orb[1] + N_orb])
                e0      = 0.5 * np.einsum('iijj', A[np.ix_(total_core_orb, total_core_orb, total_core_orb, total_core_orb)]) \
                        - 0.5 * np.einsum('ijji', A[np.ix_(core_orb[0], core_orb[0], core_orb[0], core_orb[0])]) \
                        - 0.5 * np.einsum('ijji', A[np.ix_(core_orb[1] + N_orb, core_orb[1] + N_orb, core_orb[1] + N_orb, core_orb[1] + N_orb)])
                e1_J    = np.einsum('klii->kl', A[np.ix_(total_active_orb, total_active_orb, total_core_orb, total_core_orb)])
                e1_K_a  = np.einsum('kiil->kl', A[np.ix_(active_orb[0], core_orb[0], core_orb[0], active_orb[0])])
                e1_K_b  = np.einsum('kiil->kl', A[np.ix_(active_orb[1] + N_orb, core_orb[1] + N_orb, core_orb[1] + N_orb, active_orb[1] + N_orb)])
                e1      = np.array([e1_J[:len(active_orb[0]), :len(active_orb[0])] - e1_K_a, 
                                    e1_J[len(active_orb[0]):, len(active_orb[0]):] - e1_K_b])
                e2      = A[np.ix_(total_active_orb, total_active_orb, total_active_orb, total_active_orb)]
                return e2, e1, e0
            else:
                e0      = 2 * np.einsum('iijj', A[np.ix_(core_orb[0], core_orb[0], core_orb[0], core_orb[0])]) \
                        - np.einsum('ijji', A[np.ix_(core_orb[0], core_orb[0], core_orb[0], core_orb[0])])
                e1      = 2 * np.einsum('klii->kl', A[np.ix_(active_orb[0], active_orb[0], core_orb[0], core_orb[0])]) \
                        - np.einsum('kiil->kl', A[np.ix_(active_orb[0], core_orb[0], core_orb[0], active_orb[0])])
                e2      = A[np.ix_(active_orb[0], active_orb[0], active_orb[0], active_orb[0])]
                return e2, e1, e0

    
    def call(self, call_id):
        # [Input] call_id : ID wants to call
        # This function exists to deal with various (but limited) code
        # And not to save many things...
        if call_id == '1e_r': #<i|r|j>
            return self.mf.mol.intor('int1e_r_cart')
        
        elif call_id == 'atom_coords':
            return self.mf.mol.atom_coords()
        
        elif call_id == 'atom_symbol':
            return self.mf.mol.atom_symbol