from pyscf import scf, mcscf, fci
from pyscf.fci import cistring
from qiskit.quantum_info import Operator, Statevector, state_fidelity
from pyqake.lib.Transpile import PySCFTranspiler
from pyqake.ansatz.UCC import UCC
from pyqake.ansatz import qEOM
import numpy as np
import sys

###=============================================<<< Setup

# Ovl(i, j) : |<CASCI_i | X_j>|, state-wise overlap between an exact CASCI (FCI)
#             eigenstate i and a quantum-computed state j
#
# CASCI  : PySCF FCI eigenstates within the active space (exact reference)
#  qEOM  : |n> ~ O_n|ref>,           O_n  = qEOM.gen_state_transfer_op(n)
# AqEOM  : |n> ~ (O_n - O_n^+)|ref>, the Hermitian-filtered variant used throughout
#          AqEOM/Code/Running/qEOM/*.py (there called fqEOM/AqEOM) -- removes the
#          part of O_n that isn't a genuine excitation/de-excitation operator
#   ref  : UCCSD ground state (reloaded from saved amplitudes, not re-optimized)

# Basis match: PyQAKE's JW convention (pyqake/ansatz/Initial.py) blocks qubits as
# [0..N_orb) = alpha spin-orbitals, [N_orb..2N_orb) = beta spin-orbitals, occupation
# bit -> X gate directly. PySCF's FCI civector is indexed by (alpha_string,
# beta_string) bitmasks over the same active orbitals (pyscf.fci.cistring) -- direct
# index map, no basis rotation needed:
#   qubit_index = alpha_bitmask | (beta_bitmask << N_orb)

# !!! Use fci.direct_spin1.FCI, NOT direct_spin0 !!!
# direct_spin0 only diagonalizes the alpha<->beta-symmetric (singlet) subspace and
# silently drops open-shell (S>0) roots even when nroots is set high enough to
# request them.

# !!! Exactly-degenerate CASCI states show low *diagonal* fidelity even when
# everything is correct !!! LiH's Li-2p-derived Pi-symmetry states come in exactly
# degenerate pairs -- PySCF and qEOM independently pick arbitrary bases within a
# degenerate subspace, so check sum_i Ovl(i,j)^2 over the degenerate block instead
# of trusting a single diagonal entry.

# !!! Please handle active space manually !!!

dist            = np.round(float(sys.argv[1]), 1)  # Bond distance
master_dir      = '***'
system          = 'LiH'
active_space    = 5
active_e        = (1, 1)  # Li 1s core (2 electrons) frozen

N_states        = 7  # full active space is 25-dim (5x5 determinants); catches the
                      # ground state + the first two exactly-degenerate Pi pairs

###=============================================<<< Helper Functions

def casci_to_statevector(ci_vec, ncas, nelecas):
    #  [Input] ci_vec : PySCF FCI coefficient array, shape (n_alpha_str, n_beta_str)
    #  [Input]   ncas : Number of active spatial orbitals
    #  [Input] nelecas : (N_alpha, N_beta) active electrons
    # [Output]      sv : length-2^(2*ncas) qubit statevector, PyQAKE JW convention
    na, nb  = nelecas
    sv      = np.zeros(2 ** (2 * ncas), dtype=complex)
    for ia in range(ci_vec.shape[0]):
        alpha_str = cistring.addr2str(ncas, na, ia)
        for ib in range(ci_vec.shape[1]):
            c = ci_vec[ia, ib]
            if c != 0:
                beta_str = cistring.addr2str(ncas, nb, ib)
                sv[alpha_str | (beta_str << ncas)] = c
    return sv


def build_state(Op, ref_sv):
    #  [Input]    Op : SparsePauliOp acting on |ref> (not necessarily unitary/normalized)
    #  [Input] ref_sv : Statevector of the reference state
    # [Output]  (Statevector |Op*ref>/||.|| , norm before normalizing)
    sv   = Operator(Op).data @ ref_sv.data
    norm = np.linalg.norm(sv)
    return Statevector(sv / norm), norm


def overlap_matrix(casci_svs, other_svs):
    # [Output] Ovl[i, j] = |<CASCI_i | other_j>| (magnitude only -- each solver's
    #                                             overall phase is arbitrary)
    Ovl = np.zeros((len(casci_svs), len(other_svs)))
    for i, sv_c in enumerate(casci_svs):
        for j, sv_o in enumerate(other_svs):
            Ovl[i, j] = abs(np.vdot(sv_c.data, sv_o.data))
    return Ovl


def print_overlap_matrix(Ovl, label):
    N_c, N_o = Ovl.shape
    header   = "        " + "".join(f"{label}_{j:<{9-len(label)}}" for j in range(N_o))
    print(header)
    for i in range(N_c):
        row = "".join(f"{Ovl[i, j]:<11.4f}" for j in range(N_o))
        print(f"CASCI_{i}  {row}")

###=============================================<<< Calculation

# 1. HF (reloaded from chkfile; this script never regenerates HF/UCC/qEOM data)
mol, mf_chk = scf.chkfile.load_scf(f"{master_dir}/Data/{system}/Chkfile/scf_{dist}")
mf_hf       = scf.RHF(mol)
mf_hf.__dict__.update(mf_chk)
Transpiled  = PySCFTranspiler(mf_hf, active_space, active_e)  # !!! Active Space Here !!!
print("HF Reload Done")

# 2. UCCSD reference state |ref>
ucc_amp      = np.load(f"{master_dir}/Data/{system}/Res/UCC_{dist}.npy")
mf_qc        = UCC(classic_object=Transpiled, spin_symm=True, amplitudes=ucc_amp)
mf_qc.build()
ref_sv       = Statevector(mf_qc.ansatz.assign_parameters(mf_qc.amplitudes))
mf_qc.energy = ref_sv.expectation_value(Operator(mf_qc.H)).real  # not set by build(), only by run()
print("UCCSD Reload Done")

# 3. qEOM secular-equation solution
mf_qEOM     = qEOM.qEOM(mf_qc)
mf_qEOM.e   = np.load(f"{master_dir}/Data/{system}/qEOM/E_{dist}.npy")
mf_qEOM.XY  = np.load(f"{master_dir}/Data/{system}/qEOM/XY_{dist}.npy")
mf_qEOM._gen_ops()
N_states    = min(N_states, int(mf_qEOM.e.shape[0] / 2) + 1)  # +1 : ground state
print("qEOM Reload Done")

# 4. CASCI reference (same active space as the qEOM run)
mc                  = mcscf.CASCI(mf_hf, active_space, active_e)
mc.fcisolver        = fci.direct_spin0.FCI(mol)
mc.fcisolver.nroots = 25
mc.kernel(mf_hf.mo_coeff)
N_states            = min(N_states, len(mc.ci))
casci_svs           = [Statevector(casci_to_statevector(mc.ci[i], mc.ncas, mc.nelecas)) for i in range(N_states)]
print("CASCI Done")

# 5. Build qEOM and AqEOM state families, state by state
qeom_svs, AqEOM_svs, qeom_E = [], [], []
for n in range(N_states):
    O_n       = mf_qEOM.gen_state_transfer_op(excitation_level=n)
    sv_n, _   = build_state(O_n, ref_sv)
    qeom_svs.append(sv_n)

    fO_n      = mf_qEOM.gen_filtered_state_transfer_op(excitation_level=n, state_transfer_op=O_n)
    sv_fn, _  = build_state(fO_n, ref_sv)
    AqEOM_svs.append(sv_fn)

    qeom_E.append(mf_qc.energy if n == 0 else np.sort(mf_qEOM.e[mf_qEOM.e > 0].real)[n - 1] + mf_qc.energy)
print("qEOM / AqEOM State Construction Done")

# 6. Diagonal (energy-matched) fidelity summary
print(f"\n{'n':>2}  {'E_CASCI (Ha)':>14}  {'E_qEOM (Ha)':>13}  {'F(qEOM)':>9}  {'F(AqEOM)':>9}")
for n in range(N_states):
    F_qEOM  = state_fidelity(qeom_svs[n],  casci_svs[n])
    F_AqEOM = state_fidelity(AqEOM_svs[n], casci_svs[n])
    print(f"{n:>2}  {mc.e_tot[n]:>14.6f}  {qeom_E[n]:>13.6f}  {F_qEOM:>9.6f}  {F_AqEOM:>9.6f}")

# 7. All-to-all overlap matrices: Ovl(i, j) = |<CASCI_i | X_j>|
Ovl_qEOM  = overlap_matrix(casci_svs, qeom_svs)
Ovl_AqEOM = overlap_matrix(casci_svs, AqEOM_svs)

print("\nOvl(CASCI, qEOM):")
print_overlap_matrix(Ovl_qEOM, "qEOM")
print("\nOvl(CASCI, AqEOM):")
print_overlap_matrix(Ovl_AqEOM, "AqEOM")

# 8. Save
np.save(f"{master_dir}/Data/{system}/CASCI_Overlap/{dist}_Ovl_qEOM" , Ovl_qEOM)
np.save(f"{master_dir}/Data/{system}/CASCI_Overlap/{dist}_Ovl_AqEOM", Ovl_AqEOM)
print("Save Done")
