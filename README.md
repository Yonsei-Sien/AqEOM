# AqEOM — Anti-Hermitized quantum Equation-of-Motion

Code accompanying

> **Excitation-Energy Accuracy Does Not Guarantee Observable Fidelity:
> Rethinking Validation of Excited-State Quantum Algorithms**
> Gyumin Kim, Youngsam Kim, Sehun Kim, and Eunji Sim — Department of Chemistry, Yonsei University

> Code written by Gyumin Kim

The repository contains `pyqake`, a small Qiskit + PySCF toolkit for UCC / qEOM / QSE excited-state
calculations with direct observable evaluation, the production scripts that generated every raw result in
the paper (`Reproduce/Running/`), and the notebooks that turn those results into the figures and tables
(`Reproduce/Analysis/`).

| File | Content |
| --- | --- |
| `AqEOM_Main.pdf` | Main text (13 pp.) |
| `AqEOM_SI.pdf` | Supplementary Materials (40 pp., SI.1–SI.13) |

---

## What the paper does

qEOM returns accurate excitation energies, but the state-transfer operator $\hat O_n^\dagger$ rebuilt from its
eigenvectors is only constrained through commutator expectation values. When the same operator is used to
prepare states for a *direct* property evaluation, $\langle 0|\hat O_n \hat A \hat O_m^\dagger|0\rangle$, Hermitian
components such as ground-state leakage ($|0\rangle\langle 0|$) show up as large errors in dipole moments,
state overlaps and RDMs, even though the excitation energies look fine.

AqEOM applies an a posteriori Cartesian decomposition and keeps only the anti-Hermitian part:

$$
\hat U_n^\dagger \equiv \frac{\hat O_n^\dagger - \hat O_n}
{\sqrt{\langle 0|(\hat O_n - \hat O_n^\dagger)(\hat O_n^\dagger - \hat O_n)|0\rangle}}
$$

The qEOM generalized eigenvalue problem itself is unchanged, and no extra quantum measurements are needed.
The diagnostics the paper uses are:

| Quantity | Definition (main text) | Where it is computed |
| --- | --- | --- |
| Direct-operator energy $E^O_n$, $E^U_n$ | Eq. 10 | `Running/qEOM/*.py` step 5 |
| Permanent / transition dipole $\mu^O_{nm}$, $\mu^U_{nm}$ | Eqs. 11–12 | `Running/qEOM/*.py` step 6 |
| State overlap $S_{mn}$ and $\Delta S$ | Eqs. 14–16 | `Running/qEOM/*.py` step 7 (`VAC/`) |
| DC-DFT RDM diagnostic $D_\mathrm{DFA}[n]$ | Eq. 13 | `Running/DC/*.py` |
| Hermitian contamination $C^H_n$ | Eq. 17 | `Running/qEOM/*.py` step 9 |

---

## Repository layout

```
pyqake/
├── ansatz/
│   ├── Initial.py       HF / bitstring reference circuits (Jordan-Wigner, parity)
│   ├── UCC.py           UCC(SD) VQE: build, run (SciPy minimizer), RDMs, <S^2>, sampling
│   ├── qEOM.py          qEOM secular problem; state-transfer operators; AqEOM projection
│   └── QSE.py           Quantum Subspace Expansion (operator pool includes the identity)
├── operator/
│   ├── Source.py        JW/parity creation-annihilation ops, (double) commutators,
│   │                    Cartesian (Hermitian / anti-Hermitian) decomposition, zero-op filter
│   ├── gate/
│   │   └── Excitation.py    spin-adapted excitation operator pools (s, d, t, q, p, h)
│   └── observable/
│       ├── Hamilonian.py    second-quantized H (Cholesky-decomposed ERIs)  [sic: filename typo]
│       ├── Dipole.py        electronic + core dipole operators, <0|L† μ R|0> evaluation
│       ├── RDM.py           n-body RDM operator generation
│       └── Spin_Square.py   <S^2> operators
└── lib/
    ├── Transpile.py             PySCFTranspiler (MO integrals, active space) + Qiskit
    │                            Estimator/Sampler cost functions
    ├── MO2AO.py                 RDM basis transform, active→full space restoration
    └── Cholesky_Decomposition.py  ERI Cholesky decomposition

Reproduce/
├── Running/                     production calculations (write .npy files into Data/)
│   ├── qEOM/                    HF → UCCSD → qEOM → <S^2>, energies, dipoles, overlaps,
│   │   ├── qEOM.sh              RDMs, Hermitian contamination (qEOM and AqEOM)
│   │   └── H2 H2_Large H2_Huge HeH LiH LiH_Large BeH2 NH3 H2O H2O_Small HF CH4 .py
│   ├── CASCI.ipynb              CASCI reference energies, dipoles and 1-RDMs for every system
│   ├── QSE/                     QSE comparison (H2, LiH, H2O_Small)
│   ├── DC/                      DC-DFT RDM diagnostic over 10 DFAs (H2, LiH, H2O)
│   ├── CASCI_Overlap/           |<CASCI_i|qEOM_j>| and |<CASCI_i|AqEOM_j>| state-vector overlaps
│   ├── Natural_Occupancy/       natural occupations matched to CASCI natural orbitals
│   ├── Origin_Shift/            dipole origin-dependence tests (H2 @ 1.0 Å, LiH @ 2.0 Å)
│   └── Noise/                   H2 @ 0.7 Å with qiskit-aer depolarizing noise + finite shots
└── Analysis/                    figures and tables (read Data/)
    ├── Fig1.ipynb               H2 energies / dipoles: CASCI, QSE, qEOM, AqEOM; RMSE summary
    ├── Fig2.ipynb               LiH error correlations (energy error, GS leakage,
    │                            Hermitian contamination, dipole error)
    ├── Fig3.ipynb               dissociation-curve errors for HeH+, LiH, BeH2, NH3, H2O, HF
    ├── Fig4.ipynb               DC-r2SCAN RDM diagnostic and state-overlap matrix (H2)
    ├── Statistics.ipynb         Table 1: Spearman / Pearson, C^H_n vs. dipole error
    ├── SI.ipynb                 every SI figure (correlations, energies, dipoles, DC-DFT,
    │                            orthogonality, CASCI overlap, origin shift, natural
    │                            occupancy, bond-length trend subtraction / Table S17)
    └── CSV.ipynb                dumps energies and |μ_z| to Data/CSV/ (SI tables)
```

> **Naming note.** In script variables and on-disk file names, `fqEOM` ("filtered qEOM") is **AqEOM**.

### Key entry points

| Object | Purpose |
| --- | --- |
| `Transpile.PySCFTranspiler(mf, active_space, active_e)` | Turns a PySCF `scf`/`dft` object (or chkfile) into MO integrals with an optional (No, (Nα, Nβ)) active space |
| `UCC.UCC(classic_object, ...)` | `.build()`, `.run(Estimator)` → VQE ground state; also `.make_rdm()`, `.spin_square()` |
| `qEOM.qEOM(mf_qc, property='ee'\|'ea'\|'ip')` | `.run(Estimator)` → excitation energies `e` and coefficients `XY` |
| `qEOM.gen_state_transfer_op(n)` | the plain qEOM operator $\hat O_n^\dagger$ |
| `qEOM.gen_filtered_state_transfer_op(n, Normalize=True, Estimator=...)` | **the AqEOM operator** $\hat U_n^\dagger$ |
| `QSE.QSE(classic_object, ...)` | QSE reference calculation |
| `Dipole.run(mf_qc, Estimator, L, R)` | $\langle 0|\hat L^\dagger \hat\mu \hat R|0\rangle$ for permanent and transition moments |

---

## Requirements

- Python 3.10+
- [PySCF](https://pyscf.org/) — HF, DFT, CASCI reference values, integrals
- [Qiskit](https://www.ibm.com/quantum/qiskit) (v1 primitives API: `Estimator.run([(circuit, obs, params)])`)
- [qiskit-aer](https://github.com/Qiskit/qiskit-aer) — only for `Running/Noise/`
- NumPy, SciPy (COBYLA for VQE, `linear_sum_assignment` for natural-orbital matching, `scipy.stats`)
- Jupyter + Matplotlib for the notebooks
- bash (the `*.sh` launchers use associative arrays and `wait -n`, i.e. bash ≥ 4.3)

```bash
pip install pyscf qiskit qiskit-aer numpy scipy matplotlib jupyter
```

There is no packaging metadata; put the repository root on `PYTHONPATH` so that `import pyqake` resolves.

---

## Quick start

```python
from qiskit.primitives import StatevectorEstimator
from pyscf import gto, scf

from pyqake.lib import Transpile
from pyqake.ansatz import UCC, qEOM
from pyqake.operator.observable import Dipole

# 1. Classical reference
mol = gto.M(atom="H 0 0 0; H 0 0 0.7", basis="sto-3g")
mf  = scf.RHF(mol).run()
ref = Transpile.PySCFTranspiler(mf)                 # or (mf, active_space=5, active_e=(1, 1))

# 2. UCCSD ground state
estimator = StatevectorEstimator()
mf_qc = UCC.UCC(ref, spin_symm=True)
mf_qc.build()
mf_qc.run(estimator)

# 3. qEOM excitation energies
eom  = qEOM.qEOM(mf_qc, property="ee")
e, XY = eom.run(estimator, ex_code="sd")

# 4. State-transfer operators for state n = 1
O1 = eom.gen_state_transfer_op(1)                                   # plain qEOM
U1 = eom.gen_filtered_state_transfer_op(1, Normalize=True,          # AqEOM
                                        Estimator=estimator)

# 5. Permanent dipole of state 1 from each operator
_, mu_qEOM  = Dipole.run(mf_qc, estimator, L=O1, R=O1)   # still needs 1/<0|O O†|0> (Eq. 11)
_, mu_AqEOM = Dipole.run(mf_qc, estimator, L=U1, R=U1)   # U is already normalized (Eq. 12)
```

`gen_filtered_state_transfer_op` also exists on `QSE.QSE` with the same signature, and accepts a
precomputed operator through `state_transfer_op=` if you already have $\hat O_n^\dagger$ in hand.
`Reproduce/Running/qEOM/H2.py` is the most complete worked example (normalization, spin-state selection,
overlaps, RDMs).

---

## Reproducing the paper

### Systems

All calculations use Jordan–Wigner mapping, UCCSD (COBYLA) and a singles+doubles qEOM manifold. States are
labelled by $\langle S^2\rangle$ (singlets, $\langle S^2\rangle<1$) and cross-checked against CASCI overlaps.
The CASCI reference always uses the same active space as qEOM.

| `system` | Molecule | Basis | Active space | Bond lengths (Å) | States | Used in |
| --- | --- | --- | --- | --- | --- | --- |
| `H2` | H₂ | STO-3G | full (2e, 2o) | 0.5–2.0 | 3 | Figs. 1, 4; Tables 1–2 |
| `HeH` | HeH⁺ | STO-3G | full (2e, 2o) | 0.5–2.0 | 3 | Fig. 3a; Table 1 |
| `LiH` | LiH | STO-3G | (2e, 5o) | 1.0–2.0 | 4 | Figs. 2, 3b; Tables 1–2 |
| `BeH2` | BeH₂ (linear) | STO-3G | (2e, 5o) | 0.7–1.6 | 3 | Fig. 3c |
| `NH3` | NH₃ | STO-3G | (2e, 4o) | 0.5–2.0 | 3 | Fig. 3d; Table 1 |
| `H2O` | H₂O | STO-3G | (8e, 6o) | 0.7–1.6 | 3 | Fig. 3e; Table 2 |
| `HF` | HF | STO-3G | (8e, 5o) | 0.5–2.0 | 3 | Figs. 1d, 3f; Table 1 |
| `H2_Large` | H₂ | 6-31G | full | 0.5–2.0 | 3 | Fig. S15 |
| `H2_Huge` | H₂ | 6-311G | full | 0.5–2.0 | 3 | Fig. S16 |
| `LiH_Large` | LiH | 6-31G | (2e, 5o) | 1.0–2.0 | 4 | Fig. S17 |
| `H2O_Small` | H₂O | STO-3G | (8e, 5o) | 0.7–1.6 | 3 | SI.10 (QSE comparison) |
| `CH4` | CH₄ | STO-3G | (2e, 5o) | — | 3 | script only, not in the paper |

The heaviest atom sits at the origin. Molecule geometry and active space are **hard-coded in each script**
(look for `!!! Active Space Here !!!`); the scripts take only the bond length as `sys.argv[1]`.

### Step 0 — set paths

Every script and notebook has its paths stubbed out:

- `.py` / `.ipynb`: `master_dir = '***'` → the directory that will hold `Data/`
- `.sh`: `project="***"` → same directory

The launchers call `$project/Code/Running/<Task>/<system>.py`, i.e. they expect this repository's
`Reproduce/` directory to be available as `$project/Code/`. Either copy/symlink it there
(`ln -s /path/to/repo/Reproduce $project/Code`) or edit the path in the `.sh` files. Logs go to
`$project/Output/<system>/`.

Each launcher runs one job per bond length in the background, capped at `max_jobs` (32, or 10 for QSE and
Noise). **Edit `system_order` before running**: it is currently set to a subset (e.g. `qEOM.sh` runs only
`H2O_Small`, `QSE.sh` only `LiH_QSE`); the full list is in the comment next to it and in `sys_dists`.

### Step 1 — qEOM / AqEOM production run

```bash
bash Reproduce/Running/qEOM/qEOM.sh
```

For each system and bond length: RHF (saved to `Chkfile/`) → UCCSD → qEOM (`ex_code='sd'`) →
$\langle S^2\rangle$ for up to 8 qEOM states → singlet selection → for qEOM and AqEOM, direct-operator
energies, the dipole matrix, the overlap matrix (`VAC/`), 1-RDMs, and the Hermitian contamination. Each
stage has a `do_*` flag; setting it to `False` reloads the previous result from `Data/` instead of
recomputing it.

### Step 2 — CASCI reference

Run `Reproduce/Running/CASCI.ipynb` (one cell per system). It reloads the chkfiles from Step 1 and writes
`Energy/{dist}_CAS.npy`, `Dipole/{dist}_CAS.npy` and `Density/CASCI_{n}_{dist}.npy`. Step 1 must finish
first.

### Step 3 — supplementary calculations

| Launcher | Needs | Produces | Paper |
| --- | --- | --- | --- |
| `QSE/QSE.sh` | nothing (runs its own HF/UCCSD into `Data/{system}_QSE/`) | QSE energies, dipoles, $\langle S^2\rangle$ | Fig. 1, SI.10 |
| `DC/DC.sh` | Steps 1–2 (`Density/`, `Spin_Square/`, `Chkfile/`) | `Energy/{dist}_{CAS,CASCI_MATCH_qEOM,CASCI_MATCH_fqEOM}_{xc}.npy` for SVWN, BLYP, PBE, B97, r2SCAN, TPSS, M06-L, B3LYP, r2SCAN0, PBE0 | Table 2, Fig. 4a, SI.5 |
| `CASCI_Overlap/CASCI_Overlap.sh` | Step 1 (`Chkfile/`, `Res/`, `qEOM/`) | `CASCI_Overlap/{dist}_Ovl_{qEOM,AqEOM}.npy` | SI.9 |
| `Natural_Occupancy/Natural_Occupancy.sh` | Steps 1–2 | `Natural_Occupancy/*.npy` | SI.12 |
| `Origin_Shift/Origin_Shift.sh` | nothing; argument is the origin displacement (H₂: 0.0–1.0 Å, LiH: 0.0–2.0 Å) | same outputs as Step 1, plus CASCI via the last two cells of `CASCI.ipynb` | SI.11 |
| `Noise/Noise.sh` | nothing; argument is the noise level | qEOM, AqEOM and QSE energies/dipoles for H₂ @ 0.7 Å | Table S12, Fig. S18 |

Noise levels used by `Noise/H2_Noise.py` (depolarizing error on 1-/2-qubit gates, `qiskit_aer` `EstimatorV2`):

| Level | 1-qubit error | 2-qubit error | Shots |
| --- | --- | --- | --- |
| 0 | 1e-5 | 1e-4 | 100,000 |
| 1 | 1e-4 | 1e-3 | 10,000 |
| 2 | 5e-4 | 5e-3 | 4,096 |
| 3 | 1e-3 | 1e-2 | 1,000 |

`Noise.sh` currently runs levels `1 2 3`; add `0` to `sys_dists` for the full table.

### Step 4 — analysis

Set `master_dir` in each notebook under `Reproduce/Analysis/` and run it. Mapping to the manuscript:

| Notebook | Main text | SI |
| --- | --- | --- |
| `Fig1.ipynb` | Fig. 1(b)–(d) | |
| `Fig2.ipynb` | Fig. 2 | |
| `Fig3.ipynb` | Fig. 3 | |
| `Fig4.ipynb` | Fig. 4 | |
| `Statistics.ipynb` | Table 1 | |
| `SI.ipynb` | | Figs. S1–S33, Table S17 |
| `CSV.ipynb` | | Tables S1–S10 (CSV export) |

### `Data/` layout

```
{master_dir}/Data/{system}/
├── Chkfile/scf_{dist}                      PySCF RHF chkfile
├── Res/UCC_{dist}.npy                      UCCSD amplitudes
├── qEOM/{E,XY}_{dist}.npy                  qEOM eigenvalues and eigenvectors
├── QSE/{E,XY}_{dist}.npy                   (QSE runs only)
├── Spin_Square/{dist}_{qEOM,fqEOM,QSE}.npy <S^2> in raw qEOM state order
├── Energy/{dist}_CAS.npy                   CASCI
│         {dist}_CASCI_MATCH_qEOM.npy       direct-operator energy, qEOM   (Eq. 10)
│         {dist}_CASCI_MATCH_qEOM_Diag.npy  qEOM eigenvalue + E_0
│         {dist}_CASCI_MATCH_fqEOM.npy      direct-operator energy, AqEOM
│         {dist}_*_{xc}.npy                 DC-DFT energies
│         {dist}_QSE.npy
├── Dipole/{dist}_{CAS,CASCI_MATCH_qEOM,CASCI_MATCH_fqEOM,QSE}.npy   (N, N, 3), atomic units
├── VAC/{dist}_CASCI_MATCH_{qEOM,fqEOM}.npy state-overlap matrix S (Eqs. 14–15)
├── Density/CASCI_{n}_{dist}.npy, qEOM_{n}_{dist}.npy, fqEOM_{n}_{dist}.npy   1-RDMs
├── Hermitian_Contamination/{dist}.npy      ||O||_F, ||H||_F, ||A||_F per state
│                          C_{dist}.npy     C^H_n (Eq. 17)
├── CASCI_Overlap/{dist}_Ovl_{qEOM,AqEOM}.npy
└── Natural_Occupancy/*.npy
{master_dir}/Data/CSV/                      written by CSV.ipynb and SI.ipynb
```

For the noise runs, `{dist}` in file names is the noise level; for origin-shift runs it is the displacement.

**The `Data/` directory is not included in this repository.** Regenerate it with the steps above, or obtain
the raw `.npy` results from the authors. All numerical values reported in the paper are also tabulated in
the Supplementary Information.

---

## Citation

If you use this code, please cite the paper above. Correspondence: Eunji Sim (esim@yonsei.ac.kr).
