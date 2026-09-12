# AqEOM — Anti-Hermitized quantum Equation-of-Motion

> Code written by Gyumin Kim

The repository contains `pyqake`, a small Qiskit + PySCF toolkit for UCC / qEOM / QSE excited-state
calculations with direct observable evaluation, and the notebooks used to produce the figures of the paper.

The two PDFs in the repository root are the manuscript and its supplementary information.
**Note that they are currently swapped:** `SI.pdf` is the main text (13 pp.) and `Main.pdf` is the
Supplementary Materials (38 pp.).

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
│       ├── Dipole.py        electronic + nuclear dipole operators, <0|L† μ R|0> evaluation
│       ├── RDM.py           n-body RDM operator generation
│       └── Spin_Square.py   <S^2> operators
├── lib/
│   ├── Transpile.py             PySCFTranspiler (MO integrals, active space) + Qiskit
│   │                            Estimator/Sampler cost functions
│   ├── MO2AO.py                 RDM basis transform, active→full space restoration
│   └── Cholesky_Decomposition.py  ERI Cholesky decomposition
└── Reproduce/
    └── Analysis/
        ├── CSV.ipynb    dumps the raw .npy results to the CSV tables in the SI
        ├── Fig1.ipynb   H2: energies and dipoles, CASCI / QSE / qEOM / AqEOM
        ├── Fig2.ipynb   LiH: error correlations (energy error, GS leakage,
        │                Hermitian contamination, dipole error)
        ├── Fig3.ipynb   dissociation curves and RMSE across the seven systems
        └── Fig4.ipynb   DC-DFT RDM diagnostics (r2SCAN) and overlap matrices
```

### Key entry points

| Object | Purpose |
| --- | --- |
| `Transpile.PySCFTranspiler(mf, active_space, active_e)` | Turns a PySCF `scf`/`dft` object (or chkfile) into MO integrals with an optional (Ne, No) active space |
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
- NumPy, SciPy
- Jupyter + Matplotlib for the notebooks in `Reproduce/`

```bash
pip install pyscf qiskit qiskit-aer numpy scipy matplotlib jupyter
```

There is no packaging metadata; run from the repository root (or put it on `PYTHONPATH`) so that
`import pyqake` resolves.

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
mf_qc = UCC.UCC(ref, ex_code="sd")
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
mu_nuc, mu_qEOM  = Dipole.run(mf_qc, estimator, L=O1, R=O1)
mu_nuc, mu_AqEOM = Dipole.run(mf_qc, estimator, L=U1, R=U1)
```

`gen_filtered_state_transfer_op` also exists on `QSE.QSE` with the same signature, and accepts a
precomputed operator through `state_transfer_op=` if you already have $\hat O_n^\dagger$ in hand.

---

## Reproducing the paper

The notebooks in `Reproduce/Analysis/` read pre-computed results from a `Data/` tree:

```
{master_dir}/Data/{system}/Energy/{dist}_{CAS|CASCI_MATCH_qEOM|CASCI_MATCH_fqEOM}.npy
{master_dir}/Data/{system}/Dipole/...
{master_dir}/Data/{system}/Hermitian_Contamination/C_{dist}.npy
{master_dir}/Data/{system}/VAC/...
```

**That `Data/` directory is not included in this repository**, and every notebook has its path stubbed
out as `master_dir = "***"`. To rerun the analysis you must either obtain the raw `.npy` results from the
authors or regenerate them with `pyqake` and point `master_dir` at your own tree. All the numerical values
themselves are tabulated in the Supplementary Information.
