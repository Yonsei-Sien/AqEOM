import numpy as np


def run(V, eps):
    #  [Input]   V : 4D Tensor to decompose
    #  [Input] eps : Require accuracy of Cholesky decomposition
    # [Output]   L : Decomposed 2D matrices
    # [Output]  ng : Number of Cholesky vectors
    # Cholesky Decomposition
    # For lowering the circuit depth

    no = V.shape[0]
    chmax, ng = 20 * no, 0
    W = V.reshape(no**2, no**2)
    L = np.zeros((no**2, chmax))
    Dmax = np.diagonal(W).copy()
    nu_max = np.argmax(Dmax)
    vmax = Dmax[nu_max]
    while vmax > eps:
        L[:, ng] = W[:, nu_max]
        if ng > 0:
            L[:, ng] -= np.dot(L[:, 0:ng], (L.T)[0:ng, nu_max])
        L[:, ng] /= np.sqrt(vmax)
        Dmax[: no**2] -= L[: no**2, ng] ** 2
        ng += 1
        nu_max = np.argmax(Dmax)
        vmax = Dmax[nu_max]
    L = L[:, :ng].reshape((no, no, ng))
    print(
        "Accuracy of Cholesky decomposition ",
        np.abs(np.einsum("prg,qsg->prqs", L, L) - V).max(),
    )
    return L, ng


def run_ng_fixed(V, eps, trg_ng):
    #  [Input]      V : 4D Tensor to decompose
    #  [Input]    eps : Require accuracy of Cholesky decomposition
    #  [Input] trg_ng : Target number of Cholesky vector
    # [Output]      L : Decomposed 2D matrices
    # [Output]     ng : Number of Cholesky vectors
    # Cholesky decomposition with fixed number of Cholesky vector
    # Sometimes... it is useful...

    no = V.shape[0]
    chmax, ng = 20 * no, 0
    W = V.reshape(no**2, no**2)
    L = np.zeros((no**2, chmax))
    Dmax = np.diagonal(W).copy()
    nu_max = np.argmax(Dmax)
    vmax = Dmax[nu_max]
    while vmax > eps and ng < trg_ng:
        L[:, ng] = W[:, nu_max]
        if ng > 0:
            L[:, ng] -= np.dot(L[:, 0:ng], (L.T)[0:ng, nu_max])
        L[:, ng] /= np.sqrt(vmax)
        Dmax[: no**2] -= L[: no**2, ng] ** 2
        ng += 1
        nu_max = np.argmax(Dmax)
        vmax = Dmax[nu_max]
    L = L[:, :ng].reshape((no, no, ng))
    print(
        "Accuracy of Cholesky decomposition ",
        np.abs(np.einsum("prg,qsg->prqs", L, L) - V).max(),
    )
    return L, ng