from qiskit.circuit import QuantumCircuit


def HF(N_orb, N_alpha, N_beta, mapping='jordan_wigner'):
    #  [Input]   N_orb : Number of spatial orbitals
    #  [Input] N_alpha : Number of alpha electrons
    #  [Input]  N_beta : Number of beta electrons
    #  [Input] mapping : Mapping method (jordan_wigner, parity)
    # [Output] circuit : Quantum circuit with HF state suitable to mapping algorithm
    circuit = QuantumCircuit(2*N_orb)
    if mapping == 'jordan_wigner':
        # 1111....00000 @ 1111...00000
        #    (alpha)         (beta)
        for i in range(N_alpha):
            circuit.x(i)
        for j in range(N_beta):
            circuit.x(N_orb + j)
    elif mapping == 'parity':
        # each qubit saves SUM(Number of electron up to current spin-orbital) modular 2
        for i in range(N_alpha):
            if (i + 1) % 2:
                circuit.x(i)
        for j in range(N_beta):
            if (N_alpha + j + 1) % 2:
                circuit.x(N_orb + j)
        if N_alpha % 2:
            for k in range(N_alpha, N_orb):
                circuit.x(k)
        if (N_alpha + N_beta) % 2:
            for l in range(N_beta, N_orb):
                circuit.x(N_orb + l)
    #elif mapping == 'bravyi_kitaev':
    else:
        raise ValueError("Unsupported mapping.")
    return circuit


def Bitstring(N_orb, Bitstring, mapping='jordan_wigner'):
    #  [Input]    N_orb : Number of spatial orbitals
    #  [Input] Bitsting : Bitstring with 1:occ. and 0:virt  |(beta) Nth N-1th ... 2nd 1st (alpha)> --> "1st 2nd ... N-1 N" with no spacebar
    #  [Input]  mapping : Mapping method (jordan_wigner, parity)
    # [Output]  circuit : Quantum circuit with HF state suitable to mapping algorithm
    circuit = QuantumCircuit(2*N_orb)
    if mapping == 'jordan_wigner':
        for ind, val in enumerate(Bitstring):
            if val == '1':
                circuit.x(ind)
    elif mapping == 'parity':
        N_occ = 0
        for ind, val in enumerate(Bitstring):
            if val == '1':
                N_occ += 1
            if N_occ % 2:
                circuit.x(ind)
    else:
        raise ValueError("Unsupported mapping.")
    return circuit