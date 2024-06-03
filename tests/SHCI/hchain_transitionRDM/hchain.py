import numpy as np
import scipy
from pyscf import scf, gto, ao2mo, fci, lib, mcscf
import shci

# INPUT PARAMETERS
natom = 8 
# The number of states to compute for each hamiltonian
nstates = 3


def chain(atom, natom, bond_length, numbering=None):
    """Open boundary condition version of 1D ring"""
    atoms = []
    if isinstance(atom, str):
        atom = [atom]
    for i in range(natom):
        atom_i = atom[i % len(atom)]
        if numbering is not None:
            atom_i += str(int(numbering) + i)
        atoms.append([atom_i, np.asarray([i * bond_length, 0.0, 0.0])])
    return atoms


##set up molecule and perform HF calculation
bond_length = 1.
mol = gto.Mole()
mol.build(atom = chain("H", natom, bond_length=1.),basis = "sto-3g", unit = 'B', verbose=4)
mf = scf.RHF(mol)
mf.kernel()

##do fci calculation using pyscf solver
ci = fci.direct_spin1.FCISolver(mol)
h1e = np.einsum('ai,ab,bj->ij', mf.mo_coeff, mf.get_hcore(), mf.mo_coeff)
eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), mol.nao)
e_fci, ci_hf = ci.kernel(h1e, eri, natom, natom, nroots=nstates)

##evaluate 1 and 2 RDM
rdm1, rdm2 = ci.trans_rdm12(ci_hf[2], ci_hf[0], natom,natom)

##set up CASCI calculation with HCI
mc2 = mcscf.CASCI(mf, natom, natom)
mc2.fcisolver = shci.SHCI(mol)
mc2.fcisolver.sweep_iter = [0]
mc2.fcisolver.sweep_epsilon = [1e-10]
mc2.fcisolver.scratchDirectory = "."
mc2.fcisolver.nroots = 3

##one has to give an initial determinant for all three states so it finds the correct FCI state
mc2.fcisolver.initialStates=[[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,9],[0,1,2,3,4,5,6,11]]
mc2.kernel(mf.mo_coeff)

##function to read the 2-RDM
def read2RDM(fname):
    f = open(fname)

    lines = f.readlines()
    norbs = int(lines[0])
    rdm2 = np.zeros((norbs, norbs, norbs, norbs))
    for l in lines[1:]:
        tokens = l.split()
        val, i, j, k, l = float(tokens[4]), int(tokens[0]), int(tokens[1]), int(tokens[2]), int(tokens[3])

        rdm2[i,j,k,l] = val
    return rdm2

##function to read the 1-RDM
def read1RDM(fname):
    f = open(fname)

    lines = f.readlines()
    norbs = int(lines[0])
    rdm1 = np.zeros((norbs, norbs))
    for l in lines[1:]:
        tokens = l.split()
        val, i, j = float(tokens[2]), int(tokens[0]), int(tokens[1])

        rdm1[i,j] = val
    return rdm1


rdm2hci = read2RDM("spatialRDM.2.0.txt")
rdm1hci = read1RDM("spatial1RDM.2.0.txt")

import pdb
pdb.set_trace()
##check of internal consistency
assert(np.allclose(np.einsum('prqr->pq',rdm2hci)/(natom-1), rdm1hci, atol=1.e-5))

##check consistency with pyscf rdm
##the abs is included because either of the two states coming from Dice and Pyscf can have different
##signs
assert(np.allclose(abs(rdm1), abs(rdm1hci).T, atol=1.e-5))  
assert(np.allclose(abs(rdm2), abs(-np.einsum('pqrs->prqs', rdm2hci)), atol=1.e-5))
