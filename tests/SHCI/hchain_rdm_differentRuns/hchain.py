import numpy as np
import scipy
from pyscf import scf, gto, ao2mo, fci, lib, mcscf
import shci, PostProcessStates
from mpi4py import MPI

world_comm = MPI.COMM_WORLD
world_size = world_comm.Get_size()
my_rank = world_comm.Get_rank()


# INPUT PARAMETERS
natom = 8 
# The number of states to compute for each hamiltonian
nstates = 2
bond_lens = [1., 1.2]

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

all_ci_hf = []
for i in range(len(bond_lens)):
    ##set up molecule and perform HF calculation
    bond_length = 1.
    mol = gto.Mole()
    mol.build(atom = chain("H", natom, bond_length=bond_lens[i]),basis = "sto-3g", unit = 'B', verbose=4)
    mf = scf.RHF(mol)
    mf.kernel()
    
    ##do fci calculation using pyscf solver
    ci = fci.direct_spin1.FCISolver(mol)
    h1e = np.einsum('ai,ab,bj->ij', mf.mo_coeff, mf.get_hcore(), mf.mo_coeff)
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), mol.nao)
    e_fci, ci_hf = ci.kernel(h1e, eri, natom, natom, nroots=nstates)
    all_ci_hf.append(ci_hf)
    
    ##set up CASCI calculation with HCI
    mc2 = mcscf.CASCI(mf, natom, natom)
    mc2.fcisolver = shci.SHCI(mol)
    mc2.fcisolver.sweep_iter = [0]
    mc2.fcisolver.sweep_epsilon = [1.e-14]
    mc2.fcisolver.scratchDirectory = "."
    mc2.fcisolver.nroots = nstates
    mc2.fcisolver.davdisonTol = 1.e-10
    mc2.fcisolver.prefix = f"run{i}"
    
    ##one has to give an initial determinant for all three states so it finds the correct FCI state
    mc2.fcisolver.initialStates=[[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,9],[0,1,2,3,4,5,6,11]]
    mc2.kernel(mf.mo_coeff)
    
    print(e_fci + mf.energy_nuc())


##after running the calculations let us check the rdms
##first generate the rdm using dice from two different directories
print("\n\n CALCULATE TRANSITION RDM BETWEEN DIFFERENT RUNS\n\n")
PostProcessStates.makeTransitionRDM("run0", "run1", mol.nao, mol.nelectron)


for i in range(2*nstates):
    for j in range(i+1):
        run1, state1 = i//nstates, i%nstates
        run2, state2 = j//nstates, j%nstates

        print(f"Test between   run:{run1},state:{state1}  and  run:{run2},state:{state2} ")
        ##evaluate 1 and 2 RDM
        rdm1, rdm2 = ci.trans_rdm12(all_ci_hf[run1][state1], all_ci_hf[run2][state2], natom,natom)
        #rdm2 = 1.*np.ascontiguousarray(np.einsum('pqrs->prqs', rdm2))
        rdm1 = 1.*np.ascontiguousarray(rdm1.T)
        #rdm2 = 1.*np.einsum('pqrs->prqs', rdm2)
        #rdm1 = 1.*rdm1.T
        #rdm1 = np.load(f"pyscf_1RDM{i}{j}.npy")
        #rdm2 = np.load(f"pyscf_2RDM{i}{j}.npy")
        
        rdm2hci = read2RDM(f"spatialRDM.{state1+run1*nstates}.{state2+run2*nstates}.txt")
        rdm1hci = read1RDM(f"spatial1RDM.{state1+run1*nstates}.{state2+run2*nstates}.txt")

        if (i != j):
            a,b = np.unravel_index(abs(rdm1).argmax(), rdm1.shape)
            sgn = np.sign(rdm1[a,b]/rdm1hci[a,b])
            rdm1, rdm2 = sgn*rdm1, sgn*rdm2

        np.save(f"pyscf_1RDM{i}{j}.npy", rdm1)
        np.save(f"pyscf_2RDM{i}{j}.npy", rdm2)

        rdm2 = 1.*np.ascontiguousarray(np.einsum('pqrs->prqs', rdm2))

        ##check of internal consistency
        #assert(np.allclose(np.einsum('prqr->pq',rdm2hci)/(natom-1), rdm1hci, atol=1.e-5))

        print("max difference in RDM2 {0:8.3e}".format(np.max(abs(rdm2hci-rdm2))))
        print()
        #print("max difference in RDM1 ", np.max(abs(rdm1-rdm1.T)))
        #assert(np.allclose(rdm2hci, rdm2, atol=1.e-5))
        #assert(np.allclose(rdm1hci, rdm1.T, atol=1.e-5))
