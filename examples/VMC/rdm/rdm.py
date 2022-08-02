import os
import numpy as np
import scipy.linalg as la
from pyscf import gto, scf, tools, fci
import QMCUtils

# change this if running elsewhere
vmc_binary = "../../../bin/VMC"

r = 2.0
atomstring = ""
for i in range(6):
  atomstring += "H 0 0 %g\n"%(i*r)
mol = gto.M(atom=atomstring, basis='sto-6g', unit='bohr')
mf = scf.RHF(mol)
mf.kernel()
norb = mol.nao

# local orbitals
lmo = QMCUtils.localizeAllElectron(mf, "lowdin")
QMCUtils.writeFCIDUMP(mol, mf, lmo, fname = 'FCIDUMP')

# fci rdm's for comparison
mf.mo_coeff = lmo
cisolver = fci.FCI(mf)
fci_ene, fci_vec = cisolver.kernel()
print(f'fci_ene: {fci_ene}')
(dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = cisolver.make_rdm12s(fci_vec, norb, mol.nelec)


# preparing ghf jastrosalter calculation
# ghf
gmf = scf.GHF(mol)
gmf.kernel()
mo1 = gmf.stability()
dm1 = gmf.make_rdm1(mo1, gmf.mo_occ)
gmf = gmf.run(dm1)
gmf.stability()

# write mo coeffs
overlap = mf.get_ovlp(mol)
overlap_b = la.block_diag(overlap, overlap)
lmo_b = la.block_diag(lmo, lmo)
ghf_coeffs = lmo_b.T.dot(overlap_b).dot(gmf.mo_coeff)
np.savetxt("hf.txt", ghf_coeffs)


# deterministic calculation with jastrowslater
QMCUtils.write_vmc_input(wavefunction_name = "jastrowslater", hfType = "ghf", complexQ = True, deterministic = True, writeOneRDM = True, writeTwoRDM = True, fname = "vmcDet.json")
os.system(f'''
              export OMP_NUM_THREADS=1;
              rm BestDeterminant.txt -f;
              mpirun {vmc_binary} vmcDet.json > vmcDet.out;
              mv oneRDM.dat oneRDMDet.dat; mv twoRDM.dat twoRDMDet.dat
           ''')

# read rdm files
dm1_det = np.loadtxt('oneRDMDet.dat')
dm2_det = np.loadtxt('twoRDMDet.dat')
print(f'\ndm1_det_tr: {np.trace(dm1_det)}')
print(f'dm2_det_tr: {np.trace(dm2_det)}\n')


# stochastic calculation with jastrowslater
# initial det to start vmc run with
fileh = open("bestDet", 'w')
fileh.write('1.   a b a b a b\n')
fileh.close()

QMCUtils.write_vmc_input(wavefunction_name = "jastrowslater", hfType = "ghf", determinants = "bestDet", complexQ = True, seed = 414, stochasticIter = 1000, writeOneRDM = True, writeTwoRDM = True, fname = "vmcStoc.json")
os.system(f'''
              export OMP_NUM_THREADS=1;
              rm BestDeterminant.txt -f;
              mpirun {vmc_binary} vmcStoc.json > vmcStoc.out;
              mv oneRDM.dat oneRDMStoc.dat; mv twoRDM.dat twoRDMStoc.dat
           ''')

# read rdm files
dm1_stoc = np.loadtxt('oneRDMStoc.dat')
dm2_stoc = np.loadtxt('twoRDMStoc.dat')
print(f'\ndm1_stoc_tr: {np.trace(dm1_stoc)}')
print(f'dm2_stoc_tr: {np.trace(dm2_stoc)}\n')
dm1_stoc_err = np.loadtxt('oneRDMErr.dat')
dm2_stoc_err = np.loadtxt('twoRDMErr.dat')
