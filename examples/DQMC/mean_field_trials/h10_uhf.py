# H10 chain ground state

from pyscf import gto, scf
import QMCUtils
import numpy

# this needs to be provided
vmc_root = "/projects/anma2640/VMC/master/VMC"

# mean field calculation
r = 2.0
nH = 10
atomstring = ""
for i in range(nH):
  atomstring += f"H 0 0 {r * i + r * (-nH + 1) / 2.}\n"
mol = gto.M(atom = atomstring, basis = 'sto-6g', unit = 'bohr', verbose = 3, symmetry = 0)
mf = scf.RHF(mol)
print('RHF calculation:')
mf.kernel()

print('\nUHF calculation:')
norb = mol.nao
dm = [numpy.zeros((norb, norb)), numpy.zeros((norb, norb))]
for i in range(norb//2):
  dm[0][2*i, 2*i] = 1.
  dm[1][2*i+1, 2*i+1] = 1.
umf = scf.UHF(mol)
umf.kernel(dm)

# afqmc calculation
QMCUtils.run_afqmc(umf, vmc_root = vmc_root, burn_in = 100)

