import numpy
import math
from pyscf import gto, scf, ao2mo, mcscf, tools, fci
from pyscf.shciscf import shci, settings
from pyscf.lo import pipek

mol = gto.M(
    atom = 'H 0 0 0; H 0 0 1.0;H 0 0 2.0;H 0 0 3.0;H 0 0 4.0;H 0 0 5.0;',
    basis = 'sto-3g',
    verbose=5,
    spin = 0)
myhf = scf.RHF(mol)
myhf.kernel()

print myhf.e_tot

mc = mcscf.CASCI(myhf, myhf.mo_coeff.shape[0], 6)
mc.fcisolver = shci.SHCI(mol)
mc.fcisolver.sweep_iter=[0]
mc.fcisolver.sweep_epsilon=[0.12]
print mc.kernel(myhf.mo_coeff)



lmo = pipek.PM(mol).kernel(myhf.mo_coeff)
S = myhf.get_ovlp(mol)

#uc(lmo, mo)
uc = reduce(numpy.dot, (myhf.mo_coeff.T, S, lmo)).T

norbs = myhf.mo_coeff.shape[0]
fileHF = open("hf.txt", 'w')
for i in range(norbs):
    for j in range(norbs):
        fileHF.write('%16.10e '%(uc[i,j]))
    fileHF.write('\n')

mc.fcisolver.onlywriteIntegral = True
mc.kernel(lmo)

