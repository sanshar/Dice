import numpy
import math
from pyscf import gto, scf, ao2mo, mcscf, tools, fci
from pyscf.shciscf import shci, settings
from pyscf.lo import pipek

r = 0.529177
atomstring = ""
for i in range(20):
    atomstring += "H 0 0 %g\n"%(i*r)

mol = gto.M(
    atom = atomstring,
    basis = 'sto-6g',
    verbose=2,
    symmetry=0,
    spin = 0)
myhf = scf.RHF(mol)
myhf.kernel()

print myhf.e_tot
#localized orbitals
lmo = pipek.PM(mol).kernel(myhf.mo_coeff)

#print the atom with which the lmo is associated
for i in range(lmo.shape[1]):
	print numpy.argmax(numpy.absolute(lmo[:,i])), 
print

#this gives the mo coeffficients in terms of lmo
S = myhf.get_ovlp(mol)
uc = reduce(numpy.dot, (myhf.mo_coeff.T, S, lmo)).T

#write the UC(lmo, mo) to disk
norbs = myhf.mo_coeff.shape[0]
fileHF = open("hf.txt", 'w')
for i in range(norbs):
    for j in range(norbs):
        fileHF.write('%16.10e '%(uc[i,j]))
    fileHF.write('\n')

#start a casci calculation for printing FCIDUMP file
mc = mcscf.CASCI(myhf, myhf.mo_coeff.shape[0], 20)
mc.fcisolver = shci.SHCI(mol)
mc.kernel(lmo)
