import numpy as np
import math
from pyscf import gto, scf, ao2mo, mcscf, tools, fci, mp
#from pyscf.shciscf import shci, settings
from pyscf.lo import pipek, boys
import sys
from scipy.linalg import fractional_matrix_power

UHF = False
r = float(sys.argv[1])*0.529177
n = 10
order = 2
sqrt2 = 2**0.5

atomstring = ""
for i in range(n):
    atomstring += "H 0 0 %g\n"%(i*r)

mol = gto.M(
    atom = atomstring,
    basis = 'sto-6g',
    verbose=4,
    symmetry=0,
    spin = 0)
mf = scf.RHF(mol)
if (UHF) :
    mf = scf.UHF(mol)
print mf.kernel()

if UHF :
    mocoeff = mf.mo_coeff[0]
else:
    mocoeff = mf.mo_coeff

lmo = fractional_matrix_power(mf.get_ovlp(mol), -0.5).T
#lmo = pipek.PM(mol).kernel(mocoeff)
h1 = lmo.T.dot(mf.get_hcore()).dot(lmo)
eri = ao2mo.kernel(mol, lmo)
tools.fcidump.from_integrals('FCIDUMP', h1, eri, 10, 10, mf.energy_nuc())

print mf.mo_coeff
print "local"
print lmo

#print the atom with which the lmo is associated
orbitalOrder = []
for i in range(lmo.shape[1]):
	orbitalOrder.append(np.argmax(np.absolute(lmo[:,i])) )
print orbitalOrder

ovlp = mf.get_ovlp(mol)
uc = (mocoeff.T.dot(ovlp).dot(lmo)).T

#fileh = open("hf.txt", 'w')
#for i in range(10):
#    for j in range(10):
#        fileh.write('%16.10e '%(uc[i,j]))
#    fileh.write('\n')
#fileh.close()

#fileHF = open("hf.txt", 'w')
#for i in range(10):
#    for j in range(5):
#        fileHF.write('%16.10e '%(uc[i,j]/sqrt2))
#    for j in range(5):
#        fileHF.write('%16.10e '%(uc[i,j]/sqrt2))
#    for j in range(5):
#        fileHF.write('%16.10e '%(uc[i,j+5]/sqrt2))
#    for j in range(5):
#        fileHF.write('%16.10e '%(uc[i,j+5]/sqrt2))
#    fileHF.write('\n')
#for i in range(10):
#    for j in range(5):
#        fileHF.write('%16.10e '%(uc[i,j]/sqrt2))
#    for j in range(5):
#        fileHF.write('%16.10e '%(-uc[i,j]/sqrt2))
#    for j in range(5):
#        fileHF.write('%16.10e '%(uc[i,j+5]/sqrt2))
#    for j in range(5):
#        fileHF.write('%16.10e '%(-uc[i,j+5]/sqrt2))
#    fileHF.write('\n')
#fileHF.close()
#

diag = np.zeros(shape=(10,10))
for i in range(5):
  diag[i,i] = 1.
pairMat = uc.dot(diag).dot(uc.T)
randMat = np.random.rand(20,20)
randMat = (randMat - randMat.T)/10
filePfaff = open("pairMat.txt", 'w')
for i in range(n):
    for j in range(n):
        filePfaff.write('%16.10e '%(randMat[i,j]))
    for j in range(n):
        filePfaff.write('%16.10e '%(pairMat[i,j]+randMat[i,j+n]))
    filePfaff.write('\n')
for i in range(n):
    for j in range(n):
        filePfaff.write('%16.10e '%(-pairMat[i,j]+randMat[i+n,j]))
    for j in range(n):
        filePfaff.write('%16.10e '%(randMat[i+n,j+n]))
    filePfaff.write('\n')
filePfaff.close()

