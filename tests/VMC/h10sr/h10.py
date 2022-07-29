import numpy
import math
from pyscf import gto, scf, ao2mo, mcscf, tools, fci, mp
from pyscf.shciscf import shci, settings
from pyscf.lo import pipek, boys
import sys

UHF = False
r = float(sys.argv[1])*0.529177
n = 10
order = 5

atomstring = ""
for i in range(n):
    atomstring += "H 0 0 %g\n"%(i*r)

mol = gto.M(
    atom = atomstring,
    basis = 'sto-6g',
    verbose=4,
    symmetry=0,
    spin = 0)
myhf = scf.RHF(mol)
if (UHF) :
    myhf = scf.UHF(mol)
print myhf.kernel()

if UHF :
    mocoeff = myhf.mo_coeff[0]
else:
    mocoeff = myhf.mo_coeff

lmo = pipek.PM(mol).kernel(mocoeff)

#print the atom with which the lmo is associated
orbitalOrder = []
for i in range(lmo.shape[1]):
	orbitalOrder.append(numpy.argmax(numpy.absolute(lmo[:,i])) )
print orbitalOrder

norbs = len(orbitalOrder)
f = open("correlators.txt", 'w')

#print sorted(orbitalOrder)
reorder = []
for i in range(norbs):
    reorder.append(orbitalOrder.index(i))
    
for i in range(norbs-order+1):
    l =[]
    for j in range(order):
        l.append(reorder[i+j])
    l = sorted(l)
    for j in range(order):
        f.write("%d "%(l[j]))
    f.write("\n")
f.close()


#this gives the mo coeffficients in terms of lmo
S = myhf.get_ovlp(mol)
uc = reduce(numpy.dot, (mocoeff.T, S, lmo)).T

if UHF :
    uc2 = reduce(numpy.dot, (myhf.mo_coeff[1].T, S, lmo)).T
else:
    uc2 = uc

norbs = mocoeff.shape[0]
print norbs
#write the UC(lmo, mo) to disk
fileHF = open("hf.txt", 'w')
for i in range(norbs):
    print i, norbs
    for j in range(norbs):
        fileHF.write('%16.10e '%(uc[i,j]))
    if (UHF) :
        for j in range(norbs):
            fileHF.write('%16.10e '%(uc2[i,j]))
    fileHF.write('\n')

#start a casci calculation for printing FCIDUMP file
mc = mcscf.CASCI(myhf, mocoeff.shape[0], n)
mc.fcisolver = shci.SHCI(mol)
mc.fcisolver.sweep_iter =[0]
mc.fcisolver.sweep_epsilon=[10]
mc.kernel(lmo)
