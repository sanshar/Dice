import numpy
import numpy.linalg
import math
import cmath, sys
#from pyscf import gto, scf, ao2mo

def findSiteInUnitCell(newsite, size, latticeVectors, sites):
    for a in range(-1, 2):
        for b in range(-1, 2):
            newsitecopy = [newsite[0]+a*size*latticeVectors[0][0]+b*size*latticeVectors[1][0], newsite[1]+a*size*latticeVectors[0][1]+b*size*latticeVectors[1][\
1]]
            for i in range(len(sites)):
                if ( abs(sites[i][0] - newsitecopy[0]) <1e-10 and abs(sites[i][1] - newsitecopy[1]) <1e-10):
                    return True, i

    return False, -1

sqrt2 = 2.**0.5
latticeV = [[0, sqrt2], [sqrt2, 0]]
unitCell = [[1./2., 1./2.], [1/2.+1./sqrt2, 1/2.+1./sqrt2]]
connectedSteps = [[-1./sqrt2,-1./sqrt2], [-1./sqrt2,1./sqrt2], [1./sqrt2,-1./sqrt2], [1./sqrt2,1./sqrt2]] #all the moves from a site to a interacting site
size = int(sys.argv[1])
U = 4.0
uhf = True

sites = []

for i in range(size):
    for j in range(size):
        sites.append([unitCell[0][0]+i*latticeV[0][0]+j*latticeV[1][0], unitCell[0][1]+i*latticeV[0][1]+j*latticeV[1][1]])
        sites.append([unitCell[1][0]+i*latticeV[0][0]+j*latticeV[1][0], unitCell[1][1]+i*latticeV[0][1]+j*latticeV[1][1]])

nsites =  len(sites)
print nsites

integrals = open("FCIDUMP", "w")
integrals.write("&FCI NORB=%d ,NELEC=%d ,MS2=0,\n"%(nsites, nsites))
integrals.write("ORBSYM=")
for i in range(nsites):
    integrals.write("%1d,"%(1))
integrals.write("\n")
integrals.write("ISYM=1,\n")
integrals.write("&END\n")


int1 = numpy.zeros(shape=(nsites, nsites))
int2 = numpy.zeros(shape=(nsites, nsites, nsites, nsites))
fockUp = numpy.zeros(shape=(nsites, nsites))
fockDn = numpy.zeros(shape=(nsites, nsites))

for i in range(len(sites)):
    for step in connectedSteps:
        newsite = [sites[i][0]+step[0], sites[i][1]+step[1]]

        #make all possible steps to try to come back to the lattice
        found, index = findSiteInUnitCell(newsite, size, latticeV, sites)

        if (found):
            int1[i,index] = -1.0
            fockUp[i, index] = -1.0
            fockDn[i, index] = -1.0
            integrals.write("%16.8f  %4d   %4d   %4d   %4d \n"%(-1., i+1, index+1, 0, 0))
            #print -1.0, i+1,index+1,0,0

for i in range(len(sites)):
    int2[i,i,i,i] = U
    integrals.write("%16.8f  %4d   %4d   %4d   %4d \n"%(4., i+1, i+1, i+1, i+1))

#energies = numpy.zeros(shape=(size, size))
m = 0.3
n = nsites/2

for i in range(nsites):
    fockUp[i, i] = U*(n-m*(-1)**(i+1))
    fockDn[i, i] = U*(n+m*(-1)**(i+1))

ekUp, orbUp = numpy.linalg.eigh(fockUp)
ekDn, orbDn = numpy.linalg.eigh(fockDn)

fileHF = open("hf.txt", 'w')
for i in range(nsites):
    for j in range(nsites):
        fileHF.write('%16.10e '%(orbUp[i,j]))
    for j in range(nsites):
        fileHF.write('%16.10e '%(orbDn[i,j]))
    fileHF.write('\n')


#mol = gto.M(verbose=4)
#mol.nelectron = nsites
#mol.incore_anyway = True
#mf = scf.RHF(mol)
#
#mf.get_hcore = lambda *args: int1
#mf.get_ovlp = lambda *args: numpy.eye(nsites)
#mf._eri = ao2mo.restore(8, int2, nsites)
#mf.level_shift_factor = 0.04
#mf.max_cycle = 200
#print mf.kernel()
#print mf.mo_energy
#
#fileHF = open("rhf.txt", 'w')
#for i in range(nsites):
#    for j in range(nsites):
#        fileHF.write('%16.10e '%(mf.mo_coeff[i,j]))
#    fileHF.write('\n')
#
#delta = 1
#
#fileHF = open("hf.txt", 'w')
#for i in range(nsites):
#    for j in range(nsites):
#        uj = (1-mf.mo_energy[j]/(delta**2+mf.mo_energy[j]**2))**0.5/sqrt2
#        vj = (1+mf.mo_energy[j]/(delta**2+mf.mo_energy[j]**2))**0.5/sqrt2
#        coeff = uj*mf.mo_coeff[i,j] + (-1)**(i)*vj*mf.mo_coeff[i,j]
#        fileHF.write('%16.10e '%(coeff))
#    for j in range(nsites):
#        uj = (1-mf.mo_energy[j]/(delta**2+mf.mo_energy[j]**2))**0.5/sqrt2
#        vj = (1+mf.mo_energy[j]/(delta**2+mf.mo_energy[j]**2))**0.5/sqrt2
#        coeff = uj*mf.mo_coeff[i,j] - (-1)**(i)*vj*mf.mo_coeff[i,j]
#        fileHF.write('%16.10e '%(coeff))
#    fileHF.write('\n')
#
correlators = []
for i in range(nsites):
    c = [i]
    for j in range(nsites):
        if (int1[i,j] != 0):
            c.append(j)
    c = list(set(c))
    correlators.append(c)
f = open("correlators.txt", 'w')
for c in correlators:
    for t in c:
        f.write("%d  "%(t))
    f.write("\n")

