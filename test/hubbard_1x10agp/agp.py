import numpy as np

xmax, ymax = 1, 10
nsites = xmax*ymax
U = 4.
corrLen = 1
sqrt2 = 2**0.5

data = []
for i in range(xmax):
  for j in range(ymax):
    data.append([i,j])


integralsFh = open("FCIDUMP", "w")
integralsFh.write("&FCI NORB=%d ,NELEC=%d ,MS2=0,\n"%(nsites, nsites))
integralsFh.write("ORBSYM=")
for i in range(nsites):
  integralsFh.write("%1d,"%(1))
integralsFh.write("\n")
integralsFh.write("ISYM=1,\n")
integralsFh.write("&END\n")

int2 = np.zeros((nsites, nsites, nsites, nsites))
int1 = np.zeros((nsites, nsites))
fock = np.zeros((nsites, nsites))

for i in range(nsites):
  int2[i,i,i,i] = U
  integralsFh.write("%16.8f  %4d   %4d   %4d   %4d \n"%(U, i+1, i+1, i+1, i+1))

for i in range(nsites):
  for j in range(i+1, nsites):
    d1, d2 = data[i], data[j]
    if ( (abs(d1[0] - d2[0]) == 1 or abs(d1[0] - d2[0]) == xmax-1) and d1[1]== d2[1]) :
      int1[i, j] = int1[j, i] = -1.
      fock[i, j] = fock[j, i] = -1.
      integralsFh.write("%16.8f  %4d   %4d   %4d   %4d \n"%(-1., i+1, j+1, 0, 0))
    elif ( (abs(d1[1] - d2[1]) == 1 or abs(d1[1] - d2[1]) == ymax-1) and d1[0]== d2[0]) :
      int1[i, j] = int1[j, i] = -1.
      fock[i, j] = fock[j, i] = -1.
      integralsFh.write("%16.8f  %4d   %4d   %4d   %4d \n"%(-1., i+1, j+1, 0, 0))

integralsFh.close()

correlators=[]

#all nearest neighbors
#for i in range(nsites):
#    c = [i]
#    for j in range(nsites):
#        if (int1[i,j] != 0):
#            c.append(j)
#    c = list(set(c))
#    correlators.append(c)

#neighbor pairs for 1 d
for i in range(nsites - 1):
  correlators.append([i, i + 1])
correlators.append([nsites - 1, 0])

#all pairs
#for i in range(nsites):
#  for j in range(nsites - 1 - i):
#    correlators.append([i, i + j + 1])

correlatorsFh = open("correlators.txt", 'w')
for c in correlators:
  for t in c:
    correlatorsFh.write("%d  "%(t))
  correlatorsFh.write("\n")

correlatorsFh.close();

ek, orb = np.linalg.eigh(fock)
diag = np.zeros((nsites, nsites))
for i in range(nsites/2):
  diag[i, i] = 1.
pairMat = np.mat(orb) * np.mat(diag) * np.mat(np.transpose(orb))

pairFh = open("pairMat.txt", 'w')
for i in range(nsites):
  for j in range(nsites):
    pairFh.write('%16.10e '%(pairMat[i,j]))
  pairFh.write('\n')

pairFh.close()
