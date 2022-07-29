import numpy

xmax, ymax = 1,6
nsites = xmax*ymax
U = 4.
writeLocal = True
corrLen = 1
sqrt2 = 2**0.5

data = []
for i in range(xmax):
    for j in range(ymax):
        data.append([i,j])

correlators=[]

print  "&FCI NORB=%d ,NELEC=%d ,MS2=0,"%(nsites, nsites)
print  "ORBSYM=",
for i in range(nsites):
    print "%d,"%(1),
print
print  "ISYM=1,"
print "&END"

int2 = numpy.zeros((nsites, nsites, nsites, nsites))
int1 = numpy.zeros((nsites, nsites))
fock = numpy.zeros(shape=(nsites, nsites))
fockUp = numpy.zeros(shape=(nsites, nsites))
fockDn = numpy.zeros(shape=(nsites, nsites))

for i in range(len(data)):
    if (writeLocal):
        print U, i+1, i+1, i+1, i+1
    int2[i,i,i,i] = U

for i in range(len(data)):
    for j in range(i+1,len(data)):
        d1, d2 = data[i], data[j]
        if ( (abs(d1[0] - d2[0]) == 1 or abs(d1[0] - d2[0]) == xmax-1) and d1[1]== d2[1]) :
            if (writeLocal):
                print "-1.", i+1, j+1, 0, 0
            int1[i,j] = int1[j,i] = -1
            fock[i,j] = fock[j,i] = -1
            fockUp[i,j] = fockUp[j,i] = -1
            fockDn[i,j] = fockDn[j,i] = -1
        elif ( (abs(d1[1] - d2[1]) == 1 or abs(d1[1] - d2[1]) == ymax-1) and d1[0]== d2[0]) :
            if (writeLocal):
                print "-1.", i+1, j+1, 0, 0
            int1[i,j] = int1[j,i] = -1
            fock[i,j] = fock[j,i] = -1
            fockUp[i,j] = fockUp[j,i] = -1
            fockDn[i,j] = fockDn[j,i] = -1

m = 0.4
n = nsites/2

for i in range(nsites):
    fockUp[i, i] = U*(n-m*(-1)**(i+1))
    fockDn[i, i] = U*(n+m*(-1)**(i+1))

print fockUp
print fockDn

ekUp, orbUp = numpy.linalg.eigh(fockUp)
ekDn, orbDn = numpy.linalg.eigh(fockDn)

print ekUp
print orbUp
print ekDn
print orbDn

fileHF = open("uhf.txt", 'w')
for i in range(nsites):
    for j in range(nsites):
        fileHF.write('%16.10e '%(orbUp[i,j]))
    for j in range(nsites):
        fileHF.write('%16.10e '%(orbDn[i,j]))
    fileHF.write('\n')

fileHF = open("ghf2.txt", 'w')
for i in range(nsites):
    for j in range(nsites/2):
        fileHF.write('%16.10e '%(orbUp[i,j]/sqrt2))
    for j in range(nsites/2):
        fileHF.write('%16.10e '%(orbDn[i,j]/sqrt2))
    for j in range(nsites/2):
        fileHF.write('%16.10e '%(orbUp[i,j+nsites/2]/sqrt2))
    for j in range(nsites/2):
        fileHF.write('%16.10e '%(orbDn[i,j+nsites/2]/sqrt2))
    fileHF.write('\n')
for i in range(nsites):
    for j in range(nsites/2):
        fileHF.write('%16.10e '%(orbUp[i,j]/sqrt2))
    for j in range(nsites/2):
        fileHF.write('%16.10e '%(-orbDn[i,j]/sqrt2))
    for j in range(nsites/2):
        fileHF.write('%16.10e '%(orbUp[i,j+nsites/2]/sqrt2))
    for j in range(nsites/2):
        fileHF.write('%16.10e '%(-orbDn[i,j+nsites/2]/sqrt2))
    fileHF.write('\n')

for i in range(nsites):
    c = [i]
    for j in range(nsites):
        d1, d2 = data[i], data[j]
        if ( (abs(d1[0] - d2[0]) <= corrLen or abs(d1[0] - d2[0]) >= xmax-corrLen) and d1[1]== d2[1]) :
            c.append(j)
        elif ( (abs(d1[1] - d2[1]) <= corrLen or abs(d1[1] - d2[1]) >= xmax-corrLen) and d1[0]== d2[0]) :
            c.append(j)
    c = list(set(c))
    correlators.append(c)

#f = open("correlators.txt", 'w')
#for c in correlators:
#    for t in c:
#        f.write("%d  "%(t))
#    f.write("\n")

[d,v] = numpy.linalg.eigh(int1)
fileHF = open("rhf.txt", 'w')
for i in range(nsites):
    for j in range(nsites):
        fileHF.write('%16.10e '%(v[i,j]))
    fileHF.write('\n')

fileHF = open("ghf1.txt", 'w')
for i in range(nsites):
    for j in range(nsites/2):
        fileHF.write('%16.10e '%(v[i,j]))
    for j in range(nsites/2):
        fileHF.write('%16.10e '%(0.))
    for j in range(nsites/2):
        fileHF.write('%16.10e '%(v[i,j+nsites/2]))
    for j in range(nsites/2):
        fileHF.write('%16.10e '%(0.))
    fileHF.write('\n')
for i in range(nsites):
    for j in range(nsites/2):
        fileHF.write('%16.10e '%(0.))
    for j in range(nsites/2):
        fileHF.write('%16.10e '%(v[i,j]))
    for j in range(nsites/2):
        fileHF.write('%16.10e '%(0.))
    for j in range(nsites/2):
        fileHF.write('%16.10e '%(v[i,j+nsites/2]))
    fileHF.write('\n')

if (not writeLocal) :

    newint = numpy.einsum('ij,iklm->jklm', v  , int2)
    int2   = numpy.einsum('ij,kilm->kjlm', v  , newint)
    newint = numpy.einsum('ij,klim->kljm', v  , int2)
    int2   = numpy.einsum('ij,klmi->klmj', v  , newint)

    for i in range(len(data)):
        for j in range(i, len(data)):
            for k in range(len(data)):
                for l in range(k, len(data)):
                    print int2[i,j,k,l], i+1, j+1, k+1, l+1

    for i in range(len(data)):
        print d[i], i+1, i+1, 0, 0

