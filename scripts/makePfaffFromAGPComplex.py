import numpy as np

norb = 18

matr = np.full((norb, norb), 0.)
mati = np.full((norb, norb), 0.)

row = 0
fileh = open("pairMatAGP.txt", 'r')
for line in fileh:
  col = 0
  for coeff in line.split():
    m = coeff.strip()[1:-1]
    matr[row, col], mati[row, col] = [float(x) for x in m.split(',')]
    col = col + 1
  row = row + 1
fileh.close()

rmat =  (np.random.rand(norb,norb) + 1j * np.random.rand(norb,norb))/50
rmat = rmat - rmat.T

fileHF = open("pairMat.txt", 'w')
for i in range(norb):
    for j in range(norb):
        fileHF.write('%16.10e '%(rmat[i,j].real))
    for j in range(norb):
        fileHF.write('%16.10e '%(matr[i,j]))
    fileHF.write('\n')
for i in range(norb):
    for j in range(norb):
        fileHF.write('%16.10e '%(-matr[i,j]))
    for j in range(norb):
        fileHF.write('%16.10e '%(rmat[i,j].real))
    fileHF.write('\n')

fileHF.close()

fileHF = open("pairMati.txt", 'w')
for i in range(norb):
    for j in range(norb):
        fileHF.write('%16.10e '%(rmat[i,j].imag))
    for j in range(norb):
        fileHF.write('%16.10e '%(mati[i,j]))
    fileHF.write('\n')
for i in range(norb):
    for j in range(norb):
        fileHF.write('%16.10e '%(-mati[i,j]))
    for j in range(norb):
        fileHF.write('%16.10e '%(rmat[i,j].imag))
    fileHF.write('\n')

fileHF.close()


