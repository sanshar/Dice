import numpy as np

nsites = 20
sqrt2 = 2**0.5

orbUp = np.zeros(shape=(nsites, nsites))
orbDn = np.zeros(shape=(nsites, nsites))

fh = open('uhf.txt', 'r')
i = 0
for line in fh:
  for j in range(nsites):
    orbUp[i, j] = line.split()[j]
  for j in range(nsites):
    orbDn[i, j] = line.split()[j + nsites]
  i = i + 1

fileHF = open("ghf.txt", 'w')
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
