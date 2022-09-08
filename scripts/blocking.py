import sys
import numpy as np
import csv
import pandas as pd

np.set_printoptions(precision=5, linewidth=1000, suppress=True)
fname = str(sys.argv[1])
nEql = int(sys.argv[2])
print(f'reading samples from {fname}, ignoring first {nEql}')
cols = list(range(2))
df = pd.read_csv(fname, delim_whitespace=True, usecols=cols, header=None)
samples = df.to_numpy()
nSamples = samples.shape[0] - nEql
weights = samples[nEql:,0]
energies = samples[nEql:,1]
weightedEnergies = np.multiply(weights, energies)
print(f'mean: {weightedEnergies.sum() / weights.sum()}')
print('blocked statistics:')
blockSizes = np.array([ 1, 2, 5, 10, 20, 50, 70, 100, 200, 500 ])
print('block size    # of blocks        mean                error')
for i in blockSizes[blockSizes < nSamples/2.]:
    nBlocks = nSamples//i
    blockedWeights = np.zeros(nBlocks)
    blockedEnergies = np.zeros(nBlocks)
    for j in range(nBlocks):
        blockedWeights[j] = weights[j*i:(j+1)*i].sum()
        blockedEnergies[j] = weightedEnergies[j*i:(j+1)*i].sum() / blockedWeights[j]
    v1 = blockedWeights.sum()
    v2 = (blockedWeights**2).sum()
    mean = np.multiply(blockedWeights, blockedEnergies).sum() / v1
    error = (np.multiply(blockedWeights, (blockedEnergies - mean)**2).sum() / (v1 - v2 / v1) / (nBlocks - 1))**0.5
    print(f'  {i:4d}           {nBlocks:4d}       {mean:.8e}       {error:.6e}')
