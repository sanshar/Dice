'''Script to analyze pt2_energies files, producing an output which
   contains the averages of key quantities from each file.'''

import pandas as pd
import sys
import os

def extract_data():

    column_names = [ 'energy', 'bias', 'time' ]

    energies = []
    bias = []
    time = []

    # Count the number of files to read
    n = 0
    for data_file in os.listdir('.'):
        if data_file.startswith('pt2_energies') and 'avg' not in data_file:
            n += 1

    for i in range(n):
        data_file = 'pt2_energies_' + str(i) + '.dat'
        f = open(data_file)

        # Read in the data table from data_file
        data = pd.io.parsers.read_table(data_file, sep='\s+',
                   engine='python', names=column_names,
                   comment='#', usecols=[1, 4, 8])

        f.close()

        # Store the averages of the energy and bias, and the cumulative time
        energies.append(data['energy'].mean())
        bias.append(data['bias'].mean())
        time.append(data['time'].sum())

    return energies, bias, time

if __name__ == '__main__':
    energies, bias, time = extract_data()

    # Printing the final values to standard output
    n = 0
    sys.stdout.write('# 1. proc label     2. energy            3. bias correction   4. time\n')
    for e, b, t in zip(energies, bias, time):
        sys.stdout.write('%15s' % n)
        sys.stdout.write('    %.12e' % e)
        sys.stdout.write('   %.12e' % b)
        sys.stdout.write('   %.4e' % t)
        sys.stdout.write('\n')
        n += 1
