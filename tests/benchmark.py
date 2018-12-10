#!/usr/bin/env python
import subprocess
from subprocess import Popen
import os
import re
import pandas as pd

home_dir = os.getcwd()


def run_and_collect(home_dir, test_dir):
    """
    Runs /usr/bin/time on benchmark calculation and returns relevant info.
    """

    nproc = 12
    command = '/usr/bin/time --verbose mpirun -np %d ../../build/Dice' % nproc
    cmd_list = command.split()

    # Go home first and then go to relative test directory
    os.chdir(home_dir)  # In case the test fails
    os.chdir(test_dir)

    #
    process = subprocess.run(
        cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    time_output = process.stderr.decode("utf-8")
    lines = time_output.splitlines()

    # Get data
    data = {}

    for i in range(1, len(lines)):
        lsplit = lines[i].split()

        if lsplit[0] == "Elapsed": continue

        # Assemble dictionary key
        raw_key = lsplit[0]
        for tok in lsplit[1:-1]:
            raw_key += "_" + tok

        key = raw_key.strip(":")

        # Convert value to float
        try:
            val = float(lsplit[-1])
        except ValueError:
            val = float(lsplit[-1][:-1])

        data[key] = val

    # Reset directory if test succeeds
    os.chdir(home_dir)

    return data


#
# Main
#
test_dirs = [
    "o2_omp1_det",
    "o2_omp1_stoc",
    "c2_pt_rdm",
    "cr2_dinfh_trev_rdm",
    "mn_salen_quick",
]

data = {}
for test_dir in test_dirs:
    data[test_dir] = run_and_collect(home_dir, test_dir)

df = pd.DataFrame(data)
print(df)
df.to_csv("benchmark.csv")
