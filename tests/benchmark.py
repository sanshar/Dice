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

    command = '/usr/bin/time -l ../../build/Dice'
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

    lsplit = lines[0].split()
    data[lsplit[1]] = float(lsplit[0])
    data[lsplit[3]] = float(lsplit[2])
    data[lsplit[5]] = float(lsplit[4])

    for i in range(1, len(lines)):
        lsplit = lines[i].split()
        key = lsplit[1]
        for tok in lsplit[2:]:
            key += "_" + tok
        val = float(lsplit[0])
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
