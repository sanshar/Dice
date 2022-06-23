import sys
import struct
import pandas as pd

tol = 1e-6
energy_file = None
if len(sys.argv) > 3:
    raise ValueError(
        "Wrong number of CLI.\nUse python compare_energies.py pyscf_energies.csv 1e-6"
    )
elif len(sys.argv) == 3:
    energy_file = sys.argv[1]
    tol = float(sys.argv[2])

df = pd.read_csv(energy_file)

for i, irrep in enumerate(df["irreps"]):

    file1 = open(f"_energies/{irrep}_energy.e", "rb")
    calc_e = struct.unpack("d", file1.read(8))[0]

    abs_err = abs(calc_e - df["energy"][i])
    if abs_err < tol:
        print(f"\tEnergy test for irrep = {irrep:5s}\033[92m Passed \033[00m")

    else:
        print(
            f"\tEnergy test for irrep = {irrep:5s} (error={abs_err:.2e})\033[91m Failed \033[00m"
        )

