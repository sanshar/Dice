import numpy as np
import pandas as pd

#
# Parse Benchmark Data
#

benchmark_file = "benchmark_raw.out"

with open(benchmark_file, "r") as f:
    lines = f.readlines()

# Convert data to floats and collect in data dictionary

dd = {"Max. Mem.": [], "Wall Time": []}  # data dictionary
mem_tok = "Maximum resident set size"  # memory token
wt_tok = "Elapsed (wall clock) time"  # Wall time token

for line in lines:
    if mem_tok in line:
        # print(line.split())
        dd["Max. Mem."].append(float(line.split()[-1]))

    if wt_tok in line:
        # print(line.split()[-1].split(":")[-1])
        seconds = float(float(line.split()[-1].split(":")[-2])) * 60
        seconds += float(line.split()[-1].split(":")[-1])
        dd["Wall Time"].append(seconds)

df = pd.DataFrame(dd)
print(df.to_markdown(showindex=False))

#
# Statistics
#
mem_mean, mem_std = (np.mean(df["Max. Mem."]), np.std(df["Max. Mem."]))
wt_mean, wt_std = (np.mean(df["Wall Time"]), np.std(df["Wall Time"]))
print(f"Average Max. Mem. {mem_mean:10.1e} \u00B1 {mem_std:8.1e} (kbytes)")
print(f"Average Wall Time {wt_mean:10.3f} \u00B1 {wt_std:8.2f} (s)")
