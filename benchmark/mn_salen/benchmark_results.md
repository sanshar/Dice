# Benchmarking Dice with Mn(salen)

This file contains a record of the benchmarks using Mn(salen) as a test system. There are two benchmarks here [small](#performance-for-small-test) and [large](#performance-for-large-test). The small test has roughly 900,000 determinants and takes about a minute per Dice calculation, while the larger test has roughly 2.5 million determinants and takes about 2.5 minutes per Dice calculations. 


- [Benchmarking Dice with Mn(salen)](#benchmarking-dice-with-mnsalen)
  - [Guidlines:](#guidlines)
  - [Performance for Small Test:](#performance-for-small-test)
  - [Performance for Large Test:](#performance-for-large-test)
  - [Previous Performance for Small Test:](#previous-performance-for-small-test)
  - [Previous Performance for Large Test:](#previous-performance-for-large-test)

---
## Guidlines:

To run (or rerun) this benchmark calculation, edit the `run_all.sh` script with your desired settings (there are only three options)
1. `N_PROCS`, the number of MPI processes to use for test.
2. `N_REPEATS`, the number of repeats to do (for the statistics)
3. `TEST_TYPE`, either "small" or "large".

and run `run_all.sh` in the command line.

Once you run the benchmark, the raw and analyzed outputs will be written to `benchmark_raw.out` and `benchmark.out` respectively.

If you are planning to update this file (`benchmark_results.md`), please use the following template:

```markdown
<!-- Info: -->
|   Date    | Commit  |   Node    | #Cores |  CPU  |  RAM   | DISK  |
| :-------: | :-----: | :-------: | :----: | :---: | :----: | :---: |
| 4/22/2020 | 8de8a0a | bnode0306 |   28   | 2.4Hz | 128 GB |  1TB  |

<!-- Results: -->
| Max. Mem. | Wall Time |
| --------: | --------: |
|    852140 |      54.9 |
.    .      .      .    .s
.    .      .      .    .s
.    .      .      .    .s

Average Max. Mem.    7.9e+05 ±  3.2e+04 (kbytes)
Average Wall Time     54.151 ±     0.32 (s)

```

---

## Performance for Small Test:

|   Date    | Commit  |   Node    | #Cores |  CPU  |  RAM   | DISK  |
| :-------: | :-----: | :-------: | :----: | :---: | :----: | :---: |
| 4/22/2020 | 8de8a0a | bnode0306 |   28   | 2.4Hz | 128 GB |  1TB  |


| Max. Mem. | Wall Time |
| --------: | --------: |
|    852140 |      54.9 |
|    792176 |     53.89 |
|    799336 |     54.35 |
|    798896 |     53.97 |
|    849400 |     54.03 |
|    749396 |     54.27 |
|    770564 |     54.45 |
|    772760 |     53.88 |
|    771016 |     53.88 |
|    774164 |     53.89 |

```
Average Max. Mem.    7.9e+05 ±  3.2e+04 (kbytes)
Average Wall Time     54.151 ±     0.32 (s)
```

---
## Performance for Large Test:
|   Date    | Commit  |   Node    | #Cores |  CPU  |  RAM   | DISK  |
| :-------: | :-----: | :-------: | :----: | :---: | :----: | :---: |
| 4/22/2020 | 8de8a0a | bnode0306 |   28   | 2.4Hz | 128 GB |  1TB  |

|   Max. Mem. | Wall Time |
| ----------: | --------: |
| 1.88657e+06 |    148.51 |
| 1.88736e+06 |    147.46 |
|  1.8871e+06 |     148.6 |
| 1.88819e+06 |    147.88 |
| 1.88702e+06 |    148.23 |
| 1.88734e+06 |    147.84 |
| 1.88742e+06 |    150.62 |
| 1.88824e+06 |    146.36 |
| 1.88653e+06 |     147.5 |
|  1.8875e+06 |    148.36 |

```
Average Max. Mem.    1.9e+06 ±  5.5e+02 (kbytes)
Average Wall Time    148.136 ±     1.04 (s)
```

---
## Previous Performance for Small Test:
|   Date    | Commit  |   Node    | #Cores |  CPU  |  RAM   | DISK  |
| :-------: | :-----: | :-------: | :----: | :---: | :----: | :---: |
| 4/22/2020 | f55ca0b | bnode0306 |   28   | 2.4Hz | 128 GB |  1TB  |

| Max. Mem. | Wall Time |
| --------: | --------: |
|    800836 |     54.23 |
|    800148 |     53.81 |
|    775816 |     53.63 |
|    795660 |      54.2 |
|    750592 |     54.27 |
|    800964 |     53.55 |
|    849140 |     53.44 |
|    789220 |     53.55 |
|    750412 |     54.01 |
|    750656 |     53.95 |

```
Average Max. Mem.    7.9e+05 ±  2.9e+04 (kbytes)
Average Wall Time     53.864 ±     0.30 (s)
```

---
## Previous Performance for Large Test:
|   Date    | Commit  |   Node    | #Cores |  CPU  |  RAM   | DISK  |
| :-------: | :-----: | :-------: | :----: | :---: | :----: | :---: |
| 4/22/2020 | f55ca0b | bnode0306 |   28   | 2.4Hz | 128 GB |  1TB  |

|   Max. Mem. | Wall Time |
| ----------: | --------: |
|  1.8884e+06 |    147.82 |
| 1.88848e+06 |    149.28 |
| 1.88612e+06 |    148.47 |
| 1.88797e+06 |    147.74 |
| 1.88806e+06 |     148.6 |
| 1.88653e+06 |    148.04 |
| 1.88789e+06 |    149.32 |
|  1.8874e+06 |    147.35 |
| 1.88893e+06 |    147.48 |
|  1.8874e+06 |    149.43 |

```
Average Max. Mem.    1.9e+06 ±  8.3e+02 (kbytes)
Average Wall Time    148.353 ±     0.75 (s)
```