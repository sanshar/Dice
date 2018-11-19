"""
Small smoke tests using pytest.
"""

import os
import subprocess
import pytest
import dice_testing_utilities as dtu

mpi_call = "mpirun -np 2 ../../build/Dice"
main_test_dir = os.getcwd()


# This resets us to the main test directory BEFORE each test
@pytest.yield_fixture(autouse=True)
def dir_reset():
    os.chdir(main_test_dir)  # Go to main test directory
    yield  # Run test
    # dtu.cleanup()  # Clean up Dice files


# Tests
def test_o2_omp1_stoc():

    os.chdir("./o2_omp1_stoc")
    subprocess.call(mpi_call, shell=True)
    dtu.test_energy()
    dtu.compare_2rdm()


def test_o2_omp1_det():

    os.chdir("./o2_omp1_det")
    subprocess.call(mpi_call, shell=True)
    dtu.test_energy()


def test_o2_omp1_det_trev():

    os.chdir("./o2_omp1_det_trev")
    subprocess.call(mpi_call, shell=True)
    dtu.test_energy()


def test_o2_omp1_det_trev_direct():

    os.chdir("./o2_omp1_det_trev_direct")
    subprocess.call(mpi_call, shell=True)
    dtu.test_energy()


def test_o2_omp1_det_direct():

    os.chdir("./o2_omp1_det_direct")
    subprocess.call(mpi_call, shell=True)
    dtu.test_energy()


def test_restart():

    os.chdir("./restart")
    subprocess.call(
        "mpirun -np 2 ../../build/Dice input2.dat > output2.dat", shell=True)
    subprocess.call(
        "mpirun -np 2 ../../build/Dice input3.dat > output3.dat", shell=True)
    dtu.test_energy()


def test_restart_trev():

    os.chdir("./restart_trev")
    subprocess.call(
        "mpirun -np 2 ../../build/Dice input2.dat > output2.dat", shell=True)
    subprocess.call(
        "mpirun -np 2 ../../build/Dice input3.dat > output3.dat", shell=True)
    dtu.test_energy()


def test_fullrestart():

    os.chdir("./fullrestart")
    subprocess.call(
        "mpirun -np 2 ../../build/Dice input2.dat > output2.dat", shell=True)
    subprocess.call(
        "mpirun -np 2 ../../build/Dice input3.dat > output3.dat", shell=True)
    dtu.test_energy()
