"""
Small smoke tests using pytest.
"""

import os
import subprocess
import pytest
import dice_testing_utilities as dtu

dice_exe = "mpirun -np 28 ../../build/Dice"
main_test_dir = os.getcwd()


# This resets us to the main test directory BEFORE each test and cleans up
# after each test
@pytest.yield_fixture(autouse=True)
def dir_reset():
    os.chdir(main_test_dir)  # Go to main test directory
    yield  # Run test
    dtu.cleanup()  # Clean up Dice files


# Smaller Tests
def test_o2_stoc():

    os.chdir("./o2_stoc")
    subprocess.call(dice_exe, shell=True)
    dtu.test_energy()
    dtu.compare_2rdm()


def test_o2_det():

    os.chdir("./o2_det")
    subprocess.call(dice_exe, shell=True)
    dtu.test_energy()


def test_o2_det_trev():

    os.chdir("./o2_det_trev")
    subprocess.call(dice_exe, shell=True)
    dtu.test_energy()


def test_o2_det_trev_direct():

    os.chdir("./o2_det_trev_direct")
    subprocess.call(dice_exe, shell=True)
    dtu.test_energy()


def test_o2_det_direct():

    os.chdir("./o2_det_direct")
    subprocess.call(dice_exe, shell=True)
    dtu.test_energy()


# def test_c2_pt_rdm():
#     os.chdir("./c2_pt_rdm")
#     subprocess.call(dice_exe, shell=True)
#     dtu.test_energy()
#     dtu.compare_2rdm()


# Medium Tests
def test_restart():

    os.chdir("./restart")
    subprocess.call(dice_exe + " input2.dat > output2.dat", shell=True)
    subprocess.call(dice_exe + " input3.dat > output3.dat", shell=True)
    dtu.test_energy()


def test_restart_trev():

    os.chdir("./restart_trev")
    subprocess.call(dice_exe + " input2.dat > output2.dat", shell=True)
    subprocess.call(dice_exe + " input3.dat > output3.dat", shell=True)
    dtu.test_energy()


def test_fullrestart():

    os.chdir("./fullrestart")
    subprocess.call(dice_exe + " input2.dat > output2.dat", shell=True)
    subprocess.call(dice_exe + " input3.dat > output3.dat", shell=True)
    dtu.test_energy()


# The Big Tests
def test_cr2_dinfh_rdm():

    os.chdir("./cr2_dinfh_rdm")
    subprocess.call(dice_exe, shell=True)
    dtu.test_energy()
    dtu.compare_2rdm()


def test_cr2_dinfh_trev_rdm():

    os.chdir("./cr2_dinfh_trev_rdm")
    subprocess.call(dice_exe, shell=True)
    dtu.test_energy()
    dtu.compare_2rdm()
