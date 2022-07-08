#!/bin/bash

rm *.bkp BestDeterminant.txt vmc.out gfmc.out pt2_energies* -f >/dev/null 2>&1
find . -name *.bkp |xargs rm >/dev/null 2>&1
find . -name BestDeterminant.txt|xargs rm >/dev/null 2>&1
find . -name amsgrad.bkp |xargs rm >/dev/null 2>&1
find . -name cpsslaterwave.bkp|xargs rm >/dev/null 2>&1
find . -name vmc.out | xargs rm >/dev/null 2>&1
find . -name gfmc.out | xargs rm >/dev/null 2>&1
find . -name ci.out | xargs rm >/dev/null 2>&1
find . -name nevpt.out | xargs rm >/dev/null 2>&1
find . -name nevpt_print.out | xargs rm >/dev/null 2>&1
find . -name nevpt_read.out | xargs rm >/dev/null 2>&1
find . -name pt2_energies* | xargs rm >/dev/null 2>&1
find . -name stoch_samples_* | xargs rm >/dev/null 2>&1
find . -name fciqmc.out | xargs rm >/dev/null 2>&1
find . -name samples.dat | xargs rm >/dev/null 2>&1
find . -name afqmc.out | xargs rm >/dev/null 2>&1
find . -name afqmc.dat | xargs rm >/dev/null 2>&1
find . -name blocking.tmp | xargs rm >/dev/null 2>&1
find . -name blocking.out | xargs rm >/dev/null 2>&1
