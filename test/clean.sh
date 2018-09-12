#!/bin/bash

rm *.bkp BestDeterminant.txt vmc.out gfmc.out -f >/dev/null 2>&1
find . -name *.bkp |xargs rm >/dev/null 2>&1
find . -name BestDeterminant.txt|xargs rm >/dev/null 2>&1
find . -name amsgrad.bkp |xargs rm >/dev/null 2>&1
find . -name cpsslaterwave.bkp|xargs rm >/dev/null 2>&1
find . -name vmc.out | xargs rm >/dev/null 2>&1
find . -name gfmc.out | xargs rm >/dev/null 2>&1

