#!/bin/bash

find . -name "*.bkp" |xargs rm -f
find . -name "shci.e" |xargs rm -f 
find . -name "spatialRDM.*.txt"|xargs rm -f
find . -name "spatial1RDM.*.txt"|xargs rm -f
find . -name "spatial3RDM.*.txt"|xargs rm -f
# find . -name "spatial4RDM.*.txt"|xargs rm
find . -name "spin1RDM.*.txt"|xargs rm -f
