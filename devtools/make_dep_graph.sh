#!/bin/bash

# From:
# http://www.stablecoder.ca/2019/03/15/sorting-you-dependency-graph.html

cmake .. --graphviz=test.dot
dot -Tpng test.dot -o out.png