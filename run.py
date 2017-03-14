import os
import sys
import re
import numpy as np
import pandas as pd
import sklearn

# Reading in column headings.
def read_facts(filename):
    f = open(filename, 'r')
    labels = f.readline()
    print(f)


def main(argv):
    read_facts(argv[1])
