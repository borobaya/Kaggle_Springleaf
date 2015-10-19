#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle as pk
from time import time
from pprint import pprint
from os.path import isfile
import os
import sys
import re
import gc

if "../lib" not in sys.path:
    sys.path.append("../lib")
if "lib" not in sys.path:
    sys.path.append("lib")
import params
import helper

def do(a, b):
    # Drop Columns
    to_drop = ['ID'] + params.emptyColNames + params.binaryColNames + \
        params.factorTextColNames + params.locationColNames + params.textColNames
    to_drop = list(set(to_drop))
    print("Dropping %d columns..." % (len(to_drop)))
    a.drop(to_drop+['target'], axis=1, inplace=True)
    b.drop(to_drop, axis=1, inplace=True)

    # Transform Columns
    print("Transforming date columns...")
    a, b = helper.recodeDateText(a, b, params.dateColNames)
    a, b = helper.recodeDateOther(a, b, params.dateOtherColNames)

    print("Finding NaNs...")
    a, b = helper.findNans(a, b)

    return a, b

def run(savename="0nan"):
    # Load
    print("Loading datasets...")
    a = pd.read_csv('../data/train.csv')
    b = pd.read_csv('../data/test.csv') # 167,177

    # Do
    a, b = do(a, b)

    # Save
    print("Saving...")
    with open("../cache/train_"+savename+".pkl", "w") as f, open("../cache/test_"+savename+".pkl", "w") as g:
        pk.dump(a, f, protocol=2)
        pk.dump(b, g, protocol=2)

    os.system('say "Finished"')

if __name__ == "__main__":
    run()

