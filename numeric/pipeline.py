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

import _0nan
import _1prune
import _2corrP
import _3corrS
import _5numeric

def run(skip0=False):
    global a, b
    # Load
    print("Loading datasets...")
    with open("../data/trainTarget.pkl", "r") as f:
        aTarget = pk.load(f)

    if not skip0:
        a = pd.read_csv('../data/train.csv')
        b = pd.read_csv('../data/test.csv')
        a, b = _0nan.do(a, b)
        savename = "0nan"
        with open("../cache/train_"+savename+".pkl", "w") as f, open("../cache/test_"+savename+".pkl", "w") as g:
            pk.dump(a, f, protocol=2)
            pk.dump(b, g, protocol=2)
    else:
        loadname = "0nan"
        with open("../cache/train_"+loadname+".pkl", "r") as f, open("../cache/test_"+loadname+".pkl", "r") as g:
            a = pk.load(f)
            b = pk.load(g)

    # Do
    a, b = _1prune.do(a, b, aTarget)
    a, b = _2corrP.do(a, b, aTarget)
    # a, b = _3corrS.do(a, b, aTarget)
    a, b = _5numeric.do(a, b, aTarget)

    # Save
    print("Saving...")
    savename = "numeric"
    with open("../cache/train_"+savename+".pkl", "w") as f, open("../cache/test_"+savename+".pkl", "w") as g:
        pk.dump(a, f, protocol=2)
        pk.dump(b, g, protocol=2)

    os.system('say "Finished"')

# if __name__ == "__main__":
run(True)

