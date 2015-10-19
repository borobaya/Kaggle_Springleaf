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

def do(a, b, aTarget):
    # Save all key names at correct indexes
    keys = a.keys().values

    # Stats calculations
    if False:
        print("Calculating correlations between columns....")
        colCorrs = helper.corrMat(a)
        print("Saving...")
        with open("../stats/colCorrsP.pkl", "w") as f:
            pk.dump(colCorrs, f, protocol=2)
    else:
        with open("../stats/colCorrsP.pkl", "r") as f:
            colCorrs = pk.load(f)

    print("Column count: %d" % (a.columns.size))

    for thres in np.arange(1, 0.94, -0.01):
        print("Columns correlated >=%f..." % (thres))
        groups = helper.getGroups( (colCorrs>=thres) , a, keys)
        a, b = helper.regressWithinGroups(groups, thres, a, b)

    return a, b


def run(loadname="1prune", savename="2corrP"):
    # Load
    print("Loading datasets...")
    with open("../cache/train_"+loadname+".pkl", "r") as f, open("../cache/test_"+loadname+".pkl", "r") as g, \
        open("../data/trainTarget.pkl", "r") as h:
        a = pk.load(f)
        b = pk.load(g)
        aTarget = pk.load(h)

    # Do
    a, b = do(a, b, aTarget)

    # Save
    print("Saving...")
    with open("../cache/train_"+savename+".pkl", "w") as f, open("../cache/test_"+savename+".pkl", "w") as g:
        pk.dump(a, f, protocol=2)
        pk.dump(b, g, protocol=2)

    os.system('say "Finished"')

if __name__ == "__main__":
    run()


