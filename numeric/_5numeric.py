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
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

############################################################################

def replaceNans(a, b, aTarget):
    a = a.copy()
    b = b.copy()
    total = a.shape[0]

    for key in a.keys():
        # print ("Replacing NaNs in %s" % (key))
        # Calculate summary statistics
        vals = np.unique(a[key])
        if np.sum(pd.isnull(vals))==0:
            continue
        vals = vals[pd.isnull(vals)==False] # Remove NaNs

        if vals.shape[0]==1:
            print("Column %s has only one unique value (%f) and should be dropped." % (key, vals[0]))
            continue
        if vals.shape[0]>100:
            nanRows = pd.isnull(a[key])
            # replace_val = int(np.mean(a.loc[nanRows==False, key]))
            a.loc[nanRows, key] = -1
            b.loc[pd.isnull(b[key]), key] = -1
            continue

        # Calculate target percentages of each unique value
        tCents = {}
        for val in vals:
            rows = a[key]==val
            if np.sum(rows)>total*0.01:
              target_percent = np.mean(aTarget[rows])
              tCents[val] = target_percent

        if not tCents:
            # print("%s: No value is more than 1%% of data (%d unique values)" % (key, vals.shape[0]));
            nanRows = pd.isnull(a[key])
            # replace_val = int(np.mean(a.loc[nanRows==False, key]))
            a.loc[nanRows, key] = -1
            b.loc[pd.isnull(b[key]), key] = -1
            continue

        # Target percentage of NaNs
        nanRows = pd.isnull(a[key])
        nanCent = np.mean(aTarget[nanRows])

        # Calculate difference between each values target percentage and NaNs target percentage
        for val in tCents:
            tCents[val] = abs(tCents[val] - nanCent)

        replace_val = min(tCents, key=tCents.get)

        # print("Replacing NaNs in %s with %f" % (key, replace_val))

        a.loc[nanRows, key] = replace_val
        b.loc[pd.isnull(b[key]), key] = replace_val

    return a, b

############################################################################

def do(a, b, aTarget):
    # Transform Columns
    print("Replacing NaNs...")
    a, b = replaceNans(a, b, aTarget)

    # # Replace with -1
    # print("Replacing NaNs with -1...")
    # a[pd.isnull(a)] = -1
    # b[pd.isnull(b)] = -1

    return a, b

def run(loadname="3corrS", savename="numeric"): # 2corrP
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


