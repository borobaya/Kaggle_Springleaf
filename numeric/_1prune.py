# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle as pk
from time import time
from pprint import pprint
from os.path import isfile
import sys
import os
import re
import gc

if "../lib" not in sys.path:
    sys.path.append("../lib")
if "lib" not in sys.path:
    sys.path.append("lib")
import params
import helper

def do(a, b, aTarget):

    if False: # If True, re-computes stats values
        # Proportion of null values
        print("Checking proportion of null values...")
        percent_nil = pd.isnull(a).mean()
        with open("../stats/percent_nil.pkl", "w") as f:
            pk.dump(percent_nil.values, f, protocol=2)

        # Proportion of most common value
        print("Proportion of most common values...")
        percent_mcv = helper.getProportionOfMCV(a)
        with open("../stats/percent_mcv.pkl", "w") as f:
            pk.dump(percent_mcv, f, protocol=2)

        # Infer feature importances
        print("Checking correlations and distributions on untransformed data...")
        corrs, dists, preds = helper.getStatsUsingFunc(a, b, aTarget, lambda x: x)
        print("Saving...")
        with open("../stats/corrs.pkl", "w") as f, open("../stats/dists.pkl", "w") as g, open("../stats/preds.pkl", "w") as h:
            pk.dump(corrs, f, protocol=2)
            pk.dump(dists, g, protocol=2)
            pk.dump(preds, h, protocol=2)

        # Percentage of overlap in the unique values between training and test
        print("Calculating the percentage of overlap in the unique values between train and test")
        unique_overlaps = helper.getUniqueOverlaps(a, b)
        print("Saving...")
        with open("../stats/unique_overlaps.pkl", "w") as f:
            pk.dump(unique_overlaps, f, protocol=2)

        # # Correlation matrix
        # print("Calculating correlations between columns....")
        # colCorrs = helper.corrMat(a)
        # print("Saving...")
        # with open("../stats/colCorrs.pkl", "w") as f:
        #     pk.dump(colCorrs, f, protocol=2)
    else:
        print("Loading stats...")
        with open("../stats/percent_nil.pkl", "r") as f:
            percent_nil = pk.load(f)
        with open("../stats/percent_mcv.pkl", "r") as f:
            percent_mcv = pk.load(f)
        with open("../stats/corrs.pkl", "r") as f, open("../stats/dists.pkl", "r") as g, \
            open("../stats/preds.pkl", "r") as h:
            corrs = pk.load(f)
            dists = pk.load(g)
            preds = pk.load(h)
        with open("../stats/unique_overlaps.pkl", "r") as f:
            unique_overlaps = pk.load(f)
        # with open("../stats/colCorrs.pkl", "r") as f:
        #     colCorrs = pk.load(f)

    # Weak columns
    # print("Weak columns:")
    # keep = (preds>0.05) | (corrs>0.05)
    # keep &= dists>0.01
    # keep &= unique_overlaps>0.02
    # params.weakColNames = list(a.keys()[keep==False].values)
    # print repr(params.weakColNames)
    # print("%d weak columns" % (len(params.weakColNames)))

    # Almost binary columns
    print("Almost binary columns:")
    almost_binary = (percent_nil+percent_mcv>=0.95)
    almost_binary |= (percent_nil>0.8)
    almost_binary |= (percent_mcv>0.8)
    # almost_binary &= keep # Remove weak columns that will be deleted anyway
    params.almostBinaryColNames = list(a.keys()[almost_binary].values)
    print repr(params.almostBinaryColNames)
    print("%d almost binary columns" % (len(params.almostBinaryColNames)))

    # Recode almost binary columns
    print("Recoding almost binary columns...")
    aAlmostBinary, bAlmostBinary = helper.recodeAlmostBinary(a, b, aTarget, params.almostBinaryColNames)
    a[params.almostBinaryColNames] = aAlmostBinary
    b[params.almostBinaryColNames] = bAlmostBinary

    # Drop Columns
    to_drop = []
    # to_drop += params.weakColNames
    # to_drop += params.almostBinaryColNames
    # to_drop += params.na56 + params.na89 + params.na918
    to_drop += params.dateColNames + \
              [i+'M' for i in params.dateColNames] + \
              [i+'Y' for i in params.dateColNames]
    to_drop += params.dateOtherColNames + \
              [i+'Y' for i in params.dateOtherColNames]
    to_drop = list(set(to_drop))
    to_drop = [i for i in to_drop if i in a.keys()]
    print("Dropping %d columns..." % (len(to_drop)))
    a.drop(to_drop, axis=1, inplace=True)
    b.drop(to_drop, axis=1, inplace=True)

    return a, b

def run(loadname="0nan", savename="1prune"):
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


