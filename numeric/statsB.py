#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle as pk
from time import time
from pprint import pprint
from os.path import isfile
import re
import gc
import sys
from scipy.stats import ks_2samp, pearsonr, spearmanr

if "../lib" not in sys.path:
    sys.path.append("../lib")
import helper

def run(loadname="3corrS", savename="4similar"):
    print("Loading datasets...")
    with open("../cache/train_"+loadname+".pkl", "r") as f, open("../cache/test_"+loadname+".pkl", "r") as g, \
        open("../data/trainTarget.pkl", "r") as h:
        a = pk.load(f)
        b = pk.load(g)
        aTarget = pk.load(h)

    keys = a.keys()
    

    # print("Using negate function....")
    # negateCorrs, negateDists, negatePreds = helper.combine(a, b, aTarget, lambda x,y: x-y)
    # print("Saving...")
    # with open("../stats/negateCorrs.pkl", "w") as f, open("../stats/negateDists.pkl", "w") as g, open("../stats/negatePreds.pkl", "w") as h:
    #     pk.dump(negateCorrs, f, protocol=2)
    #     pk.dump(negateDists, g, protocol=2)
    #     pk.dump(negatePreds, h, protocol=2)
    # gc.collect()

    # with open("../stats/negateCorrs.pkl", "r") as f, open("../stats/negateDists.pkl", "r") as g, \
    #     open("../stats/negatePreds.pkl", "r") as h:
    #     negateCorrs = pk.load(f)
    #     negateDists = pk.load(g)
    #     negatePreds = pk.load(h)


    # To group columns to help identify new features to construct
    # Max Number of digits must be equal
    # Range, Mean
    aMin = a.min()
    aMax = a.max()
    aMean = a.mean()
    aRange = aMax - aMin
    aDigits = aMax.astype(np.int).apply(lambda x: len(str(x)))
    groupMat = np.zeros([len(keys), len(keys)], dtype=np.bool)
    for i in xrange(len(keys)):
        for j in xrange(len(keys)):
            if i>=j:
                continue
            if aDigits[i]!=aDigits[j] or aDigits[i]==1:
                continue
            if aRange[i]<10 or aRange[j]<10:
                continue
            meanRatio = float(aDigits[i])/float(aDigits[j])
            meanRatio = meanRatio if meanRatio<1.0 else 1.0/meanRatio
            if meanRatio<1.0-1e-10:
                continue
            rangeRatio = float(aRange[i])/float(aRange[j])
            rangeRatio = rangeRatio if rangeRatio<1.0 else 1.0/rangeRatio
            if rangeRatio<1.0-1e-10:
                continue
            groupMat[i,j] = 1
    
    groups = helper.getGroups(groupMat.astype(np.bool), a, keys)
    print(groups)


    from matplotlib import pyplot as plt
    i = 0
    g = [list(i) for i in groups]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a[g[i][0]], a[g[i][1]], a[g[i][2]])
    plt.show()

    plt.scatter(a[g[i][0]], a[g[i][1]], c=aTarget, s=1, edgecolors='none')
    plt.show()


    # TODO
    # Determine what each column represents
    # e.g. age, salary, house price


    # TODO
    # Number of equal values between columns
    # plt.hist(aCol/bCol)


run()



