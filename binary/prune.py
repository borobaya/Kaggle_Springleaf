#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle as pk
from time import time
from os.path import isfile
import gc
import sys

if "../lib" not in sys.path:
    sys.path.append("../lib")
if "lib" not in sys.path:
    sys.path.append("lib")
import params
import helper

def getPredPower(col, target):
    posPred = np.mean(target[col==True])
    negPred = np.mean(target[col==False])
    power = abs(posPred-negPred)
    return power

def getPredPowers(frame, target):
    powers = [getPredPower(frame[:,i], target) for i in xrange(frame.shape[1])]
    return np.array(powers)

def run():
    print("Loading datasets...")
    with open("../cache/train_binary.pkl", "r") as f, open("../cache/test_binary.pkl", "r") as g, open("../data/trainTarget.pkl", "r") as h:
        train = pk.load(f).toarray()
        test = pk.load(g).toarray()
        aTarget = pk.load(h)

    trainCount = np.sum(train, axis=0)
    trainCount2 = train.shape[0]-trainCount
    trainCount = np.min([trainCount, trainCount2], axis=0)

    testCount = np.sum(test, axis=0)
    testCount2 = test.shape[0]-testCount
    testCount = np.min([testCount, testCount2], axis=0)

    trainCount = np.min([trainCount, testCount], axis=0)

    print("Calculating predictive power of each column...")
    powers = getPredPowers(train, aTarget)

    thres = 1500
    keep = (trainCount>thres)
    keep |= (powers>0.05) & (trainCount>500)
    print("Discarding %d of %d columns..." % (np.sum(keep==False), keep.size))
    train = train[:,keep]
    test = test[:,keep]

    # Discard linear combinations
    print("Finding duplicate columns...")
    a = pd.DataFrame(train)
    colCorrs = helper.corrMat(a)
    groups = helper.getGroups( (colCorrs>=1.0) , a, a.keys().values)
    keep = np.ones([a.shape[1],], dtype=np.bool)
    for group in groups:
        for key in list(group)[1:]:
            keep[key] = False
    print("Discarding %d duplicate columns..." % (np.sum(keep==False)))
    train = train[:,keep]
    test = test[:,keep]

    print("Column count: %d" % (train.shape[1]))

    print("Saving...")
    with open("../cache/train_binary_pruned.pkl", "w") as f, open("../cache/test_binary_pruned.pkl", "w") as g:
        pk.dump(train, f, protocol=2)
        pk.dump(test, g, protocol=2)

run()