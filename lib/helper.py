#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle as pk
from time import time
from pprint import pprint
from os.path import isfile
import os
import re
import gc
import sys
from scipy.stats import linregress, rankdata, ks_2samp, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import params

# -------------------------------------------------------------------------------------------
# For Stats:

def getDist(colA, colB, limit=None):
    # Checks how similar two distributions are
    # Higher p value means that they are more similar
    colA = colA[pd.isnull(colA)==False]
    colB = colB[pd.isnull(colB)==False]
    if colA.size<=1500 or colB.size<=1500:
        return 0
    if limit is not None:
        indicesA = np.random.permutation(np.arange(colA.size))[:limit]
        colA = colA[indiceA]
        indicesB = np.random.permutation(np.arange(colB.size))[:limit]
        colB = colB[indicesB]
    D, p = ks_2samp(colA, colB)
    return p

def getCorr(colA, colB, limit=None, use_spearman=False):
    if hasattr(colA, 'values'):
        colA = colA.values
    if hasattr(colB, 'values'):
        colB = colB.values
    rows = (pd.isnull(colA)==False) & (pd.isnull(colB)==False)
    if np.sum(rows)<1500:
        return 0.0
    colA = colA[rows]
    colB = colB[rows]
    if limit is not None:
        indices = np.random.permutation(np.arange(colA.size))[:limit]
        colA = colA[indices]
        colB = colB[indices]
    corr, p = spearmanr(colA, colB) if use_spearman else pearsonr(colA, colB)
    corr = abs(corr)
    if pd.isnull(corr):
        corr = 0
        if np.unique(colA).size<=1:
            print("Correlation=0 because column A has only 1 unique value (%f)" % (colA[0]))
        elif np.unique(colB).size<=1:
            print("Correlation=0 because column B has only 1 unique value (%f)" % (colB[0]))
        else:
            print("Correlation=0 for unknown reason")
    # sys.stdout.write(str(corr)+" ")
    return corr

def getPredPower(colA, target, limit=None):
    # This function checks the difference in distributions
    # between positive and negative predicting values
    # Null hypothesis is that they are from the same distribution,
    # e.g. p<0.05 would mean that they are different distributions
    # *** Lower p is better ***
    rows = pd.isnull(colA)==False
    target = target[rows]
    colA = colA[rows]
    negValues = colA[target==False]
    posValues = colA[target==True]
    if negValues.size<=1500 or posValues.size<=1500:
        return 0
    if limit is not None:
        indices = np.random.permutation(np.arange(negValues.size))[:limit]
        negValues = negValues[indices]
        posValues = posValues[indices]
    D, p = ks_2samp(negValues, posValues)
    return 1.0-p

def corrMat(frame, use_spearman=False, verbose=False):
    if not verbose:
        # Faster:
        indices = np.random.permutation(np.arange(frame.shape[0]))[:90000]
        corrs = frame.ix[indices].corr(method=("spearman" if use_spearman else "pearson"), min_periods=1000)
        corrs = corrs.abs()
        corrs = corrs.values
        nils = pd.isnull(corrs)
        if np.sum(nils)>0:
            corrs[nils] = 0
            print("%d null correlation values found" % (np.sum(nils)))
        for i in xrange(corrs.shape[0]):
            corrs[i,:i+1] = 0
        return corrs

    # More verbose: (use for debugging)
    keys = frame.keys()
    corrs = np.zeros([keys.size,keys.size])
    n = 0
    nMax = keys.size*(keys.size-1)*0.5
    tic = time()
    for i in xrange(keys.size):
        keyI = keys[i]
        for j in xrange(keys.size):
            if i>=j:
                continue
            keyJ = keys[j]
            corr = getCorr(frame[keyI], frame[keyJ], limit=10000, use_spearman=use_spearman)
            corrs[i,j] = corr
            n += 1
            if n%500==0:
                percentage_completion = np.float(n)/nMax
                time_remaining = (1.0-percentage_completion)*(time()-tic)/percentage_completion
                sys.stdout.write(str(n)+" of "+str(int(nMax))+" done ("+str(100.0*percentage_completion)+"%)")
                sys.stdout.write("     %s remaining" % (hms_string(time_remaining)))
                sys.stdout.write("       \r")
                sys.stdout.flush()
    sys.stdout.write("\n")
    return corrs

def getStatsUsingFunc(frameA, frameB, target, f):
    keys = frameA.keys()
    corrs = np.zeros([keys.size,])
    dists = np.zeros([keys.size,])
    preds = np.zeros([keys.size,])
    for i in xrange(keys.size):
        key = keys[i]
        aTransformed = f(frameA[key])
        bTransformed = f(frameB[key])
        corr = getCorr(aTransformed, target)
        dist = getDist(aTransformed, bTransformed)
        pred = getPredPower(aTransformed, target)
        corrs[i] = corr
        dists[i] = dist
        preds[i] = pred
    return corrs, dists, preds

def combine(frameA, frameB, target, f):
    keys = frameA.keys()
    corrs = np.zeros([keys.size,keys.size])
    dists = np.zeros([keys.size,keys.size])
    preds = np.zeros([keys.size,keys.size])
    n = 0
    nMax = keys.size*(keys.size-1)*0.5
    tic = time()
    for i in xrange(keys.size):
        keyI = keys[i]
        for j in xrange(keys.size):
            if i>=j:
                continue
            keyJ = keys[j]
            aTransformed = f(frameA[keyI], frameA[keyJ])
            bTransformed = f(frameB[keyI], frameB[keyJ])
            corr = getCorr(aTransformed, target, limit=1000)
            dist = getDist(aTransformed, bTransformed, limit=1000)
            pred = getPredPower(aTransformed, target, limit=1000)
            corrs[i,j] = corr
            dists[i,j] = dist
            preds[i,j] = pred
            n += 1
            if n%500==0:
                percentage_completion = 100.0*np.float(n)/nMax
                time_remaining = (1.0-percentage_completion)*(time()-tic)/percentage_completion
                sys.stdout.write(str(n)+" of "+str(int(nMax))+" done ("+str(100.0*percentage_completion)+"%)")
                sys.stdout.write("     %s remaining" % (hms_string(time_remaining)))
                sys.stdout.write("       \r")
                sys.stdout.flush()
    sys.stdout.write("\n")
    return corrs, dists, preds

def getUniqueOverlaps(frameA, frameB):
    aUnique = [np.unique(frameA[i]) for i in frameA.keys()]
    bUnique = [np.unique(frameB[i]) for i in frameA.keys()]
    aUnique = [i[pd.isnull(i)==False] for i in aUnique]
    bUnique = [i[pd.isnull(i)==False] for i in bUnique]
    aUnique = [set(i) for i in aUnique]
    bUnique = [set(i) for i in bUnique]
    overlaps = [aUnique[i] & bUnique[i] for i in xrange(len(aUnique))]
    totals = [aUnique[i] | bUnique[i] for i in xrange(len(aUnique))]
    unique_overlaps = [float(len(overlaps[i]))/float(len(totals[i])) for i in xrange(len(aUnique))]
    unique_overlaps = np.array(unique_overlaps)
    return unique_overlaps

def getProportionOfMCV(frameA):
    # Get the proportion of the most common value in each column
    rowcount = frameA.shape[0]
    indices = ( np.random.sample(200)*rowcount ).astype(np.int) # For speedup
    keys = frameA.keys()

    proportions = np.zeros([keys.size,])
    for i in xrange(keys.size):
        key = keys[i] 
        vals = np.unique(frameA.loc[indices, key])
        vals = vals[pd.isnull(vals)==False]
        if vals.size==0:
            vals = np.unique(frameA[key])
            vals = vals[pd.isnull(vals)==False]
        for val in vals:
            proportion = np.mean(frameA[key]==val)
            if proportion>proportions[i]:
                proportions[i] = proportion

    return proportions

# -------------------------------------------------------------------------------------------
# Recoding columns

def findNans(a, b, recodeKeys=None):
    a = a.copy()
    b = b.copy()
    total = a.shape[0]

    if recodeKeys is None:
        print("findNans() input recodeKeys==None, setting it to all numeric keys")
        recodeKeys = a.keys()[a.dtypes!=object]

    for key in recodeKeys:
        if key == 'target':
            continue
        print ("Finding NaNs in %s" % (key))
        # Calculate summary statistics
        vals = np.unique(a[key])
        vals = vals[pd.isnull(vals)==False] # Remove NaN, which will cause problems later
        if vals.shape[0]<=1:
            continue

        mn = np.min(vals)
        mx = np.max(vals)

        # OUTLIER TYPE 1: NAN IS THE ONLY NEGATIVE NUMBER
        if np.sum(vals<0)==1:
            # print("%f%% of the data is %d" % (np.mean(a[key].astype(np.float)==mn), mn))
            a.loc[a[key]==mn, key] = np.nan
            b.loc[b[key]==mn, key] = np.nan
            vals = vals[vals!=mn]
            mn = np.min(vals)

        # OUTLIER TYPE 2: NUMBER STARTS WITH 9'S
        # (combined) OUTLIER TYPE 3: Second max number is less than 0.8 * max number
        max_string = str(np.int(mx))
        if max_string!="" and mn!=mx:
            max_digits = len(max_string)
            no_of_nines = 0
            cutoff = 0
            while no_of_nines<max_digits-1 and no_of_nines<2 and max_string[no_of_nines]=='9':
                no_of_nines += 1
            if no_of_nines > 0:
                cutoff = max_string[:no_of_nines].ljust(max_digits, '0')
                cutoff = np.int(cutoff) - 1
            else:
                cutoff = mx - 1.0
            if cutoff!=0:
                vals = vals[vals<=cutoff]
                if vals.shape[0]<=1:
                    continue
                mx = np.max(vals)
                ratio = mx / np.float(cutoff)
                # Don't accept cut-off as correct for outlier if ratio is above 90%
                if cutoff>50 and ratio < 0.9:
                    a.loc[a[key]>cutoff, key] = np.nan
                    b.loc[b[key]>cutoff, key] = np.nan

    return a, b

def recodeAlmostBinary(frameA, frameB, target, recodeKeys):
    # To find these keys:
    # almost_binary = (percent_nil+percent_mcv>=0.9)
    # almost_binary |= (percent_nil>0.8) & (percent_nil<0.9)
    # almost_binary |= (percent_mcv>0.8) & (percent_mcv<0.9)
    # almost_binary &= percent_nil<0.9 # These columns will be deleted anyway
    # almost_binary &= percent_mcv<0.9 # These columns will be deleted anyway
    # almost_binary_keys = a.keys()[almost_binary].values

    thres = 0.95

    frameA = frameA[recodeKeys].copy()
    frameB = frameB[recodeKeys].copy()
    rowcount = frameA.shape[0]
    indices = ( np.random.sample(200)*rowcount ).astype(np.int) # For speedup

    for key in recodeKeys:
        # Null values
        null_rows = pd.isnull(frameA[key])
        null_target = np.mean(target[null_rows])
        null_proportion = np.mean(null_rows)

        # Most common value
        vals = np.unique(frameA.loc[indices, key])
        vals = vals[pd.isnull(vals)==False]
        if vals.size==0:
            vals = np.unique(frameA[key])
            vals = vals[pd.isnull(vals)==False]
        mcv_val = np.nan
        mcv_proportion = 0.0
        for val in vals:
            proportion = np.mean(frameA[key]==val)
            if proportion>mcv_proportion:
                mcv_proportion = proportion
                mcv_val = val
        mcv_rows = frameA[key]==mcv_val
        mcv_target = np.mean(target[mcv_rows])
        tot_proportion = mcv_proportion + null_proportion

        if tot_proportion<thres:
            if null_proportion>mcv_proportion:
                frameA[key] = pd.isnull(frameA[key])
                frameB[key] = pd.isnull(frameB[key])
            else:
                frameA[key] = frameA[key]==mcv_val
                frameB[key] = frameB[key]==mcv_val
        elif np.sum(null_rows)==0:
            frameA[key] = frameA[key]==mcv_val
            frameB[key] = frameB[key]==mcv_val
        else:
            other_rows = (null_rows==False) & (mcv_rows==False)
            other_proportion = np.mean(other_rows)
            if mcv_proportion<0.01 and mcv_proportion<other_proportion:
                frameA[key] = pd.isnull(frameA[key])
                frameB[key] = pd.isnull(frameB[key])
            else:
                other_target = np.mean(target[other_rows])
                other_rowsB = (pd.isnull(frameB[key])==False) & (frameB[key]!=mcv_val)
                if abs(other_target-mcv_target)<abs(other_target-null_target):
                    frameA.loc[other_rows, key] = mcv_val
                    frameB.loc[other_rowsB, key] = mcv_val
                else:
                    frameA.loc[other_rows, key] = np.nan
                    frameB.loc[other_rowsB, key] = np.nan
                frameA[key] = frameA[key]==mcv_val
                frameB[key] = frameB[key]==mcv_val
        # print("%s   %.3f:%.3f   %.3f:%.3f   %.3f:%.3f   %s" % (key,\
        #     np.mean(null_rows), null_target,\
        #     np.mean(other_rows), other_target,
        #     np.mean(mcv_rows), mcv_target,
        #     "*" if mcv_proportion<0.01 and mcv_proportion<np.mean(other_rows) else ""))

    return frameA, frameB

def recodeDateText(a, b, recodeKeys):
    a = a.copy()
    b = b.copy()

    print("Extracting data from Date columns...")

    for key in recodeKeys:
        if key not in a.keys():
            continue
        a[key] = pd.to_datetime(a[key], format="%d%b%y:%H:%M:%S")
        b[key] = pd.to_datetime(b[key], format="%d%b%y:%H:%M:%S")

        a[key+'Y'] = a[key].map(lambda x: x.year)
        b[key+'Y'] = b[key].map(lambda x: x.year)
        a[key+'M'] = a[key].map(lambda x: x.month)
        b[key+'M'] = b[key].map(lambda x: x.month)
        a[key] = a[key].map(lambda x: x.weekday())
        b[key] = b[key].map(lambda x: x.weekday())

        # a.loc[pd.isnull(a[key]), [key, key+'Y',  key+'M']] = -1
        # b.loc[pd.isnull(b[key]), [key, key+'Y',  key+'M']] = -1

        # a[[key, key+'Y',  key+'M']] = a[[key, key+'Y',  key+'M']].astype(int)
        # b[[key, key+'Y',  key+'M']] = b[[key, key+'Y',  key+'M']].astype(int)

    return a, b

def recodeDateOther(a, b, recodeKeys):
    a = a.copy() # Work directly on a, b
    b = b.copy()

    for key in recodeKeys:
        if key not in a.keys():
            continue
        # Year
        a[key+'Y'] = a[key].map(lambda x: str(x)[:4])
        b[key+'Y'] = b[key].map(lambda x: str(x)[:4])
        a[key+'Y'] = a[key+'Y'].astype(float)
        b[key+'Y'] = b[key+'Y'].astype(float)
        # Month
        a[key] = a[key].map(lambda x: 'nan' if pd.isnull(x) else str(x)[4:])
        b[key] = b[key].map(lambda x: 'nan' if pd.isnull(x) else str(x)[4:])
        a[key] = a[key].astype(float)
        b[key] = b[key].astype(float)

    return a, b

# -------------------------------------------------------------------------------------------
# For Pruning:

def getGroups(mat, a, keys):
    assert(mat.dtype==np.bool)
    assert(len(mat.shape)==2)
    assert(mat.shape[0]==mat.shape[1])
    size = mat.shape[0]

    print("Grouping indices...")
    indices = [(i,j) for i in xrange(size) for j in xrange(size) if mat[i,j]==True]
    # Discard erroneous pairs (that do not have many matching values)
    indicesCleaned = []
    for i,j in indices:
        if keys[i] not in a.keys() or keys[j] not in a.keys():
            continue
        rows = (pd.isnull(a[keys[i]])==False)&(pd.isnull(a[keys[j]])==False)
        if np.sum(rows)>1000:
            indicesCleaned.append((keys[i],keys[j]))

    # Group highly correlated columns
    groups = []
    for i, j in indicesCleaned:
        in_a_group = False
        for group in groups:
            if i in group or j in group:
                in_a_group = True
                group.add(i)
                group.add(j)
                break
        if not in_a_group:
            groups.append(set([i,j]))
    # Merge groups containing the same keys
    n = 0
    while n<len(groups): # These loops could be made more efficient
        replaced = False
        for val in groups[n]:
            for m in xrange(len(groups)):
                if n>=m:
                    continue
                if val in groups[m]:
                    groups[n] |= groups[m]
                    groups.remove(groups[m])
                    replaced = True
                    break
            if replaced:
                break
        if not replaced:
            n += 1

    group_sizes = [len(i) for i in groups]
    print("Largest group sizes: %s" % ( np.sort(group_sizes)[::-1][:20] ))
    print("Discarding some of %d columns from each of %d groups..." % (np.sum(group_sizes), len(groups)))

    return groups

def regressWithinGroups(groups, thres, a, b, use_ranks=False, add_range=False):
    thres -= 0.01
    a = a.copy()
    b = b.copy()
    to_drop = set()

    for group in groups:
        group = list(group)
        keyI = group[0]
        is_linear_combo = False

        if use_ranks:
            useIa = rankdata(a[keyI]).copy()
            useIb = rankdata(b[keyI]).copy()
        else:
            useIa = a[keyI].copy()
            useIb = b[keyI].copy()
        
        for keyJ in group[1:]:
            if use_ranks:
                useJa = rankdata(a[keyJ])
                useJb = rankdata(b[keyJ])
            else:
                useJa = a[keyJ]
                useJb = b[keyJ]

            # Get equation to convert from one to the other
            rows = (pd.isnull(useIa)==False) & (pd.isnull(useJa)==False)
            colI = useIa[rows]
            colJ = useJa[rows]
            # Get to second parameter (I) from the first parameter (J), using linear regression
            m, c, r, p, err = linregress(colJ, colI)
            if abs(r)<thres:
                # print("Expected: %f    Got: %f" % (thres, r))
                group.remove(keyJ)
                continue
            if abs(r)==1.0:
                is_linear_combo = True
            
            # Fill in missing values
            rowsBlank = (pd.isnull(useIa)==True) & (pd.isnull(useJa)==False)
            useIa[rowsBlank] = m*useJa[rowsBlank] + c
            rowsBlank = (pd.isnull(useIb)==True) & (pd.isnull(useJb)==False)
            useIb[rowsBlank] = m*useJb[rowsBlank] + c
            
            # Discard latter column in pair
            to_drop.add(keyJ)

        # Update column with changes
        if len(group)>1:
            a[keyI] = useIa
            b[keyI] = useIb
            if add_range and not is_linear_combo: # Don't bother adding range if linear combination
                aCols = a[group]
                a[keyI+'range'] = aCols.max(axis=1) - aCols.min(axis=1)
                bCols = b[group]
                b[keyI+'range'] = bCols.max(axis=1) - bCols.min(axis=1)
        
        # # Fill in missing values in linearly correlated non-main columns
        # NOTE: TAKE RANKING INTO ACCOUNT
        # for keyJ in group[1:]:
        #     # Get equation to get from one to the other
        #     rows = (pd.isnull(a[keyI])==False) & (pd.isnull(a[keyJ])==False)
        #     colI = a.loc[rows, keyI]
        #     colJ = a.loc[rows, keyJ]
        #     m, c, r, p, err = linregress(colI, colJ)
        #     # Fill in missing values
        #     rowsBlank = (pd.isnull(a[keyI])==False) & (pd.isnull(a[keyJ])==True)
        #     a.loc[rowsBlank, keyJ] = m*a.loc[rowsBlank, keyI] + c
        #     rowsBlank = (pd.isnull(b[keyI])==False) & (pd.isnull(b[keyJ])==True)
        #     b.loc[rowsBlank, keyJ] = m*b.loc[rowsBlank, keyI] + c
        # # PCA
        # rowsA = pd.isnull(a[keyI])==False
        # rowsB = pd.isnull(b[keyI])==False
        # # scaler = StandardScaler()
        # # a.loc[rowsA, [keyI]+linKeys] = scaler.fit_transform(a.loc[rowsA, [keyI]+linKeys])
        # # b.loc[rowsB, [keyI]+linKeys] = scaler.fit_transform(b.loc[rowsB, [keyI]+linKeys])
        # pca = PCA(n_components=1)
        # a.loc[rowsA, keyI] = pca.fit_transform(a.loc[rowsA, [keyI]+linKeys])
        # b.loc[rowsB, keyI] = pca.transform(b.loc[rowsB, [keyI]+linKeys])

    print("Removing %d columns due to high correlation (>%f)..." % (len(to_drop), thres))
    a.drop(list(to_drop), axis=1, inplace=True)
    b.drop(list(to_drop), axis=1, inplace=True)

    print("Column count: %d" % (a.shape[1]))
    return a, b

# -------------------------------------------------------------------------------------------

# From: https://arcpy.wordpress.com/2012/04/20/146/
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)



