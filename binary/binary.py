
import numpy as np
import pandas as pd
import pickle as pk
from time import time
from datetime import datetime
from pprint import pprint
from os.path import isfile
import sys
import os
import re
import gc
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

if "../lib" not in sys.path:
    sys.path.append("../lib")
import params
import helper

############################################################################

def recodeFactors(a, b, recodeKeys):
    aTarget = a['target']
    keys = [i for i in recodeKeys if i in a.keys() and a[i].dtype!=np.bool]
    a = a[keys].copy()
    b = b[keys].copy()

    # Merging values with similar target percentages
    for key in keys:
        vals = np.unique(a[key])
        if np.sum(pd.isnull(vals))>0:
            vals = np.append(vals[pd.isnull(vals)==False], np.nan)
        count = len(vals)

        tCents = {}
        for val in vals:
            rows = pd.isnull(a[key]) if pd.isnull(val) else a[key]==val
            target_percent = np.mean(aTarget[rows])
            if pd.isnull(val):
                tCents[np.nan] = target_percent
            else:
                tCents[val] = target_percent

        isMerge = np.ones([count, count])
        for i, val1 in enumerate(vals):
            for j, val2 in enumerate(vals):
                if i>=j:
                    continue
                cent1 = tCents[np.nan] if pd.isnull(val1) else tCents[val1]
                cent2 = tCents[np.nan] if pd.isnull(val2) else tCents[val2]
                isMerge[i,j] = np.abs(cent1-cent2)
        isMerge = isMerge<0.02

        # Merge linked groups together
        for i in xrange(isMerge.shape[0]):
            replaces = isMerge[i, :]
            join = np.sum(isMerge[replaces, :], axis=0)
            isMerge[i, :] = np.bitwise_or(replaces, join)
            isMerge[replaces, :] = False

        # Dictionary of which value replaces which other value
        merge = {}
        for i, val in enumerate(vals):
            isMergeVal = [False]*i + list(isMerge[i, i:]) # isMerge[i] if i==0 else 
            isMergeVal = np.array(isMergeVal)
            if pd.isnull(val):
                merge[np.nan] = list(vals[isMergeVal])
            else:
                merge[val] = list(vals[isMergeVal])

        for val in merge:
            replaces = merge[np.nan] if pd.isnull(val) else merge[val]
            for val2 in replaces:
                if pd.isnull(val2):
                    a.loc[pd.isnull(a[key]), key] = val
                    b.loc[pd.isnull(b[key]), key] = val
                else:
                    a.loc[a[key]==val2, key] = val
                    b.loc[b[key]==val2, key] = val

        a.loc[pd.isnull(a[key]), key] = -1
        b.loc[pd.isnull(b[key]), key] = -1

    # Recode negative numbers and floats
    print("recodeFactors: (warning) number of NaNs: %d" % (np.sum(pd.isnull(a.values))))
    a[pd.isnull(a)] = -1
    b[pd.isnull(b)] = -1
    a = np.abs(a.astype(np.int))
    b = np.abs(b.astype(np.int))

    # Handle remaining numeric columns
    one = OneHotEncoder(sparse=True, dtype=np.bool, handle_unknown="ignore")
    print("Fitting OneHotEncoder...")
    one.fit(a)
    temp = gc.collect()
    print("One hot encoding on a...")
    a = one.transform(a).tocsr().astype(bool)
    temp = gc.collect()
    print("One hot encoding on b...")
    b = one.transform(b).tocsr().astype(bool)
    temp = gc.collect()

    return a, b

def recodeBinary(a, b, recodeKeys):
    aTarget = a['target']
    keys = [i for i in recodeKeys if i in a.keys()]
    a = a[keys].copy()
    b = b[keys].copy()

    print("Transforming binary columns...")

    for key in keys:
        vals = np.unique(a[key])
        if np.sum(pd.isnull(vals))>0:
            vals = np.append(vals[pd.isnull(vals)==False], np.nan)
        count = len(vals)

        trueVals = []
        falseVals = []

        tCents = {}
        for val in vals:
            if val in [0, False]:
                falseVals.append(val)
            if val in [1, True]:
                trueVals.append(val)

            rows = pd.isnull(a[key]) if pd.isnull(val) else a[key]==val
            target_percent = np.mean(aTarget[rows])
            if pd.isnull(val):
                tCents[np.nan] = target_percent
            else:
                tCents[val] = target_percent

        if count<=1:
            continue
        if count==2:
            if len(trueVals)==1 and len(falseVals)==0:
                falseVals = [val for val in vals if val not in trueVals]
            if len(trueVals)==0 and len(falseVals)==1:
                trueVals = [val for val in vals if val not in falseVals]

        unknownVals = [c for c in vals if c not in trueVals+falseVals and pd.isnull(c)==False]
        if np.sum(pd.isnull(c))>0 and np.sum(pd.isnull(trueVals+falseVals))==0:
            unknownVals.append(np.nan)
        for val in unknownVals:
            valCent = tCents[np.nan] if pd.isnull(val) else tCents[val]
            trueDist = np.mean([np.abs(tCents[np.nan if pd.isnull(v) else v]-valCent) for v in trueVals])
            falseDist = np.mean([np.abs(tCents[np.nan if pd.isnull(v) else v]-valCent) for v in falseVals])
            if trueDist<falseDist:
                trueVals.append(np.nan if pd.isnull(val) else val)
            else:
                falseVals.append(np.nan if pd.isnull(val) else val)

        # Sanity Check
        unknownVals = [c for c in vals if c not in trueVals+falseVals and pd.isnull(c)==False]
        if np.sum(pd.isnull(c))>0 and np.sum(pd.isnull(trueVals+falseVals))==0:
            unknownVals.append(np.nan)
        if len(unknownVals)>0:
            print key, unknownVals, falseVals, trueVals

        for val in falseVals:
            if pd.isnull(val):
                a.loc[pd.isnull(a[key]), key] = False
                b.loc[pd.isnull(b[key]), key] = False
            else:
                a.loc[a[key]==val, key] = False
                b.loc[b[key]==val, key] = False
        for val in trueVals:
            if pd.isnull(val):
                a.loc[pd.isnull(a[key]), key] = True
                b.loc[pd.isnull(b[key]), key] = True
            else:
                a.loc[a[key]==val, key] = True
                b.loc[b[key]==val, key] = True

    a = sparse.csr_matrix( a.astype(np.bool) ).astype(bool)
    b = sparse.csr_matrix( b.astype(np.bool) ).astype(bool)

    return a, b

def recodeFactorText(a, b, recodeKeys):
    aTarget = a['target']
    keys = [i for i in recodeKeys if i in a.keys()]
    a = a[keys].copy()
    b = b[keys].copy()

    # Transform all text columns
    print("Transforming text columns...")

    for key in keys:
        vals = np.unique(a[key])
        if np.sum(pd.isnull(vals))>0:
            vals = np.append(vals[pd.isnull(vals)==False], np.nan)
        count = len(vals)

        tCents = {}
        for val in vals:
            rows = pd.isnull(a[key]) if pd.isnull(val) else a[key]==val
            target_percent = np.mean(aTarget[rows])
            if pd.isnull(val):
                tCents[np.nan] = target_percent
            else:
                tCents[val] = target_percent

        isMerge = np.ones([count, count])
        for i, val1 in enumerate(vals):
            for j, val2 in enumerate(vals):
                if i>=j:
                    continue
                cent1 = tCents[np.nan] if pd.isnull(val1) else tCents[val1]
                cent2 = tCents[np.nan] if pd.isnull(val2) else tCents[val2]
                isMerge[i,j] = np.abs(cent1-cent2)
        isMerge = isMerge<0.02

        for i in xrange(isMerge.shape[0]):
            replaces = isMerge[i, :]
            join = np.sum(isMerge[replaces, :], axis=0)
            isMerge[i, :] = np.bitwise_or(replaces, join)
            isMerge[replaces, :] = False

        # Dictionary of which value replaces which other value
        merge = {}
        for i, val in enumerate(vals):
            isMergeVal = [False]*i + list(isMerge[i, i:]) # isMerge[i] if i==0 else 
            isMergeVal = np.array(isMergeVal)
            if pd.isnull(val):
                merge[np.nan] = list(vals[isMergeVal])
            else:
                merge[val] = list(vals[isMergeVal])

        for val in merge:
            replaces = merge[np.nan] if pd.isnull(val) else merge[val]
            for val2 in replaces:
                if pd.isnull(val2):
                    a.loc[pd.isnull(a[key]), key] = val
                    b.loc[pd.isnull(b[key]), key] = val
                else:
                    a.loc[a[key]==val2, key] = val
                    b.loc[b[key]==val2, key] = val

    # Replace NaNs
    a[pd.isnull(a)] = 'nan'
    b[pd.isnull(b)] = 'nan'

    # Vectorising
    vec = DictVectorizer(sparse=False)
    vec.fit(a.to_dict(orient='records'))
    a = vec.transform(a.to_dict(orient='records')).astype(bool)
    b = vec.transform(b.to_dict(orient='records')).astype(bool)
    temp = gc.collect()
    a = sparse.csr_matrix(a).astype(bool)
    b = sparse.csr_matrix(b).astype(bool)
    temp = gc.collect()

    return a, b

def recodeText(a, b, recodeKeys):
    aTarget = a['target']
    keys = [i for i in recodeKeys if i in a.keys()]
    a = a[keys].copy()
    b = b[keys].copy()

    # Recode NaNs
    a[pd.isnull(a)] = ''
    b[pd.isnull(b)] = ''
    a[a=='-1'] = ''
    b[b=='-1'] = ''

    # Join text
    a = a['VAR_0404']+' '+a['VAR_0493']
    b = b['VAR_0404']+' '+b['VAR_0493']
    # a = a[keys].apply(lambda x: ' '.join(x), axis=1)
    # b = b[keys].apply(lambda x: ' '.join(x), axis=1)
    a = a.map(str.strip)
    b = b.map(str.strip)

    vectorizer = CountVectorizer(strip_accents='ascii', min_df=5, binary=True, dtype=np.bool, tokenizer=stringTokenizer)
    vectorizer.fit(a)

    a = vectorizer.transform(a)
    b = vectorizer.transform(b)

    return a, b

def stringTokenizer(s):
    pattern = r"(?u)\b\w\w+\b"
    token_pattern = re.compile(pattern)
    s2 = token_pattern.findall(s)
    s2 = [s[:4] for s in s2]
    return s2

def recodeInvalids(a, b):
    aTarget = a['target']
    keys = [i for i in params.otherCols if i in a.keys()]
    a = a[keys].copy()
    b = b[keys].copy()
    total = a.shape[0]

    print("Dealing with NaNs...")
    for key in keys:
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

    a = pd.isnull(a)
    b = pd.isnull(b)

    return a, b

############################################################################

def clean():
    global a, b

    # Load Data
    print("Loading datasets...")
    a = pd.read_csv('../data/train.csv')
    b = pd.read_csv('../data/test.csv') # 167,177
    aTarget = a['target'].astype(np.bool)

    print("Dropping columns")
    to_drop = params.emptyColNames + params.locationColNames
    to_drop = list(set(to_drop))
    a.drop(to_drop, axis=1, inplace=True)
    b.drop(to_drop, axis=1, inplace=True)

    # Transform Columns
    print("Recoding columns")

    # Dates
    a, b = helper.recodeDateText(a, b, params.dateColNames)
    a, b = helper.recodeDateOther(a, b, params.dateOtherColNames)
                            # a, b need to be directly assigned
                            # since recodeFactors works on them next
    params.factorColNames = params.dateColNames + \
                          [i+'M' for i in params.dateColNames] + \
                          [i+'Y' for i in params.dateColNames] + \
                          params.dateOtherColNames + \
                          [i+'Y' for i in params.dateOtherColNames]

    # Numeric but almost binary
    # a, b = helper.findNans(a, b, params.almostBinaryColNames) # For the next line to work properly
    # aAlmostBinary, bAlmostBinary = helper.recodeAlmostBinary(a, b, aTarget, params.almostBinaryColNames)
    
    # Other
    aFactors, bFactors = recodeFactors(a, b, params.factorColNames)
    a01, b01 = recodeBinary(a, b, params.binaryColNames)
    aFactorText, bFactorText = recodeFactorText(a, b, params.factorTextColNames)
    aText, bText = recodeText(a, b, params.textColNames)
    # aOther, bOther = recodeInvalids(a, b)

    print("Joining matrices...")
    # aR = sparse.hstack([aAlmostBinary, aFactors, a01, aFactorText, aText]).astype(bool).tocsr()
    # bR = sparse.hstack([bAlmostBinary, bFactors, b01, bFactorText, bText]).astype(bool).tocsr()
    aR = sparse.hstack([aFactors, a01, aFactorText, aText]).astype(bool).tocsr()
    bR = sparse.hstack([bFactors, b01, bFactorText, bText]).astype(bool).tocsr()

    # Save
    print("Saving...")
    with open("../cache/train_binary.pkl", "w") as f, open("../cache/test_binary.pkl", "w") as g:
        pk.dump(aR, f, protocol=2)
        pk.dump(bR, g, protocol=2)

    os.system('say "Finished"')


clean()



