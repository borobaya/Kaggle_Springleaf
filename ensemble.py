
import numpy as np
import pandas as pd
import pickle as pk
from time import time
from os.path import isfile
import os
import gc

both_train_and_test_must_exist = True
timeshift = 1444000000
path = "./predictions/"

def load():
    global test_results, train_results
    # Load all result predictions
    test_results = []
    train_results = []
    filenames = os.listdir(path)
    print("Loading preditions...")

    for i in xrange(len(filenames)):
        filename = filenames[i]
        if "test_predictions" not in filename:
            continue
        train_filename = filename.replace("test", "train")
        if isfile(path+train_filename):
            r = pd.read_csv(path+train_filename)
            train_results.append( r['target'].values )
        elif both_train_and_test_must_exist:
            continue
        r = pd.read_csv(path+filename)
        test_results.append( r['target'].values )

    test_results = np.array(test_results).transpose()
    train_results = np.array(train_results).transpose()

def save_matrices():
    global test_results, train_results
    if test_results.shape[1]!=train_results.shape[1]:
        print("Ensemble dataset not created. Different numbers of train and test data found.")
        return
    # Save matrix of all result predictions
    with open("./cache/test_ensemble_results.pkl", "w") as f, open("./cache/train_ensemble_results.pkl", "w") as g:
        pk.dump(test_results, f, protocol=2)
        pk.dump(train_results, g, protocol=2)

def save_ensemble():
    global test_results
    # Save ensembled results
    test_pred = test_results.mean(axis=1)
    with open("./data/testID.pkl", "r") as f:
        test_ids = pk.load(f).values
    zOut = pd.DataFrame( zip(test_ids, test_pred), columns=["ID", "target"])
    path_out = path+"test_ensemble"+str(int(time())-timeshift)+".csv"
    zOut.to_csv(path_out, sep=',', encoding='utf-8', index=False)

def run(both_must_exist=None):
    global test_results, both_train_and_test_must_exist

    if both_must_exist is not None:
        both_train_and_test_must_exist = both_must_exist

    load()
    
    if test_results.size==0:
        print("No results data loaded")
        return
    
    # save_matrices()
    save_ensemble()

run()
