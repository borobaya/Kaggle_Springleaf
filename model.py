#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle as pk
from time import time
from os.path import isfile
import os
import sys
import gc
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, precision_recall_curve

from lib import helper

# Parameters
timeshift = 1444000000
save_stamp = 0
is_ensemble = False
fn_binary = "binary_pruned"
fn_numeric = "numeric"
fn_ensemble = "ensemble_results"

def load():
    global Xtrain, Xtest, ytrain, ytest, indices_to_keep
    # Load input dataset
    print("Loading datasets...")
    with open("./cache/train_"+fn_binary+".pkl", "r") as f, open("./cache/train_"+fn_numeric+".pkl", "r") as g, \
        open("./data/trainTarget.pkl", "r") as h:
        X = pk.load(f)
        Xnumeric = pk.load(g)
        y = pk.load(h).values
    X = np.hstack([X, Xnumeric])

    if False:
        # Train on first/second half of the data separately for a proper ensemble
        halfpoint = int(X.shape[0]*0.5)

        if is_ensemble:
            with open("./cache/train_"+fn_ensemble+".pkl", "r") as f:
                Xensemble = pk.load(f)
            X = np.hstack([X, Xensemble])
            X = X[halfpoint:, :]
            y = y[halfpoint:]
        else:
            X = X[:halfpoint, :]
            y = y[:halfpoint]

    # indices_to_keep = np.random.permutation(np.arange(X.shape[1]))[:int(X.shape[1]*1.0)]
    # X = X[:, indices_to_keep]

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=5000)

    # Free up memory somewhat
    print("Freeing memory...")
    X = None
    y = None
    Xnumeric = None
    gc.collect()

def init_model():
    global model, params
    print("Initiating model...")
    log_params = {
         "loss": ["log"],
         "penalty": ["l1", "l2"], # "elasticnet"
         "alpha": 10.0**-np.arange(1,7),
         "epsilon": 10.0**np.arange(2,7),
         "n_iter": [5],
         "shuffle": [True]
        }
    RF_params = {
        'n_estimators' : [1000],
        'bootstrap' : [False]
        }
    GB_params = {
        'learning_rate': [0.05],
        'min_samples_leaf' : [1, 3],
        'max_depth' : [3],
        'max_features' : [None],
        'subsample' : [0.85]
    }
    XB_params = {
        'max_depth' : [5],
        'learning_rate' : [0.03],
        'subsample' : [0.8, 0.85, 0.9],
        'colsample_bytree' : [0.9, 1.0],
        'gamma' : [0],
        'min_child_weight': [60,70]
    }

    # params = {}
    # params = log_params
    # params = RF_params
    # params = GB_params
    params = XB_params

    # model = SGDClassifier(loss="log", penalty="elasticnet", epsilon=100.0, alpha=1e-05, shuffle=True)
    # model = RandomForestClassifier(n_estimators=1000, bootstrap=False)
    # model = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=50, \
    #     subsample=0.85, min_samples_split=2, min_samples_leaf=1, \
    #     max_depth=3, init=None, random_state=None, max_features=None, verbose=0, \
    #     max_leaf_nodes=None, warm_start=False)
    model = XGBClassifier(max_depth=4, learning_rate=0.02, n_estimators=100, silent=True, \
        objective='binary:logistic', nthread=1, gamma=0, min_child_weight=30, \
        max_delta_step=0, subsample=0.9, colsample_bytree=0.92)

def train(gridsearch=False, n_estimators=None):
    global model, params, Xtrain, ytrain

    if gridsearch and bool(params):
        print("Finding best model parameters...")
        indices = np.random.permutation(np.arange(Xtrain.shape[0]))[:3000]
        clf = GridSearchCV(model, params, n_jobs=1, scoring='roc_auc')
        clf.fit(Xtrain[indices], ytrain[indices])

        print("Best parameters found:")
        print(clf.best_params_)
        model = clf.best_estimator_

    n = 5000
    indices = np.random.permutation(np.arange(Xtrain.shape[0]))
    Xval, yval = Xtrain[indices[:n]], ytrain[indices[:n]]

    print("Training model...")
    model.n_estimators = n_estimators if not None else model.n_estimators
    model.fit(Xtrain[indices[n:]], ytrain[indices[n:]],
        eval_set=[(Xval, yval)],
        eval_metric='auc',
        early_stopping_rounds=30)

    # Clear data already used
    print("Clearing training data...")
    Xtrain = None
    ytrain = None
    gc.collect()

def test():
    global model, Xtest, ytest

    print("Testing model...")
    ztest_proba = model.predict_proba(Xtest)[:,1]
    ztest = ztest_proba>0.5

    metrics =  "Cross-tabulation of test results:\n"
    metrics +=  pd.crosstab(ytest, ztest, rownames=['actual'], colnames=['preds']).to_string()
    metrics += "\n\n"

    metrics +=  "Classification Report:\n"
    metrics +=  classification_report(ytest, ztest)
    metrics += "\n"

    metrics +=  "AUC Score: " + str(roc_auc_score(ytest, ztest_proba))
    metrics += "\n"

    print(metrics)

    print("Clearing testing data...")
    Xtest = None
    ytest = None
    gc.collect()

def cvtrain(n_estimators=100):
    global model, Xtrain, ytrain
    model.n_estimators = n_estimators if not None else model.n_estimators

    print("Performing cross validation...")
    scores = cross_val_score(model, Xtrain, ytrain, cv=5, scoring='roc_auc',
        fit_params={'eval_metric':'auc'}, n_jobs=1, pre_dispatch=1)

    print("ROC AUC Scores: %s" % (scores))
    print("Average ROC AUC Score: %f" %(np.mean(scores)))

def saveTest():
    global model, save_stamp, indices_to_keep

    print("Loading validation set...")
    with open("./cache/test_"+fn_binary+".pkl", "r") as f, open("./cache/test_"+fn_numeric+".pkl", "r") as g, \
        open("./data/testID.pkl", "r") as h:
        X = pk.load(f)
        Xnumeric = pk.load(g)
        ids = pk.load(h).values
    X = np.hstack([X, Xnumeric])

    if is_ensemble:
        with open("./cache/test_"+fn_ensemble+".pkl", "r") as f:
            Xensemble = pk.load(f)
        X = np.hstack([X, Xensemble])

    # X = X[:, indices_to_keep]

    Xnumeric = None
    gc.collect()

    ### SAVE TEST SET ###
    print("Predicting on test set data...")
    y_pred = model.predict_proba(X)[:,1]

    print("Saving file to disk...")
    zOut = pd.DataFrame( zip(ids, y_pred), columns=["ID", "target"])

    save_stamp = str(int(time())-timeshift)
    path_out = "./predictions/test_predictions"+save_stamp+".csv"
    zOut.to_csv(path_out, sep=',', encoding='utf-8', index=False)

def saveTrain():
    global model, save_stamp, indices_to_keep

    print("Loading train set...")
    with open("./cache/train_"+fn_binary+".pkl", "r") as f, open("./cache/train_"+fn_numeric+".pkl", "r") as g, \
        open("./data/trainID.pkl", "r") as h:
        X = pk.load(f)
        Xnumeric = pk.load(g)
        ids = pk.load(h).values
    X = np.hstack([X, Xnumeric])
    
    if is_ensemble:
        with open("./cache/train_"+fn_ensemble+".pkl", "r") as f:
            Xensemble = pk.load(f)
        X = np.hstack([X, Xensemble])

    # X = X[:, indices_to_keep]

    Xnumeric = None
    gc.collect()

    ### SAVE TRAIN SET ###
    print("Predicting on train set data...")
    y_pred = model.predict_proba(X)[:,1]

    print("Saving file to disk...")
    zOut = pd.DataFrame( zip(ids, y_pred), columns=["ID", "target"])

    path_out = "./predictions/train_predictions"+save_stamp+".csv"
    zOut.to_csv(path_out, sep=',', encoding='utf-8', index=False)

def run():
    tic = time()
    n_estimators = 3000

    load()
    init_model()
    # cvtrain(n_estimators)
    train(False, n_estimators)
    test()
    saveTest()
    saveTrain()

    toc = time()-tic
    print("Time elapsed: %s" % (helper.hms_string(toc)))

    os.system('say "Finished"')

run()

