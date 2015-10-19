
import numpy as np
import pandas as pd
import pickle as pk
from time import time
from os.path import isfile
import gc
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, precision_recall_curve

# Load input dataset
print("Loading datasets...")
with open("../cache/train_binary_pruned.pkl", "r") as f, open("../data/trainTarget.pkl", "r") as g:
    X = pk.load(f) #.tocsr()
    gc.collect()
    y = pk.load(g).values
    gc.collect()

train_size = np.int(X.shape[0] * 0.9)

indices = np.random.permutation(np.arange(X.shape[0]))

Xtrain = X[indices[:train_size]]
ytrain = y[indices[:train_size]]
Xtest = X[indices[train_size:]]
ytest = y[indices[train_size:]]

print("Freeing memory")
# Free up memory somewhat
# a = None
X = None
y = None
gc.collect()

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
    'learning_rate' : [0.05],
    'min_samples_leaf' : [1, 3],
    'max_depth' : [3],
    'max_features' : [None],
    'subsample' : [0.85]
}
XB_params = {
    'max_depth' : [3],
    'learning_rate' : [0.1, 0.05],
    'subsample' : [0.85],
    'colsample_bytree' : [0.9],
    'gamma' : [0]
}

# params = {}
# params = log_params
# params = RF_params
# params = GB_params
params = XB_params

# model = SGDClassifier(loss="log", penalty="elasticnet", epsilon=100.0, alpha=1e-05, shuffle=True)
# model = RandomForestClassifier(n_estimators=1000, bootstrap=False)
# model = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=100, \
#     subsample=0.85, min_samples_split=2, min_samples_leaf=1, \
#     max_depth=3, init=None, random_state=None, max_features=None, verbose=0, \
#     max_leaf_nodes=None, warm_start=False)
model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, \
    objective='binary:logistic', nthread=1, gamma=0, min_child_weight=1, \
    max_delta_step=0, subsample=0.85, colsample_bytree=1)

def train(gridsearch=False, n_estimators=None):
    global model, Xtrain, ytrain

    Xtrain = Xtrain #.toarray()
    gc.collect()

    if gridsearch and bool(params):
        print("Finding best model parameters...")
        indices = np.random.permutation(np.arange(Xtrain.shape[0]))[:3000]
        clf = GridSearchCV(model, params, n_jobs=1, scoring='roc_auc')
        clf.fit(Xtrain[indices], ytrain[indices])

        print("Best parameters found:")
        print(clf.best_params_)
        model = clf.best_estimator_

    print("Training model")
    model.n_estimators = n_estimators if not None else model.n_estimators
    model.fit(Xtrain, ytrain, eval_metric='auc')

    # Clear data already used
    print("Clearing training data...")
    Xtrain = None
    ytrain = None
    gc.collect()

def test():
    global model, Xtest, ytest

    print("Testing model...")
    Xtest = Xtest #.toarray()
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

def save():
    global model

    with open("../data/test_binary_pruned.pkl", "r") as f, open("../data/testID.pkl", "r") as g:
        X = pk.load(f) #.toarray()
        gc.collect()
        bID = pk.load(g)
        gc.collect()

    ### SAVE TEST SET ###
    print("Predicting using test set data...")
    y_pred = model.predict_proba(X)[:,1]

    print("Saving file to disk...")
    test_ids = bID.values
    zOut = pd.DataFrame( zip(test_ids, y_pred), columns=["ID", "target"])

    path_out = "predictions"+str(int(time())-1441839000)+".csv"
    zOut.to_csv(path_out, sep=',', encoding='utf-8', index=False)

train(True, 100)
test()
save()

