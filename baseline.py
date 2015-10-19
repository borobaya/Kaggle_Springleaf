
import numpy as np
import pandas as pd
from time import time
import gc
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, precision_recall_curve

print("Loading data...")
# a = pd.read_pickle("train_clean.pkl")
# columns_to_use = (a.dtypes==np.float) | (a.dtypes==np.bool)

a = pd.read_csv('train.csv')
a['target'] = a['target'].astype(np.float).astype(np.bool)
a.drop(["ID"], axis=1, inplace=True)

columns_to_drop = a.dtypes==object
a.drop(a.keys()[columns_to_drop].values.tolist(), axis=1, inplace=True)
remaining_columns = np.ones([a.shape[1]], dtype=np.bool)

X = a[a.keys()[remaining_columns]].values[:,:-1]
X[pd.isnull(X)] = 0
y = a['target'].values

train_size = np.int(X.shape[0] * 0.9)


# # Scale appropriately
# scaler = StandardScaler()
# scaler.fit(X[:train_size]) # fit only on training data
# X = scaler.transform(X)

Xtrain = X[:train_size]
ytrain = y[:train_size]
Xtest = X[train_size:]
ytest = y[train_size:]

print("Freeing memory")
# Free up memory somewhat
a = None
X = None
y = None
gc.collect()

print("Finding best model parameters...")
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
	'min_samples_leaf' : [10, 20],
	'max_depth' : [3, 5],
	'max_features' : [None, "auto"],
	'subsample' : [0.8, 0.7]
}

# 'max_features': None, 'subsample': 0.8, 'max_depth': 3, 'min_samples_leaf': 10}

# params = {}
# params = log_params
# params = RF_params
params = GB_params

# model = SGDClassifier(loss="log", penalty="elasticnet", epsilon=100.0, alpha=1e-05, shuffle=True)
# model = RandomForestClassifier(n_estimators=1000, bootstrap=False)
model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, \
	subsample=0.8, min_samples_split=2, min_samples_leaf=1, \
	max_depth=3, init=None, random_state=None, max_features=2, verbose=0, \
	max_leaf_nodes=None, warm_start=False)

if True:
	clf = GridSearchCV(model, params, n_jobs=1, scoring='f1')
	clf.fit(Xtrain[:3000], ytrain[:3000])

	print("Best parameters found:")
	print(clf.best_params_)
	model = clf.best_estimator_

print("Training model")
model.n_estimators = 3000
model.fit(Xtrain, ytrain)

print("Testing model...")
ztest_proba = model.predict_proba(Xtest)[:,1]
ztest = ztest_proba>0.5

metrics =  "Cross-tabulation of test results:\n"
metrics +=  pd.crosstab(ytest, ztest, rownames=['actual'], colnames=['preds']).to_string()
metrics += "\n\n"

metrics +=  "Classification Report:\n"
metrics +=  classification_report(ytest, ztest>0.5)
metrics += "\n"

metrics +=  "AUC Score: " + str(roc_auc_score(ytest, ztest_proba))
metrics += "\n"

print(metrics)

### TEST SET ###

print("Loading test set data...")
b = pd.read_csv('test.csv')
test_ids = b["ID"].values
b.drop(["ID"], axis=1, inplace=True)
b.drop(b.keys()[columns_to_drop].values.tolist(), axis=1, inplace=True)

print("Predicting using test set data...")
X = b[b.keys()[remaining_columns[:-1]]].values
X[pd.isnull(X)] = 0
y_pred = model.predict_proba(X)[:,1]

print("Saving file to disk...")
zOut = pd.DataFrame( zip(test_ids, y_pred), columns=["ID", "target"])

path_out = "predictions"+str(int(time())-1441839000)+".csv"
zOut.to_csv(path_out, sep=',', encoding='utf-8', index=False)


