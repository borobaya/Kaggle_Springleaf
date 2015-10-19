
import numpy as np
import pandas as pd
import pickle as pk
from time import time
from os.path import isfile
import gc
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, precision_recall_curve

############################################################################

def outliers(a, b, coln=1):
    col = "VAR_%004g" % coln
    # if col not in a.keys():
    #     print("Column '%s' does not exist" % col)
    #     return a, b
    total = a.shape[0]
    # Replace "", "-" and "[]" values with NaN's
    # a.loc[a[col].str=='', col] = np.nan
    # a.loc[a[col].str=='-', col] = np.nan
    # a.loc[a[col].str=='[]', col] = np.nan
    # Make sure all values are numeric
    # for i in xrange(total):
    #     an_entry = a.loc[i, col]
    #     if hasattr(an_entry, 'replace'):
    #         break
    #     if i == total-1:
    #         print("Column has no good values")
    #         return a, b
    # if not an_entry.replace('.','').isdigit():
    #     print("Column is not a numeric data type")
    #     return a, b
    # Calculate summary statistics
    v = a[col].unique()
    v = v[pd.isnull(v)==False] # Remove NaN values
    if v.shape[0] <=1:
        return a, b
    min = np.min(v)
    max = np.max(v)
    # OUTLIER TYPE 1: NAN IS THE ONLY NEGATIVE NUMBER
    if np.sum(v<0)==1:
        # print("%f%% of the data is %d" % (np.mean(a[col].astype(np.float)==min), min))
        a.loc[a[col]==min, col] = np.nan
        b.loc[b[col]==min, col] = np.nan
        v = v[v!=min]
        min = np.min(v)
    # OUTLIER TYPE 2: NUMBER STARTS WITH 9'S
    # (combined) OUTLIER TYPE 3: Second max number is less than 0.8 * max number
    max_string = str(np.int(max))
    if max_string!="" and min!=max:
        max_digits = len(max_string)
        no_of_nines = 0
        cutoff = 0
        while no_of_nines<max_digits-1 and no_of_nines<2 and max_string[no_of_nines]=='9':
            no_of_nines += 1
        if no_of_nines > 0:
            cutoff = max_string[:no_of_nines].ljust(max_digits, '0')
            cutoff = np.int(cutoff) - 1
        else:
            cutoff = max - 1.0
        if cutoff!=0:
            v2 = v[v<=cutoff]
            if v2.shape[0]<=1:
                return a, b
            new_max = np.max(v2)
            ratio = new_max / np.float(cutoff)
            # Don't accept cut-off as correct for outlier if ratio is above 90%
            if cutoff>50 and ratio < 0.9:
                a.loc[a[col]>cutoff, col] = np.nan
                b.loc[b[col]>cutoff, col] = np.nan
    return a, b

def removeOutliers(a, b):
    for coln in xrange(np.int(a.keys()[-2][-4:])+1):
        col = "VAR_%004g" % coln
        if col in a.keys():
            # print("Column '%s' does not exist" % col)
            a, b = outliers(a, b, coln)

            # Replace NaNs with the mean / mode
            val = 0
            indices = np.random.permutation(np.arange(a.shape[0]))[:10000]
            sample = a[col][indices]

            nullCount = np.sum(pd.isnull(sample))
            nullFraction = nullCount/np.float(sample.shape[0])
            if nullFraction>0.8:
                print "Column:", col
                print("%d (%.1f%%) values are NaN!" % (nullCount, 100.0*nullFraction))
                # TODO: Make it True/False ???
            else:
                sample = sample[pd.isnull(sample)==False]
                vals, counts = np.unique(sample, return_counts=True)
                if vals.shape[0]<1000 and np.max(counts)>(0.1*sample.shape[0]):
                    val = vals[np.argmax(counts)]
                else:
                    val = np.int(np.mean(sample))

            a.loc[pd.isnull(a[col]), col] = val
            b.loc[pd.isnull(b[col]), col] = val

            # Convert all values to Boolean if there are only 2 unique values
            size = np.unique(a[col]).shape[0]
            if size==1:
                print("Dropping column '%s' with only 1 unique value..." % (col))
                a.drop(col, axis=1, inplace=True)
                b.drop(col, axis=1, inplace=True)
            elif size==2:
                print("Column '%s' is binary" % (col))
                mn = np.min(a[col])
                a.loc[a[col]==mn, col] = False
                b.loc[b[col]==mn, col] = False
                a[col] = a[col].astype(np.bool)
                b[col] = b[col].astype(np.bool)
                if np.unique(a[col]).shape[0]!=2:
                    print("WARNING: Data has been lost")

            temp = gc.collect()
    return a, b

############################################################################

### Data cleaning parameters ###
emptyRowIDs = [ 19557, 23643, 25823, 26606, 32800, 34703, 40755, \
    51095, 54627, 65755, 71950, 72792, 79851, 80412, 81536, 82180, \
    86705, 87153, 91644, 110304, 112373, 114369, 117156, 120414, \
    124105, 124822, 131228, 132013, 138555, 139947, 148774, 158360, \
    158851, 168762, 170188, 175629, 176336, 183753, 188527, 189262, \
    189307, 192082, 195773, 210219, 223967, 230996, 234104, 244618, \
    245028, 248723, 267804, 271797, 277426, 278457, 286441, 287745]
emptyColNames = ['VAR_0008', 'VAR_0009', 'VAR_0010', 'VAR_0011', \
    'VAR_0012', 'VAR_0043', 'VAR_0196', 'VAR_0207', 'VAR_0213', \
    'VAR_0214', 'VAR_0229', 'VAR_0239', 'VAR_0246', 'VAR_0530', \
    'VAR_0840']


if isfile("train_clean.pkl") and isfile("test_clean.pkl"):
    print("Loading datasets...")
    with open("train_clean.pkl", "r") as f, open("test_clean.pkl", "r") as g:
        a = pk.load(f)
        b = pk.load(g)
else:
    ### TRAINING SET ###
    print("Loading training set...")
    a = pd.read_csv('train.csv')#, nrows=60000)
    a['target'] = a['target'].astype(np.float).astype(np.bool)

    # Empty Rows
    a = a[a.ID.isin(emptyRowIDs)==False]
    # Empty Columns
    a.drop(emptyColNames, axis=1, inplace=True)
    # Drop complicated columns
    # ['VAR_0157', 'VAR_0214', 'VAR_0226', 'VAR_0230', 'VAR_0232', 'VAR_0236']
    columns_to_drop = a.dtypes==object
    a.drop(a.keys()[columns_to_drop].values.tolist(), axis=1, inplace=True)


    ### TEST SET ###
    print("Loading test set...")
    b = pd.read_csv('test.csv')#, nrows=100) # 167,177
    # Remove empty rows
    b = b[b.ID.isin(emptyRowIDs)==False]
    # Empty Columns
    b.drop(emptyColNames, axis=1, inplace=True)
    # Drop complicated columns
    b.drop(b.keys()[columns_to_drop].values.tolist(), axis=1, inplace=True)

    # ID column
    test_ids = b["ID"].values
    #b.drop(["ID"], axis=1, inplace=True)


    # Remove Outliers
    a, b = removeOutliers(a, b)

    ### Save ###
    with open("train_clean.pkl", "w") as f, open("test_clean.pkl", "w") as g:
        pk.dump(a, f, protocol=2)
        pk.dump(b, g, protocol=2)

# Create input dataset
X = a.values[:,1:-1]
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
    'learning_rate': [0.1, 0.01],
    'min_samples_leaf' : [12, 14],
    'max_depth' : [2, 3],
    'max_features' : ["auto"],
    'subsample' : [0.8]
}

# params = {}
# params = log_params
# params = RF_params
params = GB_params

# model = SGDClassifier(loss="log", penalty="elasticnet", epsilon=100.0, alpha=1e-05, shuffle=True)
# model = RandomForestClassifier(n_estimators=1000, bootstrap=False)
model = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=100, \
    subsample=0.8, min_samples_split=2, min_samples_leaf=14, \
    max_depth=2, init=None, random_state=None, max_features='auto', verbose=0, \
    max_leaf_nodes=None, warm_start=False)

def train(gridsearch=False, n_estimators=None):
    if gridsearch:
        print("Finding best model parameters...")
        clf = GridSearchCV(model, params, n_jobs=1, scoring='roc_auc')
        clf.fit(Xtrain[:3000], ytrain[:3000])

        print("Best parameters found:")
        print(clf.best_params_)
        model = clf.best_estimator_

    print("Training model")
    model.n_estimators = n_estimators if not None else model.n_estimators
    model.fit(Xtrain, ytrain)

def test():
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

def save():
    ### SAVE TEST SET ###
    print("Predicting using test set data...")
    X = b.values[:,1:]
    X[pd.isnull(X)] = 0
    y_pred = model.predict_proba(X)[:,1]

    print("Saving file to disk...")
    test_ids = b["ID"].values
    zOut = pd.DataFrame( zip(test_ids, y_pred), columns=["ID", "target"])

    path_out = "predictions"+str(int(time())-1441839000)+".csv"
    zOut.to_csv(path_out, sep=',', encoding='utf-8', index=False)

train(False, 10000)
test()
save()

