import numpy as np
import pandas as pd
import gc
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, precision_recall_curve
from scipy.stats import spearmanr, pearsonr

# Build model using only the highly correlated features

########################################################################
# Code to calculate Distance Correlation
# https://gist.github.com/josef-pkt/2938402

def dist(x, y):
    #1d only
    return np.abs(x[:, None] - y)
    

def d_n(x):
    d = dist(x, x)
    dn = d - d.mean(0) - d.mean(1)[:,None] + d.mean()
    return dn


def dcov_all(x, y):
    dnx = d_n(x)
    dny = d_n(y)
    
    denom = np.product(dnx.shape)
    dc = (dnx * dny).sum() / denom
    dvx = (dnx**2).sum() / denom
    dvy = (dny**2).sum() / denom
    dr = dc / (np.sqrt(dvx) * np.sqrt(dvy))
    # return dc, dr, dvx, dvy
    return dr

########################################################################

### START ###

a = pd.read_csv('train.csv')
a['target'] = a['target'].astype(np.float).astype(np.bool)
a.drop(a.keys()[a.dtypes==object].values.tolist(), axis=1, inplace=True)
a[pd.isnull(a)] = 0

# Pearson's Rank Correlation Coefficient
corrP = np.array([pearsonr(a[key], a["target"]) \
	if a[key].dtype!=object else (0,0) \
	for key in a.keys()])
corrP = np.abs(corrP[:,0])
useP = corrP>0.2

# Spearman's Rank Correlation Coefficient
corrS = np.array([spearmanr(a[key], a["target"]) \
	if a[key].dtype!=object else (0,0) \
	for key in a.keys()])
corrS = np.abs(corrS[:,0])
useS = corrS>0.13

# Maximal Information Coefficient

# TODO



# Distance Correlation Coefficient
corrD = np.zeros([a.keys().shape[0]])
i = 0
try:
	for i in xrange(corrD.shape[0]):
		key = a.keys()[i]
		if a[key].dtype==object:
			continue
		print("Doing i=%d" % (i))
		corrD[i] = dcov_all(a[key].values[:3000], a["target"].values[:3000])
		temp = gc.collect()
except Exception, e:
	print("Exception raised at i=%d: %s" % (i, e))
# corrD = np.array([dcov_all(a[key].values, a["target"].values) \
# 	if a[key].dtype!=object else 0 \
# 	for key in a.keys()])
useD = corrD>0.13

# http://blog.datadive.net/selecting-good-features-part-i-univariate-selection/
# http://minepy.sourceforge.net/
#  pip install https://pypi.python.org/packages/source/m/minepy/minepy-1.0.0.tar.gz

use = useS | useP
keys_to_discard = a.keys()[use==False]
a.drop(keys_to_use.values.tolist(), axis=1, inplace=True)





