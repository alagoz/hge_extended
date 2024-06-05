"""
Repository to call classifier algorithms
"""

# built-in libraries
import numpy as np
import os
from sklearn.linear_model import RidgeClassifierCV as Ridge
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# libraries to be installed
from xgboost import XGBClassifier # pip install xgboost

#Parameters
rs=0
alphas = np.logspace(-3, 3, 10)
n_jobs = os.cpu_count()-0

n_est=500 # for xgb
lr_=0.01  # for xgb

clf_dict={
# ensemble
'xgb': XGBClassifier(),
'rf': RandomForestClassifier(),
'gb': GradientBoostingClassifier(),

# classics
'svc': SVC(),
'svmLin': SVC(kernel='linear',probability=True),
'svmPol': SVC(kernel='poly',gamma='auto',probability=True),
'svmRbf': SVC(kernel='rbf',gamma='auto',probability=True),
'lor': LR(C=1, solver='saga', max_iter=500000, n_jobs=n_jobs,random_state=rs),
'gnb': GaussianNB(),
'mnb': MultinomialNB(alpha=0.001),
'lda': LinearDiscriminantAnalysis(), #solver='lsqr')
'qda': QuadraticDiscriminantAnalysis(),
'ridge': Ridge(alphas=alphas),
'cart': DecisionTreeClassifier(random_state=rs)
}