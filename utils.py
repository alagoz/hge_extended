# built-in libraries
import numpy as np
import pandas as pd
import os
import logging
import sys
import pickle
import warnings
import math
import seaborn as sns
from time import time
from itertools import combinations
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from scipy import stats
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import (TSNE,
                              LocallyLinearEmbedding as lle,
                              Isomap,)

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score,
                             f1_score,
                             roc_auc_score,
                             log_loss,)
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler,MinMaxScaler
# libraries to be installed
import prince

def select_classes(sel_class, y, x=None, reorder=True):
    sel_inds = np.array([],dtype=int)
    y_sel = y.copy()
    for i,g in enumerate(sel_class):            
        # find where indices for selected memberships occur
        if (type(g) is list or type(g) is np.ndarray) and len(g)>1:
            for j in g:
                loc_ = np.where(y==j)[0].astype(int)                    
                sel_inds = np.r_[sel_inds,loc_]
                if reorder:
                    y_sel[y==j]=i
                else:
                    y_sel[y==j]=g[0]
        else:
            loc_ = np.where(y==g)[0].astype(int)                
            sel_inds = np.r_[sel_inds,loc_]           
            if reorder:
                y_sel[y==g]=i
            else:
                y_sel[y==g]=g
    y_sel = y_sel[sel_inds]
    if x is None:
        return y_sel
    else:        
        x_sel = x[sel_inds,:].copy()         
        return y_sel, x_sel
        
def class_labels_sanity_check(y_train,y_test):
    classes_train = np.unique(y_train)
    classes_test = np.unique(y_test)
    if np.all(classes_train==classes_test):
        classes=classes_train
    else:
        raise ValueError("Discrepancy bw train and test labels. Class labels don't match.")
    return classes
   
def split_data(x_all,
               y_all, 
               stratify=True, 
               k=5, 
               kth=1,
               rs=None,
               shuffle_=False):
    np.random.seed(seed=None)
    # split the data into training, validation and test set
    if stratify:    
        kf= StratifiedKFold(n_splits=k, random_state=rs, shuffle=shuffle_)
    else:
        kf=KFold(n_splits=k, random_state=rs, shuffle=shuffle_)
    i = 0
    for tr,te in kf.split(x_all,y_all):
        # print('train -  {}   |   test -  {}'.format(np.bincount(y_all[tr]), np.bincount(y_all[te])))
        x_train, x_test = x_all[tr,:], x_all[te,:]
        y_train, y_test = y_all[tr], y_all[te]
        i += 1
        if i==kth:
            break
    # X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=1/k, random_state=rseed, stratify=y_all)    
    return x_train, x_test, y_train, y_test

def quantize(x,qlevel=2**8):
    return (x*qlevel/(x.max()-x.min())).astype(int) 

def plotData(data,labels, close_all=True):
    if close_all:plt.close('all')
    classes = np.unique(labels)
    plt.figure()
    for c in classes:
        c_x_train = data[labels == c]
        plt.plot(c_x_train[0], label="class " + str(c))
    plt.legend(loc="best")

def log_config(log_fname='log_file',log_=0):
    logd={0:print,1:logging.info}
    
    # Log info        
    log_file = '%s.log'%(log_fname) 
    targets = logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)
    logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=targets)
    
    return logd[log_]
   
def monotonize_rescale_(Z,swap=False):
    # Monotonize and rescale the tree linkage
    Zh = Z[:,2]
    n = len(Zh)
    
    start_ = Zh.max()/(n)
    stop_ = Zh.max()
    Zh[:] = np.round(np.linspace(start_,stop_,n),2)
    
    if Zh.ptp()/(n-1) < 0.05:
        scale_=0.05*(n-1)/Zh.ptp()
        Zh *= scale_  
    
    # Swap column 0 and 1 for compatiblity with dendogram
    if swap:
        z0=Z[:,0].copy()
        Z[:,0]=Z[:,1]
        Z[:,1]=z0
    
    return Z

def parse_ndarray_from_csv(df,col_name,row_id,shape_=None,wrtype=None):
    # df = pd.read_csv(fname)
    if row_id is None:
        n_row = df.shape[0]
        row_id = n_row-1
    obj = df[col_name][row_id]
    
    if wrtype is None:
        if '\n' in obj:
            wrtype='ndarray'
        else:
            wrtype='list'
    
    if wrtype == 'ndarray':        
        for char in "[]\n":
            obj=obj.replace(char,"")    
        
        elems_=obj.split(sep=' ')    
        temp_list=[]
        for i in elems_:
            if i != '':
                temp_list.append(eval(i))
        if shape_ is None:        
            n=int(np.sqrt(len(temp_list)))
            arr=np.array(temp_list).reshape(n,n)
        else:
            arr=np.array(temp_list).reshape(shape_)
    
    elif wrtype == 'list':
        arr=np.array(eval(obj))
    return arr

def generate_rand_dist(n_dim,type_='diss'):
    np.random.seed(seed=None)    
    d = np.zeros((n_dim,n_dim))
    
    a = 1
    b = 0
    if type_ == 'perturb':
        a = 2
        b = 0.5
    
    for i in combinations(range(n_dim),2):        
        rr=np.round(a*(np.random.rand()-b),2)
        d[i]=rr
        d[i[::-1]]=rr
        
    return d

def nCk(n,k):
    f = math.factorial
    return f(n) // f(k) // f(n-k)

def C_n(n):
    """All divisions of an n-element cluster into two non-empty 
       subsets: 2**(n-1)-1
    """
    sum_=0
    for k in range(1,round(n/2)+1):
        if n%2==0 and k==n/2:
            sum_ += int(nCk(n,k)/2)
        else:
            sum_ += nCk(n,k)
    print(sum_)
    return sum_

def T_n(n):
    "Estimate total number of trees given number of classes"    
    if n==2:        
        return 1
    elif n==3:
        return 3
    elif n>3:
        sum_=0
        for i in range(1,round(n/2)+1):
            if n%2==0 and i==n/2:
                sum_ += int(nCk(n,n-i)/2)*T_n(n-i)
            else:
                sum_ += nCk(n,n-i)*T_n(n-i)
        return sum_

def T_n_look(n):
    "Estimate total number of trees given number of classes"
    table_=[0,0]
    if n==2:
        table_.append(1)
        return table_
    elif n==3:
        table_.append(1)
        table_.append(3)
        return table_
    elif n>3:
        table_.append(1)
        table_.append(3)
        for n_i in range(4,n+1):
            sum_=0           
            for k in range(1,round(n_i/2)+1):
                if n_i%2==0 and k==n_i/2:
                    sum_ += int(nCk(n_i,n_i-k)/2)*table_[n_i-k]                    
                else:
                    sum_ += nCk(n_i,n_i-k)*table_[n_i-k]
            table_.append(sum_)
        return table_
    
def compare_(root1, root2, sol):
    if root1 is not None and root2 is not None:
        if root1.subsets[0]==root2.subsets[0]:
            sol.append(1)
            compare_(root1.left, root2.left, sol)
            compare_(root1.right, root2.right, sol)
        elif root1.subsets[0]==root2.subsets[1]:
            sol.append(2)
            compare_(root1.left, root2.right, sol)
            compare_(root1.right, root2.left, sol)
        else:            
            sol.append(0)
    else:
        sol.append(-1)

def isEqual(root1, root2):
    eq_=[]
    compare_(root1,root2,eq_)
    if 0 in eq_:
        return False
    else:
        return True

def compare_tree(n_i,Yi,n_j,Yj):
    if np.all(Yi==Yj):
        return True
    elif isEqual(n_i[0],n_j[0]):
        return True
    else:
        return False
    
def build_Y(Z):
    Y = np.zeros((Z.shape[0]+1,Z.shape[0]))
    for i in range(Z.shape[0]):
        Y[:,i]=sch.cut_tree(Z, n_clusters=i+2).squeeze()
    return Y

def plot_settings(use_sns=False):
    # Plot settings
    mpl.rcParams['figure.dpi'] = 100
    plt.close('all')
    # use_sns = 0
    if use_sns:    
        sns.set_style('darkgrid', {'axes.facecolor': '.9'})
        sns.set_palette(palette='deep')
        sns_c = sns.color_palette(palette='deep')
        sns.set(rc={'figure.figsize':(7.4,4.2)})

def plot_dendogram(Z, close_all=0, orient="top", leafFont=9, title_=False, class_list=None):
    if close_all:plt.close('all')
    if title_: title_text= "Hierarchical Clustering Dendrogram"
    if class_list is not None:
        sch.dendrogram(Z,orientation=orient,leaf_font_size=leafFont,labels=[txt for txt in class_list])
    else:
        sch.dendrogram(Z,orientation=orient,leaf_font_size=leafFont)
    if title_: plt.title(title_text)

def extract_clf_name(clf_tag):
    end_=None
    if clf_tag.find('_') != -1: end_=clf_tag.find('_')
    clf_name=clf_tag[:end_]
    return clf_name

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def select_initial_clusters(x,y):
    indexes = np.unique(y, return_index=True)[1]
    classes = [y[index] for index in sorted(indexes)]
    init_=np.array([x[y==c_][0] for c_ in classes])
    return init_

def get_significance(scores_hc,scores_fc,les_=None):
    le_=np.mean(scores_hc)/np.mean(scores_fc)
    if scores_fc==scores_hc:
        sig_=np.nan
    else:
        t_stat, p_val = stats.wilcoxon(scores_hc, scores_fc)
        if p_val<0.05:
            sig_=np.sign(le_-1)
        else:
            sig_=np.nan
    if les_ is not None: les_.append(le_)
    return le_,p_val,sig_

def get_score(y_true, y_pred=None, pred_proba=None, eval_metric='f1'):
    if eval_metric=='acc':
        score_ = accuracy_score(y_true, y_pred)
    elif eval_metric=='bac':        
        score_ = balanced_accuracy_score(y_true, y_pred)
    elif eval_metric=='f1':
        score_ = f1_score(y_true, y_pred, average='macro')
    elif eval_metric=='auc':
        score_ = roc_auc_score(y_true, pred_proba, average='weighted', multi_class='ovr')
    elif eval_metric=='nll':
        score_ = log_loss(y_true, pred_proba)
    return score_

def reduction_model(data, ndim=0, n_neighbor=5, redu_meth='lle', rseed=None, verbose=False):
    if data is tuple and len(data)==2:
        X, y = data
    else:
        X = data
    n_object, dim_prior = X.shape     
        
    if n_neighbor==0:     
        n_neighbor = min(n_object-1, 4+int(n_object/136))
        if verbose: print(f'n_neighbor:{n_neighbor}')
    
    if redu_meth=='lle':
        try:
            model = lle(n_neighbors=n_neighbor, 
                        n_components=ndim, 
                        random_state=rseed,
                        n_jobs=-1)
            X_reduced = model.fit_transform(X)
        except:
            model = lle(n_neighbors=n_neighbor, 
                        n_components=ndim, 
                        random_state=rseed,
                        eigen_solver='dense',
                        n_jobs=-1)
            X_reduced = model.fit_transform(X)
                
    elif redu_meth=='isomap':
        try:            
            model = Isomap(n_neighbors=n_neighbor,
                           n_components=ndim,
                           n_jobs=-1)
            X_reduced = model.fit_transform(X)
        except:
            model = Isomap(n_neighbors=n_neighbor,
                           n_components=ndim,
                           eigen_solver='dense',
                           n_jobs=-1)
            X_reduced = model.fit_transform(X)
    
    elif redu_meth=='tsne':
        try:
            model = TSNE(n_components=ndim,
                         random_state=rseed,
                         n_jobs=-1)
            X_reduced = model.fit_transform(X)
        except:
            model = TSNE(n_components=ndim,
                         perplexity = data.shape[0]-5,
                         random_state=rseed,
                         n_jobs=-1)
            X_reduced = model.fit_transform(X)
        
    return X_reduced

"""
Automatically setting number of dimesions after reduction
ndim= 0: set using PCA
ndim=-1: set using LDA
ndim=-2: set using MCA
"""
def dim_reduction(data,**kwargs):
    if type(data) is tuple and len(data)==2:
        X, y = data        
    else:
        X = data
    n_sample, n_feat = X.shape
    
    if 'verbose' in kwargs.keys():
        verbose = kwargs['verbose']
    else:
        verbose = 0
        
    # Scaling Option
    if 'scale_' in kwargs.keys():
        if kwargs['scale_']=='min_max':
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        elif kwargs['scale_']=='std':
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
    
    if 'ndim' in kwargs.keys():
        ndim = kwargs['ndim']
    else:
        ndim = 0
    if ndim in [0,-1,-2]:        
        if ndim==-2:
            # Multiple correspondence analysis            
            mca_ = prince.MCA(n_components=n_feat)
            Xdf=pd.DataFrame(data=X)
            mca_.fit(Xdf)
            cum_sum_=mca_.cumulative_percentage_of_variance_
        else:    
            if ndim==-1: # supervised dim_reduce
                mdl=LinearDiscriminantAnalysis()
                mdl.fit(X,y)
            elif ndim==0:
                mdl=PCA() # unsupervised dim_reduce
                mdl.fit(X)
            sum_=100*mdl.explained_variance_ratio_
            cum_sum_=np.cumsum(sum_)
        if verbose: print(f'Cumulative percentage of variance:{cum_sum_}')
        # automatically choose ndim
        if 'cum_var_thresh' in kwargs.keys(): 
            var_thresh=kwargs['cum_var_thresh']
        else:
            var_thresh=95
        ndim=len(cum_sum_[cum_sum_<var_thresh])
        if verbose: print(f'{ndim} out of {n_feat} components selected.')
        if ndim==n_feat:
            if verbose==2: print('No need to reduce dimensionality.')
            return X
        # sum_=0
        # for i,c in enumerate(mdl.explained_variance_ratio_):
        #     sum_ += c
        #     if sum_>=0.95:
        #         ndim = i+1
        #         # print(ndim,'out of',X.shape[1],' components selected')
        #         break
               
    if 'model' in kwargs.keys():
        model = kwargs['model']
    else:
        model = 'pca'
    if model in ['lda','nca']:
        models = {'lda': LinearDiscriminantAnalysis(n_components=ndim),
                     'nca': NeighborhoodComponentsAnalysis(n_components=ndim)}
        models[model].fit(X,y)
        Xr = models[model].transform(X)        
    elif model=='pca':
        pca_ = PCA(n_components=ndim)
        Xr = pca_.fit_transform(X)    
    elif model == 'mca':
        mca_ = prince.MCA(n_components=ndim)
        Xdf=pd.DataFrame(data=X)
        Xr = np.array(mca_.fit_transform(Xdf))        
    else:
        # print('dim reduction with',model)
        nn = min(n_sample-1, 4+int(n_sample/136))        
        # print('nearest_neighbors:',nn)
        Xr = reduction_model(X, ndim=ndim, n_neighbor=nn, redu_meth=model)
        
    return Xr
