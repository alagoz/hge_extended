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
import openml
import aeon
from aeon.datasets import load_from_tsfile,load_classification

def loadOpenMLdata(dset_id=None,
                   dset_name=None,
                   clean_nan=False,
                   nan_percent_thres=10,
                   min_class_size=5,
                   verbose=1):

    if dset_id == None or dset_name == None:
        if dset_name == None and dset_id == None:
            raise ValueError('Either dataset name or id should be provided.')        
        if dset_id == None:            
            token_=dset_name
        if dset_name == None:
            token_=int(dset_id)
    else:
        token_=int(dset_id)
            
    t_load = time()    
    dset = openml.datasets.get_dataset(token_,
                                       download_data=False,
                                       error_if_multiple=True,
                                       download_qualities=False,
                                       download_features_meta_data=False)
    dset_name = dset.name
    if verbose: print(f'{dset_name} with id [{dset_id}] is being downloaded... ', end = '')
    X_df, y_df, _, _ = dset.get_data(dataset_format="dataframe", target=dset.default_target_attribute)
    # X_, ys_, _, _ = dset.get_data(dataset_format="to_numpy", target=dset.default_target_attribute)
       
    dur_load = time()-t_load
    if verbose: print(f'in {dur_load:.1f} secs')
            
    X_num_cols = X_df.select_dtypes(exclude=['category']).columns
    X_sym_cols = X_df.select_dtypes(include=['category']).columns
    n_feat_num=len(X_num_cols)
    n_feat_sym=len(X_sym_cols)
    if verbose==2: print(f'Feature numbers without cleaning: numeric:{n_feat_num}, symbolic:{n_feat_sym}')
    
    # remove few populated classes
    classes_ini, class_counts_ini = np.unique(y_df, return_counts=True)
    if verbose==2: print(f'Class counts without cleaning:{class_counts_ini}')
    classes_to_remove=classes_ini[class_counts_ini<min_class_size]
    for c_ in classes_to_remove:
        X_df.drop(y_df[y_df==c_].index, inplace=True)
        y_df.drop(y_df[y_df==c_].index, inplace=True)    
    
    if clean_nan:
        n_sample_ini, n_feat_ini = X_df.shape
        # drop columns (features) if they have nan percentage above threshold
        perc_=X_df.isna().sum()/n_feat_ini*100>nan_percent_thres
        X_df=X_df.drop(X_df.columns[perc_],axis=1)
        
        # drop rows (samples) if they include nan
        Xy_df = pd.concat([X_df,y_df],axis=1)
        Xy_df_dropped = Xy_df.dropna()
        
        X_df = Xy_df_dropped[Xy_df_dropped.columns[:-1]]
        y_df = Xy_df_dropped[Xy_df_dropped.columns[-1]]
        
        X_num_cols = X_df.select_dtypes(exclude=['category']).columns
        X_sym_cols = X_df.select_dtypes(include=['category']).columns
        n_feat_num=len(X_num_cols)
        n_feat_sym=len(X_sym_cols)
        if verbose==2: print(f'Feature numbers after cleaning nans: numeric:{n_feat_num}, symbolic:{n_feat_sym}')
    
    # encode categorical features
    for col_ in X_sym_cols:
        slice_=X_df[col_]
        col_encoded=slice_.cat.codes
        col_encoded[col_encoded==-1]=np.nan
        
        X_df_copy=X_df.copy()
        X_df_copy[col_]=col_encoded
        X_df=X_df_copy
    
    # separate numerical and categorical features
    X_num_df = X_df[X_num_cols]
    X_sym_df = X_df[X_sym_cols]
    X_num = np.array(X_num_df)
    X_sym = np.array(X_sym_df)
    
    # encode class labels
    le = LabelEncoder()
    y_ = pd.Series(le.fit_transform(y_df),name='class')    
    if verbose==2:
        classes, class_counts = np.unique(y_, return_counts=True)
        print(f'Class counts after cleaning:{class_counts}')
    
    return (X_num,X_sym), y_

def prep_data(dset_name=None,
              dset_id=None,
              repo='ucr',              
              exclude_categories=True,
              return_class_labels=False,
              orig_split=False,
              return_xy=False,
              sel_class=None,
              reorder=False,               
              stratify=True, 
              k=5, 
              kth=1,
              rs=0,
              shuffle_=False,
              verbose=False):
    
    if repo in ['ucr_local','ucr_drive']:        
        path_=repo[repo.find('_'):]
        print(path_)
        x_train, x_test, y_train, y_test, classes=loadTSdataOffline(dset_name,
                                                                    path_=path_,
                                                                    sel_class=sel_class,
                                                                    orig_split=orig_split,
                                                                    reorder=reorder,)
        dset = x_train, x_test, y_train, y_test
        X = np.r_[x_train, x_test]
        y = np.r_[y_train, y_test]
        
        if not orig_split:
            x_train_new, x_test_new, y_train_new, y_test_new=split_data(X,
                                                                        y,
                                                                        stratify=stratify,
                                                                        k=k,
                                                                        kth=kth,
                                                                        rs=rs,
                                                                        shuffle_=shuffle_)
            dset = x_train_new, x_test_new, y_train_new, y_test_new
            X = np.r_[x_train_new, x_test_new]
            y = np.r_[y_train_new, y_test_new]  
    
    elif repo=='ucr':
        if return_xy:
            X, y = load_classification(name=dset_name)
        else:
            x_train, y_train = load_classification(name=dset_name,split='train')
            x_test, y_test = load_classification(name=dset_name,split='test')
            dset = x_train, x_test, y_train, y_test
        
    elif repo=='uci_drive':
        path = '../../datasets/dsets_uci/'
        full_path =f'{path}{dset_name}.csv'
        X, y = loadUCIdataOffline(full_path)
        
    elif repo=='sklearn_drive':
        path = '../../datasets/dsets_sklearn/'
        full_path =f'{path}{dset_name}.pkl'
        X, y = loadSklearndataOffline(full_path)
        
    if return_xy:
        out_ = X, y
    else:
        if 'dset' not in vars():
            x_train_new, x_test_new, y_train_new, y_test_new=split_data(X,
                                                                        y,
                                                                        stratify=stratify,
                                                                        k=k,
                                                                        kth=kth,
                                                                        rs=rs,
                                                                        shuffle_=shuffle_)
            dset = x_train_new, x_test_new, y_train_new, y_test_new
        out_ = dset
    
    if return_class_labels:
        return out_, np.unique(y)
    else:
        return out_, []

def loadTSdataOffline(fname,
                      rdtype='numpy2d',
                      path_='local',
                      truncation=False,
                      sort_class=True,
                      sel_class=None,
                      reorder=False, # option of re-ordering in case of sel_class
                      sqz=False):
    if path_=='local':
        DATA_PATH = os.path.join(os.path.dirname(aeon.__file__), 'datasets/data')
    elif path_=='drive':
        DATA_PATH = '../../datasets/dsets_ucr_ts_uv'
    
    x_train, y_train = load_from_tsfile(os.path.join(DATA_PATH, f'{fname}/{fname}_TRAIN.ts'))
    x_test, y_test = load_from_tsfile(os.path.join(DATA_PATH, f'{fname}/{fname}_TEST.ts'))
    
    # try: # if all equal length and univariate
    #     x_train, y_train = load_from_tsfile(os.path.join(DATA_PATH, f'{fname}/{fname}_TRAIN.ts'), return_data_type=rdtype)
    #     x_test, y_test = load_from_tsfile(os.path.join(DATA_PATH, f'{fname}/{fname}_TEST.ts'), return_data_type=rdtype)
    # except:
    #     warnings.warn('Data type can not be read as numpy 2d')
    #     try:
    #         x_train, y_train = load_from_tsfile(os.path.join(DATA_PATH, f'{fname}/{fname}_TRAIN.ts'), return_data_type='numpy3d')
    #         x_test, y_test = load_from_tsfile(os.path.join(DATA_PATH, f'{fname}/{fname}_TEST.ts'), return_data_type='numpy3d')
    #     except:
    #         warnings.warn('Data type can not be read as numpy 3d')
    #         x_train, y_train = load_from_tsfile(os.path.join(DATA_PATH, f'{fname}/{fname}_TRAIN.ts'), return_data_type='nested_univ')
    #         x_test, y_test = load_from_tsfile(os.path.join(DATA_PATH, f'{fname}/{fname}_TEST.ts'), return_data_type='nested_univ')
            
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    # Check for varying length 
    if type(x_train)==pd.core.frame.DataFrame:
        # x_all = pd.concat(x_train,x_test)
        len_list_train=[]
        for i in range(len(x_train)):
            len_list_train.append(len(x_train['dim_0'][i]))
        len_list_test=[]
        for i in range(len(x_test)):
            len_list_test.append(len(x_test['dim_0'][i]))
        train_fixed= np.all(np.array(len_list_train)==len_list_train[0])
        test_fixed = np.all(np.array(len_list_test)==len_list_test[0])
        if not (train_fixed and test_fixed):
            warnings.warn('Data samples have varying length!')
            if truncation:
                x_train= truncate_and_drop(x_train) 
                x_test = truncate_and_drop(x_test)
                classes= np.unique(y_train)
    
    if sqz:
        x_train= np.squeeze(x_train,axis=1)
        x_test = np.squeeze(x_test,axis=1)
        
    # Standard order of class ids: 0,1,..,N-1
    if sort_class:
        y_train, y_test = sort_class_id(y_train, y_test)
        
    if sel_class is not None:
        y_train, x_train = select_classes(sel_class, y_train, x_train, reorder=reorder)
        y_test, x_test = select_classes(sel_class, y_test, x_test, reorder=reorder)      
        
    classes = class_labels_sanity_check(y_train,y_test)
    return x_train, x_test, y_train, y_test, classes

def loadUCIdataOffline(full_path,return_df=False):
    df = pd.read_csv(full_path)
    vals = df.values
    X, y = vals[:, :-1], vals[:, -1]
    y = LabelEncoder().fit_transform(y)
        
    if return_df:
        return df, X, y
    else:
        return X, y
    
def loadSklearndataOffline(full_path,return_df=False):
    with open(full_path, 'rb') as bunch:
        df = pickle.load(bunch)
    X = df.data
    y = df.target
    y = LabelEncoder().fit_transform(y)
    if return_df:
        return df, X, y
    else:
        return X, y

def sort_class_id(y_train,y_test):
    classes=np.unique(y_train)
    Nclass = len(classes)
    # Standard order of class ids: 0,1,..,N-1
    if np.any(classes!=np.arange(Nclass)):
        for i in range(Nclass):
            y_train[y_train==classes[i]]=i
            y_test[y_test==classes[i]]=i                
        # classes=np.unique(y_train)
    return y_train, y_test

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

def get_indices_for_selected_groups(sel_class, y_train, y_test):
    # example:sel_class=[0,[10,13],8]
    sel_inds_train=np.array([],dtype=int)
    sel_inds_test=np.array([],dtype=int)
    
    y_train_new = y_train.copy()
    y_test_new = y_test.copy()       
    for i,g in enumerate(sel_class):            
        # find where indices for selected memberships occur
        if (type(g) is list or type(g) is np.ndarray) and len(g)>1:
            for j in g:
                loc_tr = np.where(y_train==j)[0].astype(int)
                loc_te = np.where(y_test==j)[0].astype(int)
                sel_inds_train= np.r_[sel_inds_train,loc_tr]
                sel_inds_test = np.r_[sel_inds_test,loc_te]
                
                y_train_new[y_train==j]=i
                y_test_new[y_test==j]=i                
        else:
            loc_tr = np.where(y_train==g)[0].astype(int)
            loc_te = np.where(y_test==g)[0].astype(int)
            sel_inds_train= np.r_[sel_inds_train,loc_tr]
            sel_inds_test = np.r_[sel_inds_test,loc_te]
            
            y_train_new[y_train==g]=i
            y_test_new[y_test==g]=i
            
    y_train_new= y_train_new[sel_inds_train]
    y_test_new = y_test_new[sel_inds_test]
    return sel_inds_train, sel_inds_test, y_train_new, y_test_new

def get_indices_for_selected_groups_(sel_class, y_train, y_test):
    # example:sel_class=[0,[10,13],8]
    sel_inds_train=np.array([],dtype=int)
    sel_inds_test=np.array([],dtype=int)        
    for g in sel_class:            
        # find where indices for selected memberships occur
        if (type(g) is list or type(g) is np.ndarray) and len(g)>1:
            for j in g:
                loc_tr = np.where(y_train==j)[0].astype(int)
                loc_te = np.where(y_test==j)[0].astype(int)
                sel_inds_train= np.r_[sel_inds_train,loc_tr]
                sel_inds_test = np.r_[sel_inds_test,loc_te]
                
                # merge classes in the inner celss for nested lists
                # assign the first index to the all in the nested cell so that it will be merged
                if j!=g[0]:
                    y_train[y_train==j]=g[0] 
                    y_test[y_test==j]=g[0]
        else:
            loc_tr = np.where(y_train==g)[0].astype(int)
            loc_te = np.where(y_test==g)[0].astype(int)
            sel_inds_train= np.r_[sel_inds_train,loc_tr]
            sel_inds_test = np.r_[sel_inds_test,loc_te]   
    return sel_inds_train, sel_inds_test

def truncate_and_drop(x,qntl=0.3):
    n_sample = len(x)
    lens=[]
    for i in range(n_sample):
        lens.append(len(x['dim_0'][i]))
    
    cutx = int(np.quantile(lens,qntl))
    
    done = 0          
    for i in range(n_sample):        
        if done==0 and (len(x['dim_0'][i])>cutx):
            x_truncated = np.array(x['dim_0'][i])[:cutx].reshape(1,-1)
            done = 1
            
        elif done==1 and (len(x['dim_0'][i])>cutx):
            x_truncated = np.r_[x_truncated, np.array(x['dim_0'][i])[:cutx].reshape(1,-1)]
    return x_truncated

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


    
# def nested(dset):
#     accfinal=0
#     cv_o = createCV(dset)
#     for tr_o,te_o in cv_o.lst:
#         accmax=0
#         for theta in createGrid():
#             acc=0
#             cv_i = createCV()
#             for tr_i,te_i in cv_i.lst:
#                 model_i=hdcss_(tr_i,theta)
#                 acc=acc+accuracy(model_i,te_i)
#             if acc > accmax:
#                 accmax = acc
#                 thetamax = theta
#         model_o = classtrain(tr_o,thetamax)
#         accfinal = accfinal+accuracy(model_out,te.o)
#     return accfinal/(len(cv.lst))

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

def get_precomp_diss_(dset_name,
                      diss_type='pcd',
                      clf_name ='svc',
                      cv =0,
                      rm ='pca',                     
                      i_test=0,
                      i_val =None):
    
    path_results = 'results/all_set/'
    fname = '_precomputed_pcd_ccd'
    df=pd.read_csv('%s%s.csv'%(path_results,fname), index_col=[0])
    
    ind_dset=df.index[df['dset_name']==dset_name][0]
    
    if diss_type=='pcd':
        if i_val is None:
            D = eval(df.at[ind_dset,'pcd_%s'%(clf_name)])[cv][i_test][0]
        else:
            D = eval(df.at[ind_dset,'pcd_%s'%(clf_name)])[cv][i_test][1][i_val]
        return D
    
    elif diss_type=='ccd':
        convert_rm={'pca':0,'lle':1,'isomap':2,'tsne':3}
        i_rm = convert_rm[rm]
        if i_val is None:
            X = eval(df.at[ind_dset,'ccd'])[i_rm][i_test][0]
        else:
            X = eval(df.at[ind_dset,'ccd'])[i_rm][i_test][1][i_val]
        return X

def get_precomp_diss(dset_name,
                     df = None,
                     diss_type='pcd',
                     clf_name ='svc',
                     rm ='pca',                     
                     i_test=0,
                     i_val =None):    
    if df is None:
        path_results = 'results/all_set/'
        fname = '_precomputed_diss_mats'
        df=pd.read_hdf(f'{path_results+fname}.h5','df')
    
    ind_dset=df.index[df['dset_name']==dset_name][0]
            
    if diss_type=='pcd':
        if i_val is None:
            y = df.at[ind_dset,f'pcd_{clf_name}'][i_test][0]
        else:
            y = df.at[ind_dset,f'pcd_{clf_name}'][i_test][1][i_val]        
    
    elif diss_type=='ccd':
        # convert_rm={'pca':0,'lle':1,'isomap':2,'tsne':3}
        # i_rm = convert_rm[rm]
        if i_val is None:
            y = df.at[ind_dset,f'ccd_{rm}'][i_test][0]
        else:
            y = df.at[ind_dset,f'ccd_{rm}'][i_test][1][i_val]
    return y
        
def get_precomp_flat_scores(dset_name,
                            clf_name,
                            df_= None,
                            i_test=None,
                            i_val=None,
                            cv_type='test',):
    if df_ is None:
        path_results = 'results/all_set/'
        fname = '_precomputed_flat_scores'
        df_=pd.read_hdf(f'{path_results+fname}.h5','df')
      
    ind_dset=df_.index[df_['dset_name']==dset_name][0]
    
    if cv_type=='test':
        score_ = df_.at[ind_dset,f'scores_{clf_name}'][0][0].mean()
    
    elif cv_type=='val':
        score_ = df_.at[ind_dset,f'scores_{clf_name}'][1][1][i_test]
    
    elif cv_type=='all':
        scores_te,scores_va = df_.at[ind_dset,f'scores_{clf_name}'][1]
        return scores_te,scores_va
        
    else:
        if i_val is None:
            score_ = df_.at[ind_dset,f'scores_{clf_name}'][0][i_test][0]
        else:
            score_ = df_.at[ind_dset,f'scores_{clf_name}'][0][i_test][1][i_val]
            
    return score_
    
def get_precomp_flat_scores_(dset_name,
                            clf_name,
                            i_test=None,
                            i_val=None,
                            cv_type='test',):
    
    path_results = 'results/all_set/'
    fname = '_precomputed_flat_scores_durs_svc_rocket'
    df_=pd.read_csv(f'{path_results+fname}.csv', index_col=[0])
    
    ind_dset=df_.index[df_['dset_name']==dset_name][0]
    
    if cv_type=='test':
        score_ = eval(df_.at[ind_dset,f'scores_{clf_name}'])[2][0]
    
    elif cv_type=='val':
        score_ = eval(df_.at[ind_dset,f'scores_{clf_name}'])[2][1][i_test]
    
    elif cv_type=='all':
        score_ = eval(df_.at[ind_dset,f'scores_{clf_name}'])[2]
        
    else:
        if i_val is None:
            score_ = eval(df_.at[ind_dset,f'scores_{clf_name}'])[0][i_test][0]
        else:
            score_ = eval(df_.at[ind_dset,f'scores_{clf_name}'])[0][i_test][1][i_val]
            
    return score_

def get_precomp_hc_scores(dset_name,
                          ind_z,
                          clf_name,
                          diss_type,
                          i_test=None,
                          i_val=None,
                          cv_type='test',):
    
    path_results = 'results/all_set/'
    fname = f'_all_trees_{clf_name}_{diss_type}'
    df_=pd.read_hdf(f'{path_results+fname}.h5', 'df')
    
    # df_ = pd.read_hdf('results/not_ready/_all_trees_pcd_svc-[0-32].h5', 'df')
    
    df_i = df_[df_['dset_name']==dset_name]
    
    scores_all_hc_i = df_i.at[ind_z,f'scores_{clf_name}']
          
    if cv_type in ['test','val','all']:
        scores_out = []
        scores_in = []
        for i in range(len(scores_all_hc_i)):
            scores_out.append(scores_all_hc_i[i][0])
            scores_in.append(np.round(np.array(scores_all_hc_i[i][1]).mean(),4))        
        scores_ = (np.round(np.array(scores_out).mean(),4),scores_in)
        
        if cv_type=='test':
            score_ = scores_[0]
        elif cv_type=='val':
            score_ = scores_[1][i_test]
        elif cv_type=='all':
            score_ = scores_
            
    else:
        if i_val is None:
            score_ = scores_all_hc_i[i_test][0]
        else:
            score_ = scores_all_hc_i[i_test][1][i_val]            
                
    return score_
    
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

def display_pwc_dissimilarity_matrix_stats(D):
    # Pairwise classification stats
    acc_avg = D[D!=0].mean()
    acc_max = D[D!=0].max()
    acc_min = D[D!=0].min()
    clust_max = np.unravel_index(D.argmax(),D.shape)
    D2=D.copy()
    D2[D2==0]=1
    clust_min = np.unravel_index(D2.argmin(),D.shape)    
    print("Mean accuracy:",acc_avg)
    print("Max accuracy:",acc_max, "between class", clust_max[0],"and class",clust_max[1])
    print("Min accuracy:",acc_min, "between class", clust_min[0],"and class",clust_min[1])

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
    
if __name__=='__main__':
    
    dset_id=2
    dset_name=None
    clean_nan=False
    nan_percent_thres=10
    min_class_size=5
    verbose=2

    if dset_id == None or dset_name == None:
        if dset_name == None and dset_id == None:
            raise ValueError('Either dataset name or id should be provided.')        
        if dset_id == None:            
            token_=dset_name
        if dset_name == None:
            token_=int(dset_id)
    else:
        token_=int(dset_id)
            
    t_load = time()    
    dset = openml.datasets.get_dataset(token_,
                                        download_data=False,
                                        error_if_multiple=True,
                                        download_qualities=False,
                                        download_features_meta_data=False)
    dset_name = dset.name
    if verbose: print(f'{dset_name} with id [{dset_id}] is being downloaded... ', end = '')
    X_df, y_df, _, _ = dset.get_data(dataset_format="dataframe", target=dset.default_target_attribute)
    # X_, ys_, _, _ = dset.get_data(dataset_format="to_numpy", target=dset.default_target_attribute)
       
    dur_load = time()-t_load
    if verbose: print(f'in {dur_load:.1f} secs')
            
    X_num_cols = X_df.select_dtypes(exclude=['category']).columns
    X_sym_cols = X_df.select_dtypes(include=['category']).columns
    n_feat_num=len(X_num_cols)
    n_feat_sym=len(X_sym_cols)
    if verbose==2: print(f'Feature numbers without cleaning: numeric:{n_feat_num}, symbolic:{n_feat_sym}')
    
    # remove few populated classes
    classes_ini, class_counts_ini = np.unique(y_df, return_counts=True)
    if verbose==2: print(f'Class counts without cleaning:{class_counts_ini}')
    classes_to_remove=classes_ini[class_counts_ini<min_class_size]
    for c_ in classes_to_remove:
        X_df.drop(y_df[y_df==c_].index, inplace=True)
        y_df.drop(y_df[y_df==c_].index, inplace=True)    
    
    if clean_nan:
        n_sample_ini, n_feat_ini = X_df.shape
        # drop columns (features) if they have nan percentage above threshold
        perc_=X_df.isna().sum()/n_feat_ini*100>nan_percent_thres
        X_df=X_df.drop(X_df.columns[perc_],axis=1)
        
        # drop rows (samples) if they include nan
        Xy_df = pd.concat([X_df,y_df],axis=1)
        Xy_df_dropped = Xy_df.dropna()
        
        X_df = Xy_df_dropped[Xy_df_dropped.columns[:-1]]
        y_df = Xy_df_dropped[Xy_df_dropped.columns[-1]]
        
        X_num_cols = X_df.select_dtypes(exclude=['category']).columns
        X_sym_cols = X_df.select_dtypes(include=['category']).columns
        n_feat_num=len(X_num_cols)
        n_feat_sym=len(X_sym_cols)
        if verbose==2: print(f'Feature numbers after cleaning nans: numeric:{n_feat_num}, symbolic:{n_feat_sym}')
    
    # # encode categorical features
    # for col_ in X_sym_cols:
    #     slice_=X_df[col_]
    #     col_encoded=slice_.cat.codes
        
    #     X_df_copy=X_df.copy()
    #     X_df_copy[col_]=col_encoded
    #     X_df=X_df_copy
    
    # # separate numerical and categorical features
    # X_num_df = X_df[X_num_cols]
    # X_sym_df = X_df[X_sym_cols]
    # X_num = np.array(X_num_df)
    # X_sym = np.array(X_sym_df)
    
#     # encode class labels
#     le = LabelEncoder()
#     y_ = pd.Series(le.fit_transform(y_df),name='class')    
#     if verbose==2:
#         classes, class_counts = np.unique(y_, return_counts=True)
#         print(f'Class counts after cleaning:{class_counts}')
    
#     # kk={'model':'mca','verbose':1}
#     # X_sym_df = dim_reduction(X_sym_df,**kk)
#     # Xr = dim_reduction(X_sym_df,model='mca',verbose=1)
       
#     X = np.c_[X_num,X_sym]
#     mca = prince.MCA(n_components=n_feat_sym, random_state=42)
#     mca_result = mca.fit(X_num_df)
#     print(f'eigenvalues_:{mca_result.eigenvalues_}, percentage_of_variance_:{mca_result.percentage_of_variance_}')
    
#     import copy
#     from clf_repo import clf_dict
#     from sklearn.model_selection import train_test_split
#     clf_name='xgb'
#     clf = copy.deepcopy(clf_dict[clf_name])
    
#     x_tr, x_te, y_tr, y_te = train_test_split(X, y_, test_size=0.2, random_state=1, stratify=y_)
#     clf.fit(x_tr,y_tr)
#     y_pred=clf.predict(x_te)
#     print(get_score(y_te,y_pred=y_pred))    
    

