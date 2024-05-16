import numpy as np
from time import time
from sklearn.metrics import DistanceMetric
from scipy.linalg import fractional_matrix_power as mp
from scipy.spatial import distance
from itertools import combinations
from utils import dim_reduction

def get_cc(X,y):   
    classes = np.unique(y)
    # class conditional means        
    cc=np.array([X[y == i].mean(axis=0) for i in classes])    
    return cc

def jensen_shannon_dist(x1,x2):
    pdf1, bin_edges = np.histogram(x1, bins=50, density=True)
    pdfn1 = pdf1/pdf1.sum()
    
    pdf2, bin_edges = np.histogram(x2, bins=50, density=True)
    pdfn2 = pdf2/pdf2.sum()
    
    return distance.jensenshannon(pdfn1, pdfn2, 2)

def get_jsd(X,y,verbose=False):
    classes = np.unique(y)
    data_by_class = [X[y==c] for c in classes]
    n_classes=len(classes)
    
    # Get all combinations of 2-classes
    dist_mat = np.zeros((n_classes,n_classes))
    t0 = time()
    
    comb=combinations(range(n_classes),2)
    for i,j in comb:       
        x_i=data_by_class[i]
        x_j=data_by_class[j]
        
        dist_=jensen_shannon_dist(x_i,x_j)             
                    
        dist_mat[i,j] = np.round(dist_/100,4)
        dist_mat[j,i] = dist_mat[i,j]
        
    dur_=time()-t0    
    if verbose: print(f'time for distance matrix generation {dur_:.2f} secs')
    return dist_mat

def preprocess_diss_mat(D,scale_=True):
    D[np.isnan(D)]=np.nanmax(D)
    m=D.shape[0]
    inds_i=[i for i in range(m) for j in range(m) if i!=j]
    inds_j=[j for i in range(m) for j in range(m) if i!=j]
    if len(D[inds_i,inds_j][D[inds_i,inds_j]==0])>0:
        D[inds_i,inds_j]=D[inds_i,inds_j]+(D[inds_i,inds_j][D[inds_i,inds_j]!=0]).min()
    if scale_: D = (D - np.min(D)) / (np.max(D) - np.min(D))
    return D

def decomp(M,ndim,methd='eig'):
    m, n = M.shape  
    #SVD
    hf=False # check if Hermitian
    if np.allclose(M,np.conj(M.T)): hf=True
    fm=True # Fulll matrices?
    u,s,vh=np.linalg.svd(M,full_matrices=fm,hermitian=hf)
    
    if methd=='eig':     
        #Eigen decomposition
        s, U = np.linalg.eig(M)
        #M2=U@S@U.T #Check reconstruction 
        X=U@np.diag(s)[:,:ndim]
    
    elif methd=='svd':      
        X = M @ vh[:ndim,:].T
    
    elif methd in ['lle','isomap','pca']:
        if m<5:
            n_neighbor = m-1
        elif m<10:
            n_neighbor = 5
        else:
            n_neighbor = 7
            
        X = dim_reduction(M,
                          ndim=ndim,
                          n_neighbor=n_neighbor,      
                          redu_meth=methd)    
    return X,s

def renorm(X):
    #Renormalizing each of X's rows to have unit length
    Y=X.copy()
    m=X.shape[0]
    rr=np.sqrt(np.sum(X**2,axis=1))
    for i in range(m):
        Y[i,:] = Y[i,:]/rr[i]
    return Y 

def spectral_embedding(dist,dim,sigma=0.5,nrm='symm_div',dmeth='lle',renrm=True):
    #Normalized spectral clustering from Ng (2002)
    n = dist.shape[0]
    
    #Affinity matrix A with free scale parameter sigma
    A = np.exp(-dist/(2*sigma**2))
    
    #Diagonal matrix D by lumping A    
    D = np.diag(A.sum(axis=1))
    
    #Normalization
    if nrm=='symm_div': #Symmetric Divisive
        L = mp(D,-.5) @ A @ mp(D,-.5)
    elif nrm=='div': #Divisive
        L = mp(D,-1) @ A
    elif nrm=='maxrow': #Maximum rowsum
        L = (A+D.max()*np.eye(n)-D)/D.max()
    
    X,Lambda = decomp(L,dim,methd=dmeth)
    if renrm:X=renorm(X)
    return X, Lambda 

def get_diss(X,
             y,
             pred_proba =None,
             conf_mat   =None,
             y_pred     =None,
             diss_type  ='cem', 
             out_type   ='diss_mat',             
             dt_metric  ='euclidean',                         
             redu_kwargs={'ndim':-1,                          
                          'model':'lda',},
             verbose    =False):
          
    if diss_type in ['ccm','cem','cce']:
        cc_ = get_cc(X,y,redu_kwargs=redu_kwargs)
        if out_type=='diss_mat':
            distf = DistanceMetric.get_metric(dt_metric)
            mat_ = distf.pairwise(cc_)            
            return mat_
        else:
            return cc_
    elif diss_type=='jsd':
        mat_ini = get_jsd(X,y)      
        mat_ = preprocess_diss_mat(mat_ini)
        if out_type=='obs_vec':
            # perform diss mat embedding
            vec_,_ = spectral_embedding(mat_,dim=0,sigma=0.85,dmeth='pca')
            return vec_
        else:
            return mat_
   
# if __name__=='__main__':
#     from utils import prep_data, plot_dendogram   
   