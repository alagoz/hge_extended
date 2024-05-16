import numpy as np
from sklearn.metrics import (DistanceMetric,
                             accuracy_score,
                             f1_score)
from diss_mat_embedding import spectral_embedding
from sklearn.cluster import (MeanShift,
                             AffinityPropagation,
                             KMeans,
                             DBSCAN,)
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from utils import select_classes
from clf_repo import clf_dict
from hier_struct import hier_node
from time import time
import warnings
import copy

n_clust=2
rs = None
_clust_dict ={
    'kmed'  : KMedoids(n_clusters=n_clust, metric='euclidean', random_state=rs),
    'affpro': AffinityPropagation(random_state=rs),
    'mshift': MeanShift(bandwidth=2),
    'kmeans': KMeans(n_clusters=n_clust, init='k-means++', n_init='auto', random_state=rs),
    'dbscan': DBSCAN(eps=0.3),
    'gmm': GaussianMixture(n_components=n_clust, reg_covar=1e-6, random_state=rs),    
    }
_dist_metric_dict={0:'euclidean',
                   1:'manhattan',
                   2:'chebyshev',
                   3:'minkowski',
                   4:'wminkowski',
                   5:'seuclidean',
                   6:'mahalanobis'}

class hdc():
    def __init__(self,
                 y, # dissimilarity matrix, or observation matrix, or a dataset
                 input_type=None, # pairwise dissimmilarity matrix or observation vector
                 
                 split_fun ='kmeans',
                 dt_metric ='euclidean',
                 spec_meth ='isomap',
                 sigma     =0.85,
                 dist_meth ='complete',
                 ):
        
        """Divisive hierarchical clustering makes top-down clustering"""
        # Handle input specifics        
        if y.shape[0] == y.shape[1] and np.allclose(np.diag(y), 0) and \
            np.all(y >= 0) and np.allclose(y, y.T):
            self.n_object=y.shape[0]
            if input_type in [None,'diss_mat']:                
                self.data = y
                self.input_type='diss_mat'                                
            elif input_type=='obs_vec':
                vec_,_ = spectral_embedding(y,dim=3,sigma=sigma,dmeth=spec_meth)
                
                self.data = vec_
                self.input_type= input_type
                self.spec_meth = spec_meth
                self.sigma = sigma                
            else:
                raise ValueError(f'Although {input_type} specified as input, a dissimilarity matrix is provided.') 
                
        elif y.ndim == 2:
            self.n_object, self.n_attribute = y.shape            
            if input_type in [None,'obs_vec']:
                self.data = y
                self.input_type='obs_vec'                               
            elif input_type=='diss_mat':
                distf = DistanceMetric.get_metric(dt_metric)
                mat_= distf.pairwise(y)
                                
                self.data = mat_
                self.input_type=input_type
                self.dt_metric = dt_metric
            else:
                raise ValueError(f'Although {input_type} specified as input, an observation vector is provided.')
                           
        if split_fun in [*_clust_dict.keys(),'fclust_hybrid']:
            self.clust_f = copy.deepcopy(_clust_dict[split_fun])
            if self.input_type=='diss_mat':
                try:
                    self.clust_f.set_params(metric='precomputed')
                except:
                    raise ValueError('Attempting to change metric parameter of the cluster function to precomputed is failed.')
            self.dt_metric = dt_metric
            self.dist_meth = dist_meth
        else:
            raise ValueError(f'split_fun={split_fun} is not recognized.')
                     
        self.split_fun = split_fun
        self.n_clusters = self.n_object-1 # number of non-leaf clusters to be formed
        self.Z = np.zeros((self.n_object-1,4))
        self.id_node = 2*self.n_object-2 # count down from all nodes
        self.id_next = 2*self.n_object-2
        self.id_clust = self.n_object-1  # count down from all parent (non-singleton) clusters
        self.clusters={self.id_node: list(range(self.n_object))} # non-singleton clusters
        self.is_fitted = False
        
    def preprocess_diss_mat(self,D,scale_=True):
        m=D.shape[0]
        inds_i=[i for i in range(m) for j in range(m) if i!=j]
        inds_j=[j for i in range(m) for j in range(m) if i!=j]
        if len(D[inds_i,inds_j][D[inds_i,inds_j]==0])>0:
            D[inds_i,inds_j]=D[inds_i,inds_j]+(D[inds_i,inds_j][D[inds_i,inds_j]!=0]).min()
        if scale_: D = (D - np.min(D)) / (np.max(D) - np.min(D))
        return D
        
    def split_cluster(self,verbose=False):
        # find the elements with maximum dissimilarity
        parent_ = self.clusters[self.id_next].copy()
        n_elem = len(parent_)
        if self.split_fun in [*_clust_dict.keys()]:            
            if self.split_fun=='fclust_hybrid':
                if n_elem>5:
                    self.clust_f = copy.deepcopy(_clust_dict['kmeans'])
                else:
                    self.clust_f = copy.deepcopy(_clust_dict['kmed'])
                
            if self.input_type=='diss_mat' and self.clust_f.metric=='precomputed':
                dpoints = self.data[np.ix_(parent_,parent_)]                
            else:
                dpoints = self.data[parent_,:]                
                
            try:
                y_pred=self.clust_f.fit_predict(dpoints)
            except:
                self.clust_f.fit(dpoints)
                y_pred = self.clust_f.labels_
            
            child_0 = [parent_[i] for i in range(len(parent_)) if y_pred[i]==0]
            child_1 = [parent_[i] for i in range(len(parent_)) if y_pred[i]==1]
                        
            if len(child_0)<1:
                child_0.append(child_1[-1])
                child_1.remove(child_1[-1])
            
            if len(child_1)<1:
                child_1.append(child_0[-1])
                child_0.remove(child_0[-1])        
        
        if 'height' not in vars(): height = None
        return child_0,child_1,height
    
    def reset_model(self):
        self.Z = np.zeros((self.n_object-1,4))
        self.id_node = 2*self.n_object-2 # count down from all nodes
        self.id_next = 2*self.n_object-2
        self.id_clust = self.n_object-1  # count down from all non-singleton clusters
        self.clusters={self.id_node: list(range(self.n_object))} # non-singleton clusters        
        self.is_fitted = False
        return
        
    def fit(self,
            fit_nodes=True,
            sort_=True,
            refit_=False):
        
        if self.is_fitted:
            if refit_:
                warnings.warn('Model already fitted. Model reset will occur for refit.')
                self.reset_model()
            else:
                raise ValueError('Model already fitted. Refit is not allowed.')
        
        if fit_nodes:
            nodes_=[hier_node() for i in range(self.n_clusters)]        
            for i in range(self.n_clusters):
                nodes_[i].set_id(i)
                nodes_[i].node_type='parent'
        
        # 0:left, 1:right
        Z_temp = self.Z        
        for id_n,id_clust in enumerate(range(self.n_clusters-1,-1,-1)):
            height=len(self.clusters[self.id_next])
            if height>2:
                child_0,child_1,_=self.split_cluster()
                if sort_:
                    child_0.sort()
                    child_1.sort()
            else:
                child_0 = [self.clusters[self.id_next][0]]
                child_1 = [self.clusters[self.id_next][1]]                                      
                        
            if len(child_0)>len(child_1):
                # is non-leaf
                if len(child_0)>1: 
                    self.id_node -= 1
                    child_0_id=self.id_node
                    self.clusters[child_0_id]=child_0
                # is leaf
                else: 
                    child_0_id=child_0[0]
                
                # is non-leaf
                if len(child_1)>1:
                    self.id_node -= 1
                    child_1_id=self.id_node
                    self.clusters[child_1_id]=child_1
                # is leaf
                else:
                    # print(child_1)
                    child_1_id=child_1[0]
            else:
                # is non-leaf
                if len(child_1)>1:
                    self.id_node -= 1
                    child_1_id=self.id_node
                    self.clusters[child_1_id]=child_1
                # is leaf
                else:
                    # print(child_1)
                    child_1_id=child_1[0]
                    
                # is non-leaf
                if len(child_0)>1: 
                    self.id_node -= 1
                    child_0_id=self.id_node
                    self.clusters[child_0_id]=child_0
                # is leaf
                else: 
                    child_0_id=child_0[0]            
            
            if fit_nodes:
                nodes_[id_n].append_subsets(child_0)
                nodes_[id_n].append_subsets(child_1)
                
                nodes_[id_n].set_branch_ids([child_0_id,child_1_id])
                
                id_right = id_n+self.id_next-child_1_id
                id_left = id_n+self.id_next-child_0_id
                if id_right<self.n_clusters:
                    nodes_[id_n].add_right(nodes_[id_right])
                    nodes_[id_right].add_parent(nodes_[id_n])
                if id_left<self.n_clusters: 
                    nodes_[id_n].add_left(nodes_[id_left])
                    nodes_[id_left].add_parent(nodes_[id_n])
            
            Z_temp[id_clust,0]=child_0_id
            Z_temp[id_clust,1]=child_1_id
            Z_temp[id_clust,2]=height
            Z_temp[id_clust,3]=len(child_0)+len(child_1)            
            
            self.id_next -= 1
                    
        self.is_fitted=True
        self.Z = Z_temp
        if fit_nodes:
            return Z_temp, nodes_
        else:
            return Z_temp
           
    def dist_(self,v,*argv):
        """methods for calculating the distance between the clusters"""
        if len(argv)==1:
            u = argv[0]
            if self.dist_meth=='clfd':
                method = 'clfd'
            elif self.split_fun in [*_clust_dict.keys()]:
                method = 'centroid'
            else:
                method = 'average'
        elif len(argv)==2:
            s = argv[0]
            t = argv[1]
            u = s+t
            method = self.dist_meth
        
        if 'D' in vars(self):
            D = self.data
            D_uv = D[np.ix_(u,v)]
        
        if method=='clfd':
            d = self.run_clf_(sel_class=[u,v])
            
        elif method=='single':
            # Nearest Point Algorithm
            d = D_uv.min()
            
        elif method=='complete':
            # Farthest Point Algorithm
            d = D_uv.max()
            
        elif method=='average':
            # UPGMA: Unweighted Pair Group Method with Arithmetic Mean
            d = D_uv.sum()/(len(u)*len(v))
            
        elif method=='weighted':
            # WPGMA: Weighted Pair Group Method with Arithmetic Mean
            D_sv = D[np.ix_(s,v)]
            D_tv = D[np.ix_(t,v)]
            d_sv = D_sv.sum()/(len(s)*len(v))
            d_tv = D_tv.sum()/(len(t)*len(v))
            
            d = (d_sv + d_tv)/2
            
        elif method=='centroid':
            # UPGMC
            c_u = self.data[u,:].mean(axis=0)
            c_v = self.data[v,:].mean(axis=0)
            
            distf = DistanceMetric.get_metric(self.dt_metric)
            d = distf.pairwise(np.c_[c_u,c_v].T)[0,1]
            
        elif method=='median':
            # WPGMC
            c_s = self.data[s,:].mean(axis=0)
            c_t = self.data[t,:].mean(axis=0)
            c_u = (c_s+c_t)/2
            c_v = self.data[v,:].mean(axis=0)           
            
            distf = DistanceMetric.get_metric(self.dt_metric)
            d = distf.pairwise(np.c_[c_u,c_v].T)[0,1]
            
        elif method=='ward':
            # MISSQ: Minimum Increase of Sum of Squares
            T = len(s)+len(v)+len(t)
            sv = (len(s)+len(v))/T
            tv = (len(t)+len(v))/T
            st = (len(s)+len(t))/T
            D_sv = D[np.ix_(s,v)]
            D_tv = D[np.ix_(t,v)]
            D_st = D[np.ix_(s,t)]
            d_sv = D_sv.sum()/(len(s)*len(v))
            d_tv = D_tv.sum()/(len(t)*len(v))
            d_st = D_st.sum()/(len(s)*len(t))
            
            d = np.sqrt(sv*d_sv**2+tv*d_tv**2-st*d_st**2)
            
        elif method=='ward2':
            # MISSQ: Minimum Increase of Sum of Squares
            c_u = self.data[u,:].mean(axis=0)
            c_v = self.data[v,:].mean(axis=0)
            
            distf = DistanceMetric.get_metric(self.dt_metric)
            d = np.sqrt(2*len(u)*len(v)/(len(u)+len(v)))*distf.pairwise(np.c_[c_u,c_v].T)[0,1]
            
        else:
            raise ValueError("Not a valid distance method provided.")
        return d    
    
#%%        
if __name__=='__main__':
    from utils import prep_data, plot_dendogram
        
    dset_name='Glass'
    repo='uci'
    (X,y),labels = prep_data(dset_name=dset_name,
                             repo=repo,
                             return_class_labels=True,
                             orig_split=True,
                             return_xy=True,
                             sort_classes=True,
                             super_classes=None,
                             reorder=False,
                             stratify=True, 
                             k=5, 
                             kth=1,
                             rs=None,
                             shuffle_=False,
                             verbose=False)
    classes = np.unique(y)
    n_classes = len(classes)
    
    from diss_ import get_diss
    clf=copy.deepcopy(clf_dict[clf_name])
    D = get_diss(X,y,diss_type='cc',clf=clf)
    
    model = hdc(y=D,
                split_fun = 'diana',
                use_diss_mat = True,
                clust_fun = clust_fun)
    
    Z, nodes = model.fit()
    plot_dendogram(Z, class_list=df_y)   
