"""
Classes for implementation of Hierarchical Classification task.

hier_binary_tree: A class where 
- hierarchy is defined in binary tree structure
- HC is implemented using GC and LCPN, LCPN+, and LCPN+F schemes

hier_node: a class where nodes of binary tree are defined
"""

import numpy as np
import os
import warnings
import scipy.cluster.hierarchy as sch
from sklearn.metrics import (calinski_harabasz_score as CHI,
                             davies_bouldin_score as DBI,
                             silhouette_score as SS,)
import copy
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

class hier_node:
    """Hierarchical multi-label tree node"""
    def __init__(self, name='root', level=None):
        self.name = name
        self.id = None
        self.node_type = None # 'leaf' or 'non-leaf' in LCN, 'parent' in LCPN                
        self.branch_ids = None # 0:left, 1:right in LCPN, branch id is equal to node id in case of all nodes (LCN)
        self.branch_weights_sample = None
        self.branch_weights_depth = None
        self.child_ids = None  # 0:left, 1:right       
        self.level = level
        # flat labels are groupped to form clusters 
        # e.g. [[0,2],[1,3,4]]
        self.subsets = []
        self.subsets_lcn = []
        self.clust_height = None
        self.flat_labels = []
        #assign a classifier object here
        self.clf_style='hier'
        self.clf_base = None
        self.clf_score_expected = None
        self.clf_score_measured = None
        self.left = None
        self.right = None
        self.parent = None
        self.children = []
        self.inds_train = None
        self.inds_test = None        
        self.hc_type = 'lcpn' #'lcpn' or 'lcn'
        # self.leaves = leaves
        
    def add_left(self,other):
        if self.left is None:
            self.left=other
        else:
            warnings.warn("Left node already exists!")
    
    def add_right(self,other):
        if self.right is None:
            self.right=other
        else:
            warnings.warn("Right node already exists!")
            
    def add_parent(self,other):
        if self.parent is None:
            self.parent=other
            # self.level=other.level+1
        else:
            warnings.warn("Parent node already exists!")
    
    def set_id(self,x):
        if self.id is None:
            self.id = x
    
    def set_child_ids(self,x):
        self.child_ids=x
        
    def set_level(self,level_):
        self.level=level_
        
    def set_branch_ids(self,x):
        self.branch_ids=x
        
    def set_flat_labels(self):
        flat_labels=[]
        if len(self.subsets)>0:
            for g in self.subsets:
                if type(g) is list or type(g) is np.ndarray:
                    for c in g:
                        flat_labels.append(c)
                else:
                    flat_labels.append(g)
        else:
            raise ValueError('Subsets must be set first in order to get flat labels under node%2d'%(self.id))
        self.flat_labels=flat_labels
        return flat_labels
    
    def set_children(self,x):
        self.children=x
        
    def append_subsets(self,x):
        self.subsets.append(x)
        
    def set_clf_base(self,x,rseed=0):
        np.random.seed(seed=rseed)
        self.clf_base=copy.deepcopy(x)
        
    def set_inds_train(self,x):
        self.inds_train=x
        
    def set_inds_test(self,x):
        self.inds_test=x
        
    # Utility function to check inorder traversal     
    def inOrder(self, sol, first_='left'):
        if self is None:
            return
        if first_=='left':
            self.inOrder(self.left, sol)
            sol.append(self.subsets)
            self.inOrder(self.right, sol)
        elif first_=='right':
            self.inOrder(self.right, sol)
            sol.append(self.subsets)
            self.inOrder(self.left, sol)
            
    # Utility function to check preorder traversal
    def preOrder(self, sol, first_='left'):
        if self is None:
            return
        sol.append(self.subsets)
        if first_=='left':
            self.preOrder(self.left, sol)
            self.preOrder(self.right, sol)
        elif first_=='right':
            self.preOrder(self.right, sol)
            self.preOrder(self.left, sol)
        
    # Utility function to check postorder traversal
    def postOrder(self, sol=[], first_='left'):
        if self is None:
            return
        
        if first_=='left' and self.left is not None:
            if self.left is not None:
                self.left.postOrder(sol)
            else:
                return
            
            if self.right is not None:
                self.right.postOrder(sol)
            else:
                return     
            
        elif first_=='right':
            if self.right is not None:
                self.right.postOrder(sol)
            else:
                return
            
            if self.left is not None:
                self.left.postOrder(sol)
            else:
                return
               
        sol.append(self.subsets)
        
    # Utility function to fetch parent node ids
    def traverseParentNodes(self,sol=[]):
        if self is None:
            return
        
        if self.parent is not None:
            # print(self.parent.id)
            sol.insert(0,self.parent.id)
            self.parent.traverseParentNodes(sol)
        else:
            return
    
    # Utility function to fetch branch ids in the path
    def traverseBranchPath(self,ids=[]):
        if self is None:
            return
        
        if self.parent is not None:
            # print(self.parent.branch_ids)
            ids.insert(0,self.parent.branch_ids[0])            
            self.parent.traverseBranchPath(ids=ids)
        else:
            return
    
    def traverseBranchWeights(self,weights_sample=[],weights_depth=[]):
        if self is None:
            return
        
        if self.parent is not None:
            # print(self.parent.id)            
            if self.parent.branch_weights_sample is not None:                
                weights_sample.insert(0,self.parent.branch_weights_sample)
            if self.parent.branch_weights_depth is not None:
                weights_depth.insert(0,self.parent.branch_weights_depth)
            self.parent.traverseBranchPath(weights_sample=weights_sample,weights_depth=weights_depth)
        else:
            return
        
    def isEqual(self, root2):
        eq_=0
        if self is not None and root2 is not None:        
            if self.subsets[0].sort()==root2.subsets[0].sort():
                # print(self.subsets[0])
                self.isEqual(self.left, root2.left)
                self.isEqual(self.right, root2.right)            
                eq_=1
            elif self.subsets[0].sort()==root2.subsets[1].sort():
                # print(self.subsets[0])
                self.isEqual(self.left, root2.right)
                self.isEqual(self.right, root2.left)            
                eq_=1
            else:
                return False
        if eq_==1:
            return True
    
    def __eq__(self, root2):
        eq_=0
        if self is not None and root2 is not None:        
            if self.subsets[0]==root2.subsets[0]:                
                self.__eq__(self.left, root2.left)
                self.__eq__(self.right, root2.right)            
                eq_=1
            elif self.subsets[0]==root2.subsets[1]:                
                self.__eq__(self.left, root2.right)
                self.__eq__(self.right, root2.left)            
                eq_=1
            else:
                return False
        if eq_==1:
            return True
        
    def select_subsamples(self,y,x=None):
        """
        Select from a given dataset accordingly with clustering at this node. 
           
        Data can be 1D (if y) or 2D (if x)
        Input: Dataset to be selected from (e.g x_train, y_train, x_test or y_test)
        Output: Selected dataset
        """
        sel_inds = np.array([],dtype=int)
        y_sel = y.copy()
        for i,g in enumerate(self.subsets):
            # find where indices for selected memberships occur
            if (type(g) is list or type(g) is np.ndarray) and len(g)>1:
                for j in g:
                    loc_ = np.where(y==j)[0].astype(int)
                    sel_inds = np.r_[sel_inds,loc_]
                    
                    y_sel[y==j]=i
            else:
                loc_ = np.where(y==g)[0].astype(int)
                sel_inds = np.r_[sel_inds,loc_]
                
                y_sel[y==g]=i
                
        y_sel = y_sel[sel_inds]
        if x is None:
            return y_sel
        else:        
            x_sel = x[sel_inds,:]
            return y_sel, x_sel

    def clf_fit(self,x_train,y_train,rseed=0):
        np.random.seed(seed=rseed)
        y_train, x_train = self.select_subsamples(y_train, x_train)
        self.clf_base.fit(x_train,y_train)
        
    def _clf_fit(self,x_train,y_train,usefulness=1,rseed=0):
        np.random.seed(seed=rseed)
        y_tr_sub, x_tr_sub = self.select_subsamples(y_train, x_train)
        # print(f'fitting node {self.id}')
        self.clf_base.fit(x_tr_sub,y_tr_sub)
        
        """
        Build usefulness array for each hierarchy unit. For details, see: Silla and 
        Freitas 2009, "A Global-Model Naive Bayes Approach to the Hierarchical 
        Prediction of Protein Functions") 
        """
        if usefulness:
            # estimate absolute (not normalized) usefulness
            self.usefulness_abs=[np.log2(len(y_tr_sub)),np.log2(len(y_tr_sub))]
        return self.clf_base
    
    def clf_predict(self,x_test,y_test):
        _, x_test = self.select_subsamples(y_test, x_test)
        y_pred = self.clf_base.predict(x_test)             
        self.y_pred = y_pred
        return y_pred
    
    def clf_score(self,y_test,y_pred=None,percent=True,eval_metric='acc'):
        if y_pred is None:
            if hasattr(self,'y_pred'):
                y_pred = self.y_pred                
            else:
                raise ValueError('Can not get score without y_pred. Make sure it exists.')
        else:
            if hasattr(self,'y_pred') and y_pred != self.y_pred:
                warnings.warn('Discrepancy between y_pred provided and the instance variable y_pred.')
        
        if eval_metric == 'acc':
            score = metrics.accuracy_score(self.y_test, y_pred)
        elif eval_metric == 'f1':
            score = metrics.f1_score(self.y_test, y_pred, average='macro')
            
        if percent:
            score *= 100
        
        return score
    
    def clf_predict_score(self,x_test,y_test,percent=True,eval_metric='acc'):
        """When labels for test data is not given"""                
        y_test, x_test = self.select_subsamples(y_test, x_test)        
        y_pred = self.clf_base.predict(x_test)
        
        if eval_metric == 'acc':
            score = metrics.accuracy_score(y_test, y_pred) 
        elif eval_metric == 'f1':
            score = metrics.f1_score(y_test, y_pred, average='macro')
                   
        if percent:
            score *= 100
        self.clf_score_expected = score
        return score
    
    def clf_fit_(self,train_data,rseed=0):
        """When labels for train data is not given""" 
        np.random.seed(seed=rseed)
        print(self.id)
        inds=self.inds_train
        inds_train = np.array([],dtype=int)
        y_train = np.array([],dtype=int)
        for key, val in inds.items():
            inds_train = np.r_[inds_train, val]
            y_train = np.r_[y_train, key*np.ones(len(val),dtype=int)]
        x_train = train_data[inds_train,:]
        # print('x_train.shape',x_train.shape,'y_train.shape',y_train.shape)
        # print('train - {}'.format(np.bincount(y_train)))
        self.clf_base.fit(x_train,y_train)
        
    def clf_predict_(self,test_data):
        """When labels for test data is not given"""
        self.y_pred = self.clf_base.predict(test_data)
        return self.y_pred
    
    def clf_predict_score_(self,test_data,percent=True,eval_metric='acc'):
        """When labels for test data is not given"""                
        inds=self.inds_test
        inds_test = np.array([],dtype=int)
        y_test = np.array([],dtype=int)
        for key, val in inds.items():
            inds_test = np.r_[inds_test, val]
            y_test = np.r_[y_test, key*np.ones(len(val),dtype=int)]
        x_test = test_data[inds_test,:]
        
        if eval_metric == 'acc':
            score = metrics.accuracy_score(y_test, self.clf_predict_(x_test))
        elif eval_metric == 'f1':    
            score = metrics.f1_score(y_test, self.clf_predict_(x_test), average='macro')            
            
        if percent:
            score *= 100
        self.clf_score_expected = score    
        return score        

class hier_level:
    def __init__(self, id_=None):        
        self.id = id_
        self.subsets = []
        self.branch_ids = []
        self.branch_weights_sample = None
        self.branch_weights_depth = None
        
    def append_subsets(self,x):
        if type(x[0]) is list:
            self.subsets=x
        else:
            self.subsets.append(x)
        
    def set_branch_ids(self,x):
        self.branch_ids=x
        
    def extend_branch_ids(self,x):
        if type(x) is list:
            self.branch_ids.extend(x)
        elif type(x) is int:
            self.branch_ids.extend([x])
        else:
            raise ValueError(f'Extension to the list is not possible with data type {type(x)}')
        
    def set_clf_base(self,x,rseed=0):
        np.random.seed(seed=rseed)
        self.clf_base=copy.deepcopy(x)
        
    def select_subsamples(self,y,x=None):
        """
        Select from a given dataset accordingly with clustering at this node. 
           
        Data can be 1D (if y) or 2D (if x)
        Input: Dataset to be selected from (e.g x_train, y_train, x_test or y_test)
        Output: Selected dataset
        """           
        sel_inds = np.array([],dtype=int)
        y_sel = y.copy()
        for i,g in enumerate(self.subsets):
            # find where indices for selected memberships occur
            if (type(g) is list or type(g) is np.ndarray) and len(g)>1:
                for j in g:
                    loc_ = np.where(y==j)[0].astype(int)
                    sel_inds = np.r_[sel_inds,loc_]
                    
                    y_sel[y==j]=i
            else:
                loc_ = np.where(y==g)[0].astype(int)
                sel_inds = np.r_[sel_inds,loc_]
                
                y_sel[y==g]=i
                
        y_sel = y_sel[sel_inds]
        if x is None:
            return y_sel
        else:        
            x_sel = x[sel_inds,:]
            return y_sel, x_sel

    def clf_fit(self,x_train,y_train,rseed=0):
        np.random.seed(seed=rseed)
        y_train, x_train = self.select_subsamples(y_train, x_train)
        self.clf_base.fit(x_train,y_train)
        
    def _clf_fit(self,x_train,y_train,usefulness=1,rseed=0):
        np.random.seed(seed=rseed)
        y_tr_sub, x_tr_sub = self.select_subsamples(y_train, x_train)
        # print(f'fitting node {self.id}')
        self.clf_base.fit(x_tr_sub,y_tr_sub)
        
        """
        Build usefulness array for each hierarchy unit. For details, see: Silla and 
        Freitas 2009, "A Global-Model Naive Bayes Approach to the Hierarchical 
        Prediction of Protein Functions") 
        """
        if usefulness:
            # estimate absolute (not normalized) usefulness
            self.usefullness_abs=np.log2(len(y_tr_sub))
        return self.clf_base        
    
class hier_binary_tree:
    "Hierarchical binary tree (nested dichotomies)"
    def __init__(self,
                 pnodes=None,
                 link_mat=None, #linkage matrix
                 n_class=None,
                 y_train=None,
                 y_test=None,
                 pred_proba=None,
                 clf_flat=None,
                 pred_proba_fc=None,
                 clf_glob=None,
                 pred_proba_gc=None,):
        self.y_train = y_train
        self.y_test = y_test
        self.Z = link_mat        
        self.clf_flat = clf_flat
        self.pred_proba_fc = pred_proba_fc
        self.clf_glob = clf_glob
        self.pred_proba_gc = pred_proba_gc
        self.hc_type = 'lcpn' # 'lcpn', 'lcpn+', 'lcpn+f', 'lcl+', or 'lcn+'
        self.branch_weight_proportion = 'direct' #or inverse
        
        if n_class is not None:
            pass
        elif link_mat is not None:
            n_class = link_mat.shape[0]+1
        elif y_train is not None:
            n_class = len(np.unique(y_train))
        else:
            raise ValueError('Number of flat labels has to be provided.')        
        
        self.flat_labels = np.arange(n_class).tolist()
        self.n_flat_labels = n_class
        self.max_depth = self.n_flat_labels-1
        self.n_nodes = 2*n_class-1 # total nodes (leaf and non-leaf)
        self.n_pnodes = n_class-1 # total parent nodes
        self.n_branches = 2*n_class-2
        self.branches_for_leaves = None
        self.parents_for_leaves = None
                
        self.nodes_assigned = False
        self.nodes = []
        
        self.lcn_nodes_assigned = False
        self.lcn_nodes = []
        
        if pnodes is None:
            self.pnodes_assigned = False
            self.pnodes = []
        else:
            self.pnodes = pnodes
            self.pnodes_assigned = True            
        
        self.levels_assigned = False
        self.levels = [] # list of level objects assigned for hierarchy
                        
    def set_clf_flat(self,x,rseed=0):
        np.random.seed(seed=rseed)
        self.clf_flat=copy.deepcopy(x)
    
    def set_clf_glob(self,x,rseed=0):
        np.random.seed(seed=rseed)
        self.clf_glob=copy.deepcopy(x)
        
    def set_linkage(self,z):
        self.Z = z
        
    def set_y(self,y_train=None,y_test=None):
        """set train and test labels"""
        if y_train is not None: self.y_train = y_train
        if y_test is not None: self.y_test = y_test
        
    def set_proba(self,x):
        self.pred_proba_fc = x   
    
    def set_parent_nodes(self,node_list):
        if self.pnodes_assigned:
            print('Parent nodes are already assigned.')
        self.pnodes_assigned = True
        self.pnodes = node_list
    
    def set_all_nodes(self,node_list):
        self.nodes_assigned = True
        self.nodes = node_list
    
    def assign_pnodes_from_linkage(self,verbose=False):
        """
        Utility function to create non-leaf nodes and their cluster properties 
        from linkage matrix.
        
        Non-leaf nodes correspond to non-singleton clusters whereas leaf nodes
        correspond to flat class labels
        """
        flat_labels = np.array(self.flat_labels)
        rootnode, nodelist = sch.to_tree(self.Z, rd=True)
        leaf_nodes = rootnode.pre_order()
        # ordered_leaves = np.array(leaf_nodes)
        n = self.Z.shape[0]
        n_leaf = len(leaf_nodes)
        n_node = 2*n-n_leaf
        node_num = 1
        node_list=[] #nonleaf node objects are stored in list 
        for i,j in enumerate(range(-1,-n-1,-1)):
            if i==0:
                root = hier_node(level=0)
                root.set_id(0)
                root.node_type='parent'
                node_list.append(root)
                n_node -= 1
                
                r_id = rootnode.right.id
                l_id = rootnode.left.id
                root.set_branch_ids([l_id,r_id])
                # rloc = np.where(Z==r_id)
                # ii,jj = rloc[0][0], rloc[1][0]
                
                # Go right
                if nodelist[r_id].is_leaf(): # r_id in flat_labels:
                    root.append_subsets(flat_labels[flat_labels!=r_id].tolist())
                    root.append_subsets([r_id])
                    leaf_nodes.remove(r_id)
                    n_leaf -= 1
                
                # Go left
                if nodelist[l_id].is_leaf(): #n_id in flat_labels:
                    root.append_subsets([l_id])
                    root.append_subsets(flat_labels[flat_labels!=l_id].tolist())
                    leaf_nodes.remove(l_id)
                    n_leaf -= 1
                
                if not (nodelist[r_id].is_leaf() or nodelist[l_id].is_leaf()):
                    T = sch.fcluster(self.Z, t=self.Z[j-1,2], criterion='distance')
                    L, M = sch.leaders(self.Z, T)
                    root.append_subsets(flat_labels[T==M[L==l_id]].tolist())
                    root.append_subsets(flat_labels[T==M[L==r_id]].tolist())
            else:
                node_list.append(hier_node(name='node%02d'%node_num))
                node_list[i].set_id(node_num)
                node_list[i].node_type='parent'
                n_node -= 1
                node_num += 1
                
                if i<n-1:
                    T = sch.fcluster(self.Z, t=self.Z[j-1,2], criterion='distance')
                else:
                    T = sch.fcluster(self.Z, t=self.Z[j,2]-0.1, criterion='distance')
                L, M = sch.leaders(self.Z, T)
                
                # 0:left, 1:right
                l_id, r_id = int(self.Z[j,0]), int(self.Z[j,1])
                node_list[i].set_branch_ids([l_id,r_id])
                # Check right and left (0:left, 1:right)
                if nodelist[r_id].is_leaf() and nodelist[l_id].is_leaf():
                    node_list[i].append_subsets([l_id])
                    node_list[i].append_subsets([r_id])
                    leaf_nodes.remove(l_id)
                    leaf_nodes.remove(r_id)
                    n_leaf -= 1
                # Check right only
                elif nodelist[r_id].is_leaf(): # r_id in flat_labels:
                    node_list[i].append_subsets(flat_labels[T==M[L==l_id]].tolist())
                    node_list[i].append_subsets([r_id])
                    leaf_nodes.remove(r_id)
                    n_leaf -= 1
                # Check left only
                elif nodelist[l_id].is_leaf(): # r_id in flat_labels:
                    node_list[i].append_subsets([l_id])
                    node_list[i].append_subsets(flat_labels[T==M[L==r_id]].tolist())
                    leaf_nodes.remove(l_id)
                    n_leaf -= 1
                # If none is a leaf
                else:# (nodelist[r_id].is_leaf() or nodelist[l_id].is_leaf()):
                    node_list[i].append_subsets(flat_labels[T==M[L==l_id]].tolist())
                    node_list[i].append_subsets(flat_labels[T==M[L==r_id]].tolist())
                
                # traverse back up to find the parent node 
                # and check if we are in the left or right of it
                for node_i in range(i-1,-1,-1):
                    lid, rid = int(self.Z[-node_i-1,0]), int(self.Z[-node_i-1,1])
                    # if right branch is in the right, and left branch is in 
                    # the left of previous node's right branch
                    prev_r = nodelist[rid]
                    prev_l = nodelist[lid]
                    if prev_r.right is not None or prev_r.left is not None:
                        if (prev_r.right.id==r_id and prev_r.left.id==l_id) or (prev_r.right.id==l_id and prev_r.left.id==r_id):
                            node_list[i].add_parent(node_list[node_i])
                            node_list[node_i].add_right(node_list[i])
                            if verbose:
                                print(node_list[i].name, node_list[i].branch_ids,'inserted in the right of',node_list[node_i].branch_ids)
                            break
                    if prev_l.right is not None or prev_l.left is not None:
                        if (prev_l.right.id==r_id and prev_l.left.id==l_id) or (prev_l.right.id==l_id and prev_l.left.id==r_id):
                            node_list[i].add_parent(node_list[node_i])
                            node_list[node_i].add_left(node_list[i])
                            if verbose:
                                print(node_list[i].name, node_list[i].branch_ids,'inserted in the left of',node_list[node_i].branch_ids)
                            break    
                    if node_i==0:
                        warnings.warn(node_list[i].branch_ids,'is not inserted anywhere!')
                        
        if len(leaf_nodes) != 0:
            # warnings.warn('There are leaves:',leaf_nodes,'not touching the ground!')
            warnings.warn('There are leaves not touching the ground!')
            print(leaf_nodes)
                            
        self.pnodes = node_list
        self.pnodes_assigned = True
        
        return node_list
                                    
    def assert_parent_nodes(self):
        if self.pnodes_assigned:
           PNs = self.pnodes 
        else:
            try:
                PNs = self.assign_pnodes_from_linkage()
                self.pnodes_assigned = True
            except:
                raise ValueError('Parent nodes for the hierarchy is not defined.')
        return PNs    
                
    def assign_units(self,hc_type='lcpn'):
        """Assign the unit on which the classifier will work."""
        units = self.assert_parent_nodes()
        if hc_type == 'lcpn+f':
            units_selected=[]
            for u_ in units:
                # train a parent node if it includes at least one non-leaf label
                for s in u_.subsets:
                    if len(s)>1:
                        units_selected.append(u_)
                        break
            units=units_selected            
        return units
    
    def find_path_for_leaves(self):
        # find pathway for each flat label
        path_branch = []
        path_parent = []
        PNs=self.assert_parent_nodes()
        for c_ in self.flat_labels:
            parents_c=[] # parents for class c
            branches_c=[] # branch ids within the path of class c
            for p_ in PNs:
                res_ = [(p_.id,p_.branch_ids[i]) for i in range(len(p_.subsets)) if c_ in p_.subsets[i]]
                if len(res_)>0:
                    parents_c.append(res_[0][0])
                    branches_c.append(res_[0][1])
            path_parent.append(parents_c)
            path_branch.append(branches_c)
        
        self.parents_for_leaves=path_parent
        self.branches_for_leaves=path_branch        
        return path_branch
    
    def assign_weights_for_branches(self, y_train=None, scale_='lin', verbose=False):
        if y_train is None: y_train = self.y_train
        classes, class_counts = np.unique(self.y_train, return_counts=True)
        
        units=self.assert_parent_nodes()
        
        for u_ in units:
            weights_sample=[class_counts[subset_].sum() for subset_ in u_.subsets]
            weights_depth=[len(subset_) for subset_ in u_.subsets]
            
            if scale_=='log':
                u_.branch_weights_sample=np.log2(weights_sample)
                u_.branch_weights_depth=np.log2(weights_depth)
            elif scale_=='lin':
                u_.branch_weights_sample=weights_sample
                u_.branch_weights_depth=weights_depth
            
            if verbose:
                w_samp=np.round(u_.branch_weights_sample,2)
                w_dept=np.round(u_.branch_weights_depth,2)
                print(f'node_id:{u_.id}, branch_ids:{u_.branch_ids}, branch_weights_sample:{w_samp}, branch_weights_depth:{w_dept}')                   
                
    # normalize branch weights within a leaf path
    def normalize_branch_weights(self,weights):
        if self.branch_weight_proportion=='direct':
            weights_norm = (np.array(weights)/np.array(weights).sum()).tolist()            
        else:
            weights_norm = ((1/np.array(weights))/(1/np.array(weights)).sum()).tolist()                       
        return weights_norm
        
    def assign_weight_path_for_leaves(self):
        if self.branches_for_leaves is None: 
            self.find_path_for_leaves()        
        path_branch=self.branches_for_leaves
        
        self.assign_weights_for_branches()
        
        units=self.assert_parent_nodes()
            
        # iterate through branch ids for each leaf
        path_weights_sample = []
        path_weights_depth = []
        for branches_c in path_branch:
            weights_sample_c=[]
            weights_depth_c=[]
            for branch_c_i in branches_c:
                for u_ in units:
                    if branch_c_i in u_.branch_ids:
                        weights_sample_c.append(u_.branch_weights_sample[u_.branch_ids.index(branch_c_i)])
                        weights_depth_c.append(u_.branch_weights_depth[u_.branch_ids.index(branch_c_i)])
                        break
            path_weights_sample.append(self.normalize_branch_weights(weights_sample_c))
            path_weights_depth.append(self.normalize_branch_weights(weights_depth_c))
        
        self.branch_sample_weights_for_leaves=path_weights_sample
        self.branch_depth_weights_for_leaves=path_weights_depth
        
    def display_extended_linkage(self,hc_type='lcpn'):
                
        units = self.assign_units(hc_type=hc_type)
        if hc_type in ['lcpn','lcpn+','lcpn+f']:
            print('Extended linkage table for LCPN nodes:')
            for u_ in units:
                if u_.parent is None:
                    parent_id=None
                else:
                    parent_id=u_.parent.id
                
                if u_.left is None:
                    left_id=None
                else:
                    left_id=u_.left.id
                    
                if u_.right is None:
                    right_id=None
                else:
                    right_id=u_.right.id
                
                if u_.branch_weights_sample is None:
                    print(f'node_id:{u_.id}, node_type:{u_.node_type}, subsets:{u_.subsets}, branch_ids:{u_.branch_ids}, parent_id:{parent_id}, left_id:{left_id}, right_id:{right_id}')
                else:
                    w_samp=np.round(u_.branch_weights_sample,2)
                    print(f'node_id:{u_.id}, node_type:{u_.node_type}, subsets:{u_.subsets}, branch_ids:{u_.branch_ids}, branch_weights_sample:{w_samp}, parent_id:{parent_id}, left_id:{left_id}, right_id:{right_id}')
                    
    def compose_global_training_set(self,x_train,y_train,verbose=False):
        units = self.assign_units(hc_type=hc_type)        
        for i in range(len(units)):
            y_tr_i, x_tr_i = units[i].select_subsamples(y_train, x_train)
            if i==0:
                x_tr_glob = x_tr_i
                y_tr_glob = np.where(y_tr_i,units[i].branch_ids[1],units[i].branch_ids[0])  
            else:
                x_tr_glob = np.r_[x_tr_glob,x_tr_i]
                y_tr_glob = np.r_[y_tr_glob,np.where(y_tr_i,units[i].branch_ids[1],units[i].branch_ids[0])]
        
        self.global_classes = np.unique(y_tr_glob).tolist()
                
        if verbose:
            classes_glob, class_counts_glob = np.unique(y_tr_glob, return_counts=True)
            n_class_glob = len(classes_glob)
            n_sample_glob = x_tr_glob.shape[0]
            
            classes, class_counts = np.unique(y_train, return_counts=True)
            n_class = len(classes)
            n_sample = x_train.shape[0]
            print(f'Global training set n_class:{n_class_glob}, classes:{classes_glob}, n_sample:{n_sample_glob}, class_counts:{class_counts_glob}')
            print(f'Flat training set n_class:{n_class}, classes:{classes}, n_sample:{n_sample}, class_counts:{class_counts}')
            
        return x_tr_glob, y_tr_glob
               
    # utility function to estimate chain probabilities of branches of each flat label
    def estimate_proba_chain(self,pred_proba_all, hc_type='lcpn+', weights='none', proportion='inverse', chain_type='average'):
                    
        if self.branches_for_leaves is None: 
            self.find_path_for_leaves()
        path_branch=self.branches_for_leaves
        
        if weights=='none':
            w_=np.ones(self.n_flat_labels)
        else:
            self.assign_weight_path_for_leaves()
            if weights=='sample':
                w_=self.branch_sample_weights_for_leaves
            elif weights=='depth':
                w_=self.branch_depth_weights_for_leaves
            
            if proportion in ['direct','inverse']:
                self.branch_weight_proportion=proportion
            else:
                raise ValueError('Invalid proportion definition')
                       
        for i_,c in enumerate(self.flat_labels):
            proba_chain_i = pred_proba_all[:,path_branch[i_]]*w_[i_]
            
            if chain_type=='prod':
                proba_i = np.multiply.reduce(proba_chain_i,1)
            elif chain_type=='average':
                proba_i = proba_chain_i.mean(axis=1)
                                
            if i_==0:
                pred_proba_leaves = proba_i
            else:
                pred_proba_leaves = np.c_[pred_proba_leaves,proba_i]
        
        return pred_proba_leaves
    
    def fit(self, clf_base, x_train, y_train=None, hc_type='lcpn+', rseed=None, multi_process=False):
        if y_train is None: y_train=self.y_train
        
        units = self.assign_units(hc_type=hc_type)
                    
        def fit_local_clf(units,i):
            if type(clf_base) is list:
                for i in range(n_job):
                    if i<len(clf_base):
                        units[i].set_clf_base(clf_base[i],rseed=rseed)
                    else:
                        units[i].set_clf_base(clf_base[-1],rseed=rseed)
            else:
                units[i].set_clf_base(clf_base,rseed=rseed)
            model_fitted = units[i]._clf_fit(x_train,y_train)
            return model_fitted
                    
        if hc_type in ['lcpn','lcpn+','lcpn+f']:
            """
            lcpn: Local Classifier per Node
            Training may happen disjointly and independently at each node. 
            Hence, gives room to parallel-processing.
            
            Receive the nodes as a list
            """
            n_job = len(units)
            if multi_process:
                n_jobs = os.cpu_count() #min(n_job,os.cpu_count())
                res = Parallel(n_jobs=n_jobs,verbose=0)(delayed(fit_local_clf)(units,i) for i in range(n_job))
                [units[i].set_clf_base(res[i]) for i in range(len(res))]
            else:
                [fit_local_clf(units,i) for i in range(n_job)]            
                
            if hc_type == 'lcpn+f':
                """
                lcpn+f: Local Classifier per Node (for non-leaf labels) + flat classifier (for loeaf labels)
                In this scheme, not all parent nodes are to be fitted; only when they include at least one 
                non-leaf node in their subsets.
                
                """             
                # Additionally train a flat classifier for flat (leaf) labels
                if self.pred_proba_fc is None:
                    self.set_clf_flat(clf_base,rseed=rseed)
                    self.clf_flat.fit(x_train,y_train)
            
        elif hc_type == 'gc':
            """
            To take into account the class hierarchy as a whole, augment all (hierarchical+flat) 
            classes into a single x_train and y_train set and train the model with a single run.
            """
            x_tr_glob,y_tr_glob=self.compose_global_training_set(x_train,y_train,hc_type=hc_type,verbose=0)
                       
            self.set_clf_glob(clf_base,rseed=rseed)
            self.clf_glob.fit(x_tr_glob,y_tr_glob)
                           
    def predict(self, x_test, hc_type='lcpn+', weights='none', proportion='direct', chain_type='prod', usefulness=False, multi_process=False):
                
        units = self.assign_units(hc_type=hc_type)
        
        # utility function to predict chain probabilities of branches of each flat label
        def predict_proba_chain(pred_proba_all):
            
            pred_proba_leaves=self.estimate_proba_chain(pred_proba_all,
                                                        hc_type=hc_type,
                                                        weights=weights, 
                                                        proportion=proportion, 
                                                        chain_type=chain_type)
            
            y_pred = np.argmax(pred_proba_leaves,axis=1)
            self.y_pred = y_pred
            self.pred_proba_raw = pred_proba_leaves
            return y_pred
            
        if hc_type=='lcpn':
            """
            Label each sample one by one with a “top-down” class prediction 
            strategy. For each level of the hierarchy (except the top level), 
            the decision about which class is predicted at the current level 
            is based on the class predicted at the previous (parent) level.
            """
            def predict_top_down(i):
                next_id = 0
                # y_pred_dict[i]=[]
                for j in range(n_node):
                    y_pred_j = units[next_id].clf_base.predict(x_test[i,:].reshape(1,-1))[0]
                    # y_pred_dict[i].append(y_pred_j)
                    if y_pred_j == 0:
                        if units[next_id].branch_ids[0] in self.flat_labels:
                            y_pred[i] = units[next_id].branch_ids[0]
                            break
                        else:
                            if units[next_id].left is not None:
                                next_id = units[next_id].left.id
                            else:
                                warnings.warn('Inconsistency: no leaf and no left branch!')
                    else:
                        if units[next_id].branch_ids[1] in self.flat_labels:
                            y_pred[i] = units[next_id].branch_ids[1]
                            break
                        else:
                            if units[next_id].right is not None:
                                next_id = units[next_id].right.id
                            else:
                                warnings.warn('Inconsistency: no leaf and no right branch!')
                        
            n_job = x_test.shape[0] # this step can be done in parallel
            n_node = len(units)
            y_pred = np.zeros(n_job)
            # y_pred_dict = {}
            
            if multi_process:
                Parallel(n_jobs=os.cpu_count(),verbose=0,require='sharedmem')(delayed(predict_top_down)(i) for i in range(n_job))
            else:                
                [predict_top_down(i) for i in range(n_job)]
            
            self.y_pred = y_pred            
            return y_pred#, y_pred_dict
        
        elif hc_type == 'lcpn+':
            """
            Label each sample not with "top-down" but with taking into account all classifiers
            in the parent nodes. Error propagation is thereby expected to be minimized. 
            """
            pred_proba_all=np.zeros((x_test.shape[0],self.n_branches))
            for u_ in units:
                if len(u_.branch_ids)>1:
                    pred_proba_all[:,u_.branch_ids]=u_.clf_base.predict_proba(x_test)
                else:
                    pred_proba_all[:,u_.branch_ids[0]]=u_.clf_base.predict_proba(x_test)[:,0]
            
            y_pred=predict_proba_chain(pred_proba_all)
            return y_pred
        
        elif hc_type == 'lcpn+f':
            pred_proba_all=np.zeros((x_test.shape[0],self.n_branches))            
            for u_ in units:
                if u_.clf_base is not None:
                    preds_=u_.clf_base.predict_proba(x_test)
                    for i,s in enumerate(u_.subsets):
                        if len(s)>1:
                            pred_proba_all[:,u_.branch_ids[i]]=preds_[:,i]            
                
            if self.pred_proba_fc is None:
                pred_proba_flat_only = self.clf_flat.predict_proba(x_test)
            else:
                pred_proba_flat_only = self.pred_proba_fc
            pred_proba_all[:,:pred_proba_flat_only.shape[1]]=pred_proba_flat_only
            
            y_pred=predict_proba_chain(pred_proba_all)
            return y_pred
            
        elif hc_type=='gc':
            """
            Each test example is classified by the induced model, a process that can assign classes
            at potentially every level of the hierarchy to the test example.
            """
            if usefulness:
                """
                Build usefulness array for each global label. For details, see: Silla and 
                Freitas 2009, "A Global-Model Naive Bayes Approach to the Hierarchical 
                Prediction of Protein Functions") 
                """
                useful_=np.zeros(self.n_branches)
                a_ci=.95
                for p_ in range(len(units)):
                    for i_,c_i in enumerate(units[p_].branch_ids):
                        useful_[c_i]=np.log2(len(units[p_].subsets[i_]))                        
                useful_=1-a_ci*useful_/useful_.max()
                
            else:
                useful_=1
                
            if self.pred_proba_gc is None:
                pred_proba_gc = self.clf_glob.predict_proba(x_test)
            else:
                pred_proba_gc=self.pred_proba_gc
            pred_proba_gc *= useful_
            
            y_pred=predict_proba_chain(pred_proba_gc)
            return y_pred
               
    def predict_proba(self,x_test,hc_type='lcpn+f'):
        n_class = len(np.unique(self.y_train))
        
        if not hasattr(self,'pred_proba_raw') or self.pred_proba_raw is None:
            self.predict(x_test,hc_type=hc_type)
        
        pred_probas_raw=self.pred_proba_raw
        
        pred_proba_=np.zeros(pred_probas_raw.shape)
        for i_, row_ in enumerate(pred_probas_raw):
            if row_.sum()==0:
                pred_proba_[i_]=1/n_class
            else:
                pred_proba_[i_]=row_/row_.sum()
        
        self.pred_proba=pred_proba_
        return pred_proba_    
        
    def score(self, y_pred=None, pred_proba=None, percent=False, eval_metric='f1'):
        if eval_metric in ['acc','bac','f1']:
            if y_pred is None:
                if hasattr(self,'y_pred'):
                    y_pred = self.y_pred
                else:
                    raise ValueError('Can not get score without y_pred. Make sure it exists.')
            else:
                if hasattr(self,'y_pred') and np.any(y_pred != self.y_pred):
                    warnings.warn('Discrepancy between y_pred provided and the instance variable y_pred.')
                    
            if eval_metric=='acc':
                score = metrics.accuracy_score(self.y_test, y_pred)
            elif eval_metric=='bac':
                score = metrics.balanced_accuracy_score(self.y_test, y_pred)
            elif eval_metric=='f1':
                score = metrics.f1_score(self.y_test, y_pred, average='macro')
                
        else:
            if pred_proba is None:
                if hasattr(self,'y_pred'):
                    pred_proba = self.pred_proba
                else:
                    raise ValueError('Can not get score without pred_proba. Make sure it exists.')
            else:
                if hasattr(self,'pred_proba') and np.any(pred_proba != self.pred_proba):
                    warnings.warn('Discrepancy between pred_proba provided and the instance variable pred_proba.')
            
            if eval_metric=='auc':
                score = metrics.roc_auc_score(self.y_test, pred_proba, average='weighted', multi_class='ovr')
            elif eval_metric=='nll':
                score = metrics.log_loss(self.y_test, pred_proba)                        
            
        if percent: score *= 100
        return score
          
    def assign_indices(self,node):
        # Assign indices that agree with the subsets of a node
        y_train = self.y_train
        y_test  = self.y_test
        inds_train = {}
        inds_test = {}
        for i,gr in enumerate(node.subsets):
            inds_train[i]= np.array([],dtype=int)
            inds_test[i] = np.array([],dtype=int)
            # Find out where indices for selected class memberships 
            # (i.e. subsets) occur
            if (type(gr)==list or type(gr)==np.ndarray) and len(gr)>1:
                for j in gr:
                    loc_train= np.where(y_train==j)[0].astype(int)
                    loc_test = np.where(y_test==j)[0].astype(int)
                    inds_train[i]= np.r_[inds_train[i],loc_train]
                    inds_test[i] = np.r_[inds_test[i],loc_test]
            else:
                loc_train= np.where(y_train==gr)[0].astype(int)
                loc_test = np.where(y_test==gr)[0].astype(int)
                inds_train[i]= np.r_[inds_train[i],loc_train]
                inds_test[i] = np.r_[inds_test[i],loc_test]
        node.set_inds_train(inds_train)
        node.set_inds_test(inds_test)
        # print(inds_train)
               
    def traverse_children(self,node,PNs):
        children=[]
        flat_labels_here=node.set_flat_labels()
        n_node = len(PNs)        
        for nid in range(node.id+1,n_node):
            for c in flat_labels_here:
                flat_labels_there=PNs[nid].set_flat_labels()
                if c in flat_labels_there:
                    if PNs[nid].id not in children:
                        children.append(PNs[nid].id)
        node.children=children
        return children
            
    # evaluate clustering performance given a hierarchy and a training set
    def evaluate_hier_clusters(self,
                               x_train,
                               y_train,
                               hc_type='lcpn',
                               metric_='ss',
                               weights='none',
                               proportion='direct',
                               chain_type='average',
                               verbose=False):
        
        def get_eval_score(X,y):
            if metric_=='ss':
                score_=(SS(X,y)+1)/2
            elif metric_=='chi':
                score_=CHI(X,y)
            elif metric_=='dbi':
                score_=DBI(X,y)
            return score_
            
        n_class=self.n_flat_labels
        # obtain cluster evaluation with different metrics
        clust_eval_score_flat=get_eval_score(x_train,y_train)
                 
        clust_eval_scores_all=np.zeros(self.n_branches) 
        if 'gc' in hc_type:
            x_tr_glob,y_tr_glob=self.compose_global_training_set(x_train,y_train,hc_type=hc_type,verbose=0)
            glob_class=self.global_classes            
            glob_clust_score=get_eval_score(x_tr_glob,y_tr_glob)
            clust_eval_scores_all[glob_class]=glob_clust_score              
            if verbose==2:               
                print(f'Global classes {glob_class} cluster eval score:{glob_clust_score:.4f}')
        else:
            
            units_=self.assign_units(hc_type=hc_type)
            for u_ in units_:
                y_tr_sub, x_tr_sub = u_.select_subsamples(y_train,x_train)
                unit_cluster_score=get_eval_score(x_tr_sub,y_tr_sub)
                clust_eval_scores_all[u_.branch_ids]=unit_cluster_score
                u_.cluster_eval_score=unit_cluster_score
                            
                if verbose==2:                    
                    print(f'{hc_type} node {u_.id} cluster eval score:{unit_cluster_score:.4f}')
        
        if '+f' in hc_type:
            clust_eval_scores_all[:n_class]=clust_eval_score_flat
            if verbose==2:            
                print(f'{hc_type} flat labels cluster eval score:{clust_eval_score_flat:.4f}')
            
        pred_proba_leaves=self.estimate_proba_chain(clust_eval_scores_all.reshape(1,-1),
                                                    weights=weights,
                                                    proportion=proportion,
                                                    chain_type=chain_type)
            
        classes, class_counts = np.unique(self.y_train, return_counts=True)
        class_weights_=[class_counts[i] for i in classes]/class_counts.sum()
        
        est_score=np.sum(class_weights_*pred_proba_leaves)
        
        if verbose:
            print(f'Hierarchy cluster eval score:{est_score:.4f}, flat cluster:{clust_eval_score_flat:.4f}')
        
        return est_score
    
    # evaluate clustering performance given a hierarchy and a training set using a clustering function
    def evaluate_hier_clusters_wclustf(self,
                                       x_train,
                                       y_train,
                                       hc_type='lcpn',
                                       clustf=None,
                                       rseed=0,
                                       weights='none',
                                       proportion='direct',
                                       chain_type='average',
                                       verbose=False):
                
        n_class=self.n_flat_labels
                
        # perform clustering first and then evaluate the performance
        if clustf is None:
            clustf = KMeans(n_clusters=n_class, init='k-means++', n_init='auto', random_state=rseed)
        y_pred = clustf.fit_predict(x_train)        
        clust_perf_score_flat = metrics.v_measure_score(y_train, y_pred)       
               
        clust_perf_scores_all=np.zeros(self.n_branches)    
        if 'gc' in hc_type:
            x_tr_glob,y_tr_glob=self.compose_global_training_set(x_train,y_train,hc_type=hc_type,verbose=0)
            glob_class=self.global_classes
            clustf.n_clusters = len(glob_class)
            y_pred = clustf.fit_predict(x_tr_glob)        
            glob_kmeans_score = metrics.v_measure_score(y_tr_glob, y_pred)
            clust_perf_scores_all[glob_class]=glob_kmeans_score
                
            if verbose==2:
                print(f'Global classes {glob_class} cluster perf score:{glob_kmeans_score:.4f}')
        else:
            
            units_=self.assign_units(hc_type=hc_type)
            for u_ in units_:
                y_tr_sub, x_tr_sub = u_.select_subsamples(y_train,x_train)
                clustf.n_clusters = len(u_.subsets)                
                y_pred = clustf.fit_predict(x_tr_sub)
                unit_kmeans_score_ = metrics.v_measure_score(y_tr_sub, y_pred)
                clust_perf_scores_all[u_.branch_ids]=unit_kmeans_score_
                u_.cluster_perf_score=unit_kmeans_score_
                            
                if verbose==2:
                    print(f'{hc_type} node {u_.id} cluster perf score:{unit_kmeans_score_:.4f}')
        
        if '+f' in hc_type:
            clust_perf_scores_all[:n_class]=clust_perf_score_flat
            if verbose==2:
                print(f'{hc_type} flat labels cluster perf score:{clust_perf_score_flat:.4f}')
            
        pred_proba_leaves=self.estimate_proba_chain(clust_perf_scores_all.reshape(1,-1), 
                                                    weights=weights, 
                                                    proportion=proportion, 
                                                    chain_type=chain_type)
            
        classes, class_counts = np.unique(self.y_train, return_counts=True)
        class_weights=[class_counts[i] for i in classes]/class_counts.sum()
        
        est_score=np.sum(class_weights*pred_proba_leaves)
        if verbose:
            print(f'Hierarchy clustering score with {type(clustf).__name__}:{est_score:.4f}, fclust:{clust_perf_score_flat:.4f}')
        
        return est_score
        
    def estimate_expected_testing_performance(self, hc_type='lcpn', score_fc_te=0):
        classes, class_counts = np.unique(self.y_train, return_counts=True)
        class_weights_=[class_counts[i] for i in classes]/class_counts.sum()
        if hc_type in ['lcpn','lcpn+f']:
            expected_scores=np.array([node_i.expected_score for node_i in self.pnodes])
            # print(expected_scores)
            if hc_type == 'lcpn':                    
                score_exp_te=np.sum([class_weights_[i]*np.multiply.reduce(expected_scores[self.parents_for_leaves[i]]) for i in classes])
            elif hc_type == 'lcpn+f':
                score_exp_te=np.sum([class_weights_[i]*np.multiply.reduce(expected_scores[self.parents_for_leaves[i]][:-1])*score_fc_te for i in classes])
        
        self.expected_test_perf=score_exp_te
        return score_exp_te

    def BFC(self):
        PNs=self.assert_parent_nodes()
        bfc_sum = 0
        alpha_sum = 0
        for ni in PNs:
            alpha = len(ni.subsets[1])+len(ni.subsets[0])-2
            alpha_sum += alpha
            bfc_ni = len(ni.subsets[1])-len(ni.subsets[0])
            # print(bfc_ni)
            bfc_sum += bfc_ni
        bfc_=np.abs(np.round(bfc_sum/alpha_sum,2))
        self.bfc=bfc_
        return bfc_

    def BFX(self,y=None):
        PNs=self.assert_parent_nodes()
        if y==None:
            y=self.y_train
        bfx_sum = 0
        alpha_sum = 0
        for ni in PNs:
            X_ni0 = 0
            X_ni1 = 0
            for c in ni.subsets[0]:
                X_ni0 += sum(y[y==c])
            for c in ni.subsets[1]:
                X_ni1 += sum(y[y==c])
            alpha = X_ni0+X_ni1-2
            alpha_sum += alpha
            bfx_ni = X_ni1-X_ni0
            # print(bfc_ni)
            bfx_sum += bfx_ni
        bfx_=np.round(bfx_sum/alpha_sum,2)
        return np.abs(bfx_)
        
if __name__=='__main__':
    from utils import (prep_data, 
                        plot_dendogram, 
                        generate_rand_dist,
                        monotonize_rescale_,
                        class_labels_sanity_check,)
    # from utils_params import _dataset_name_dict
    from hdc_ import hdc
    from clf_repo import clf_dict
    from sklearn import metrics
    
    fname = 'Glass'
    repo = 'uci'
    
    clfname = 'xgb'
    hc_type = 'lcpn+'
    rseed = 0
    
    dset,_ = prep_data(fname,
                      repo=repo,
                      sel_class=None,
                      reorder=False,
                      orig_split=False, 
                      stratify=True, 
                      k=5,
                      kth=2,
                      rs=None,
                      shuffle_=False)
    x_train, x_test, y_train, y_test = dset
    classes = class_labels_sanity_check(y_train,y_test)
    n_class = len(classes)
    
    #class conditional means
    ccms=np.array([x_train[y_train == i].mean(axis=0) for i in classes])
    
    Z, PNs = hdc(y=ccms,
                  split_fun='kmeans').fit()
    Z = monotonize_rescale_(Z)
    
    plot_dendogram(Z,close_all=1,orient='top')
    #%%
    # Build hc tree
    tree = hier_binary_tree(link_mat=Z,
                            y_train=y_train,
                            y_test=y_test,)
    
    tree.display_extended_linkage(hc_type=hc_type)
    
    clf=copy.deepcopy(clf_dict[clfname])
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    f1_fc = metrics.f1_score(y_test, y_pred, average='macro')
    
    clf=copy.deepcopy(clf_dict[clfname])
    tree.fit(clf,x_train,hc_type=hc_type,rseed=rseed)
    
    weights_= 'none'
    proportion_= 'direct'
    prob_= 'average'
    tree.predict(x_test, hc_type=hc_type, weights=weights_, proportion=proportion_, chain_type=prob_, usefulness=0)
    print(f'{tree.score()}')
    
    tree.evaluate_hier_clusters(x_train, y_train, hc_type=hc_type, verbose=1)
    tree.evaluate_hier_clusters_wclustf(x_train, y_train, hc_type=hc_type, verbose=1)       