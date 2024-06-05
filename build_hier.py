from hdc import hdc
from hc import hier_binary_tree
from utils import monotonize_rescale_
def build_tree(diss_,y_train,y_test,**kwd):
    if 'link_type' in kwd.keys():
        link_type = kwd['link_type']
    else:
        link_type = 'hdc'
    
    # hdc arguments
    if 'input_type' in kwd.keys():
        input_type = kwd['input_type']
    else:
        input_type = 'obs_vec'
    
    if 'split_fun' in kwd.keys():
        split_fun = kwd['split_fun']
    else:
        split_fun = 'kmeans'
    
    # hac arguments
    if 'agg_dist' in kwd.keys():
        agg_dist = kwd['agg_dist']
    else:
        agg_dist = 'ward'
        
    # tree arguments
    if 'pred_proba_fc' in kwd.keys():
        pred_proba_fc = kwd['pred_proba_fc']
    else:
        pred_proba_fc = None
        
    if link_type=='hdc':
        model = hdc(y=diss_,
                    input_type= input_type,
                    split_fun = split_fun)
        Z_, PNs = model.fit()
        Z = monotonize_rescale_(Z_)
        tree = hier_binary_tree(pnodes=PNs,
                                y_train=y_train,
                                y_test=y_test,
                                link_mat=Z,
                                pred_proba_fc=pred_proba_fc)
    elif link_type=='hac':
        Z = sch.linkage(diss_,
                        method=agg_dist,)
        Z = monotonize_rescale_(Z_)
        tree = hier_binary_tree(y_train=y_train,
                                y_test=y_test,
                                link_mat=Z,
                                pred_proba_fc=pred_proba_fc)
    return tree, Z