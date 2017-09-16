import copy
import random
from collections import Iterable

import numpy as np
import pandas as pd

from scipy import sparse as sp

# =====================================================================
# functions for splitting task data
# =====================================================================
def split_tag(A, tids_hash, train_ratio=0.9):
    """"""
    n_items = A.shape[0]
    n_train = int(n_items * train_ratio)
    n_valid = n_items - n_train

    tids = copy.deepcopy(tids_hash.keys())
    random.shuffle(tids)

    tids_train = tids[:n_train]
    tids_valid = tids[n_train:]

    Atr = A[[tids_hash[i] for i in tids_train]]
    Avl = A[[tids_hash[i] for i in tids_valid]]

    return (Atr, Avl), (tids_train, tids_valid)

def split_pref(A, tids_hash, uids_hash, train_ratio=0.9, thresh=10):
    """"""
    # this one is more subtle, since we need to delete ratings (not items)
    Q = A.copy()
    Q.data = np.ones(Q.data.shape)
    n_listens = Q.sum(axis=0)

    # find users who litened more than 10 songs
    act_user = np.where(n_listens > thresh)[1]
    act_user_hash = {k:v for k, v in enumerate(act_user)}
    P = A.tocsc()[:, act_user]

    res_tr, res_vl = [], []
    for i in xrange(len(P.indptr)-1):
        rows = P.indices[P.indptr[i]:P.indptr[i+1]]
        data = P.data[P.indptr[i]:P.indptr[i+1]]
        n_train = int(len(rows) * train_ratio)
        rnd_idx = range(len(rows))
        np.random.shuffle(rnd_idx)
        rows_tr, data_tr = rows[rnd_idx[:n_train]], data[rnd_idx[:n_train]]
        rows_vl, data_vl = rows[rnd_idx[n_train:]], data[rnd_idx[n_train:]]
        res_tr.extend(
            [(j, act_user_hash[i], d) for j, d in zip(rows_tr, data_tr)])
        res_vl.extend(
            [(j, act_user_hash[i], d) for j, d in zip(rows_vl, data_vl)])

    # collect inactive users' triplet
    inact_user = np.where(n_listens <= thresh)[1]
    inact_user_hash = {k:v for k, v in enumerate(inact_user)}
    R = A[:, inact_user].tocoo()

    res_tr.extend(
        [(row, inact_user_hash[col], dat)
         for row, col, dat
         in zip(R.row, R.col, R.data)]
    )
    Atr = sp.coo_matrix(
        (map(lambda x:x[2], res_tr),
         (map(lambda x:x[0], res_tr),
          map(lambda x:x[1], res_tr))), shape=A.shape)
    Avl = sp.coo_matrix(
        (map(lambda x:x[2], res_vl),
         (map(lambda x:x[0], res_vl),
          map(lambda x:x[1], res_vl))), shape=A.shape)

    # find train 'items' and valid 'items'
    # train items only contain rows has at least one rating
    # TODO:

    return Atr, Avl

def split_tag(A, tids_hash, train_ratio=0.9):
    """"""
    n_items = A.shape[0]

    n_train = int(n_items * train_ratio)
    n_valid = n_items - n_train

    tids = copy.deepcopy(tids_hash.keys())
    random.shuffle(tids)

    tids_train = tids[:n_train]
    tids_valid = tids[n_trian:]

    Atr = A[[tids_hash[i] for i in tids_train]]
    Avl = A[[tids_hash[i] for i in tids_valid]]

    return (Atr, Avl), (tids_train, tids_valid)

# =====================================================================
# functions for semantic analysis 
# =====================================================================
def get_topic_terms(V, terms, k=10):
    """"""
    top_k_terms = []
    for v in V:
        top_k_terms.append(
            [terms[i] for i in np.argsort(v)[-k:][::-1]])
    return top_k_terms

# =====================================================================
# functions for misc processes 
# =====================================================================
def filter_df_base_on_minfb(df, target, min_fb):
    """"""
    count_df = df.groupby(target).size()
    count = zip(
        count_df.index.tolist(),
        count_df.tolist()
    )

    targets = set(
        map(
            lambda x:x[0],
            filter(
                lambda y:y[1] > min_fb,
                count
            )
        )
    )

    # now filter out triplet based on user
    i = df.columns.tolist().index(target)
    return pd.DataFrame(
        map(
            lambda y:(y[0], y[1], y[2]),
            filter(
                lambda x:x[i] in targets,
                df.as_matrix()
            )
        ),
        columns=df.columns
    )

def densify_trp_df(df, min_fb=5):
    """"""
    if isinstance(min_fb, int):
        min_fb_tup = (min_fb, min_fb)
    elif isinstance(min_fb, Iterable):
        if len(min_fb) > 2:
            raise Exception
        min_fb_tup = min_fb

    cols = df.columns

    filt_diff = 1
    df_cur = df
    while filt_diff > 0:
        df_filt_col0 = filter_df_base_on_minfb(df_cur, cols[0], min_fb[0])
        df_filt_col1 = filter_df_base_on_minfb(df_filt_col0, cols[1],
                                               min_fb[1])
        filt_diff = len(df_cur) - len(df_filt_col1)
        df_cur = df_filt_col1

    return df_cur
