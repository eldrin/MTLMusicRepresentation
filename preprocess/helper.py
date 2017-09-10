import copy
import random
from collections import OrderedDict, Iterable
import sqlite3

import numpy as np
import pandas as pd

from scipy import sparse as sp
from sklearn.mixture import GaussianMixture
from processor import MatrixFactorization

from helper import densify_trp_df

# =====================================================================
# functions for reading task specific raw data
# =====================================================================
def read_pref(db_fn):
    """"""
    with sqlite3.connect(db_fn) as conn:
        c = conn.cursor()
        song2track = OrderedDict(
            [(r[1], r[2]) for r in c.execute('SELECT * FROM songtrack')])
        tracks = OrderedDict(
            [(r[0], r[1]) for r in c.execute('SELECT * FROM tracks')])
        triplet = [(r[1], tracks[song2track[r[2]]], r[3])
                   for r in c.execute('SELECT * FROM taste_summary')]

        tids = set([r[1] for r in triplet])
        tids_hash = OrderedDict([(v,k) for k,v in enumerate(tids)])
        uids = set([r[0] for r in triplet])
        uids_hash = OrderedDict([(v,k) for k,v in enumerate(uids)])

    n_users = len(uids)
    n_tracks = len(tids)

    A = sp.coo_matrix(
        (
            map(lambda x:int(x[2]), triplet),
            (
                map(lambda x:int(uids_hash[x[0]]), triplet),
                map(lambda x:int(tids_hash[x[1]]), triplet)
            )
        ),
        shape=(n_users, n_tracks)
    ) # (n_users, n_items)
    A = A.tocsr().T # (n_items, n_users)
    return A, tids_hash, uids_hash


def read_tag(db_fn):
    """"""
    with sqlite3.connect(db_fn) as conn:
        c = conn.cursor()

        triplet = [
            (r[0]-1, r[1]-1, r[2])
            for r in c.execute('SELECT * FROM tid_tag')
        ]

        tags = [r[0] for r in c.execute('SELECT * FROM tags')]
        tags_hash = OrderedDict([(v,k) for k,v in enumerate(tags)])
        tids = [r[0] for r in c.execute('SELECT * FROM tids')]
        tids_hash = OrderedDict([(v,k) for k,v in enumerate(tids)])

    triplet = [(r[0], r[1], r[2]+1) if r[2]==0 else r for r in triplet]

    A = sp.coo_matrix(
        (
            map(lambda x:int(x[2]), triplet),
            (
                map(lambda x:int(x[0]), triplet),
                map(lambda x:int(x[1]), triplet)
            )
        ),
        shape=(len(tids), len(tags))
    ) # (n_items, n_tags)
    return A, tids_hash, tags_hash


def read_bpm(db_fn):
    """"""
    data = joblib.load(db_fn)
    A = data['item_factors']
    tids = data['tids']
    tids_hash = OrderedDict([(v,k) for k,v in enumerate(tids)])

    return A, tids_hash

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
# functions for processing task specific raw data
# =====================================================================
def process_pref(k, A, i_hash, u_hash, alg='plsa'):
    """"""
    mf = MatrixFactorization(k, A, i_hash, u_hash, alg)
    U = mf.fit_transform(A)
    V = mf.components_
    return U, V

def process_tag(k, A, i_hash, t_hash, alg='plsa'):
    """"""
    mf = MatrixFactorization(k, A, i_hash, t_hash, alg)
    U = mf.fit_transform(A)
    V = mf.components_
    return U, V

def process_bpm(k, A, i_hash, alg='gmm'):
    """"""
    gmm = GaussianMixture(k)
    gmm.fit(A)
    U = gmm.predict_proba(A)
    return U


# =====================================================================
# functions for saving processed results
# =====================================================================
def save(fn, item_factors, term_factors, item_ids, term_ids):
    """"""
    data = {
        'item_factors':item_factors,
        'term_factors':term_factors,
        'tids':item_ids, # track ids
        'uids':term_ids
    }
    joblib.dump(data, fn)


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
