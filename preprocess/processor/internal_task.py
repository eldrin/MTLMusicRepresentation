from abc import ABCMeta, abstractmethod
import sqlite3
from collections import OrderedDict, Iterable

import numpy as np
from scipy import sparse as sp

from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture

from factorization import MatrixFactorization
from utils.misc import triplet2sparse


class BaseInternalTask:
    __metaclass__ = ABCMeta
    """"""
    def __init__(self, n_components, db_fn, n_iter, alg='plsa'):
        """"""
        self.k = n_components
        self.db_fn = db_fn
        self.n_iter = n_iter
        self.alg = 'plsa'

        self.U = None
        self.V = None
        self.doc_hash = OrderedDict()
        self.term_hash = OrderedDict()

    @abstractmethod
    def process(self):
        """"""
        pass

    @property
    def data(self):
        """"""
        return {
            'item_factors': self.U,
            'term_factors': self.V,
            'tids': self.doc_hash.keys(),
            'uids': self.term_hash.keys()
        }

    @classmethod
    @abstractmethod
    def read(cls, db_fn):
        """"""
        pass

    def save(self, fn):
        """"""
        joblib.dump(self.data, fn)


class MFTask(BaseInternalTask):
    """"""
    def __init__(self, n_components, db_fn, n_iter, alg='plsa'):
        """"""
        super(MFTask, self).__init__(n_components, db_fn, n_iter, alg)

        self.A = sp.coo_matrix((1,1),dtype=int)
        self.doc_hash = OrderedDict()
        self.term_hash = OrderedDict()

        self.mf = MatrixFactorization(
            self.k, n_iter=self.n_iter, alg=self.alg)

    def process(self):
        """"""
        self.U = self.mf.fit_transform(self.A)
        self.V = self.mf.components_


class MSDTaste(MFTask):
    """"""
    def __init__(self, n_components, db_fn, n_iter, alg='plsa'):
        """"""
        super(MSDTaste, self).__init__(n_components, db_fn, n_iter, alg)
        self.A, self.doc_hash, self.term_hash = self.read(db_fn)

    @classmethod
    def read(cls, db_fn):
        """"""
        with sqlite3.connect(db_fn) as conn:
            c = conn.cursor()
            song2track = OrderedDict(
                [(r[1], r[2]) for r in c.execute('SELECT * FROM songtrack')])
            tracks = OrderedDict(
                [(r[0], r[1]) for r in c.execute('SELECT * FROM tracks')])
            triplet = [(tracks[song2track[r[2]]], r[1], r[3])
                       for r in c.execute('SELECT * FROM taste_summary')]

            tids = set([r[0] for r in triplet])
            tids_hash = OrderedDict([(v,k) for k,v in enumerate(tids)])
            uids = set([r[1] for r in triplet])
            uids_hash = OrderedDict([(v,k) for k,v in enumerate(uids)])

        # (n_tracks, n_users)
        A = triplet2sparse(triplet, tids_hash, uids_hash)
        return A.tocsr(), tids_hash, uids_hash


class LastFMTag(MFTask):
    """"""
    def __init__(self, n_components, db_fn, n_iter, alg='plsa'):
        """"""
        super(LastFMTag, self).__init__(n_components, db_fn, n_iter, alg)
        self.A, self.doc_hash, self.term_hash = self.read(db_fn)

    @classmethod
    def read(cls, db_fn):
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

        A = triplet2sparse(triplet)
        return A.tocsr(), tids_hash, tags_hash


class MXMLyrics(MFTask):
    """"""
    def __init__(self, n_components, db_fn, n_iter, alg='plsa', tfidf=True):
        """"""
        super(MXMLyrics, self).__init__(n_components, db_fn, n_iter, alg)
        self.A, self.doc_hash, self.term_hash = self.read(db_fn)

        # for lyrics, applying TF-IDF makes difference?
        if tfidf:
            from sklearn.feature_extraction.text import TfidfTransformer
            self.tfidf = TfidfTransformer()
            self.A = self.tfidf.fit_transform(self.A)

    @classmethod
    def read(cls, db_fn):
        """"""
        with sqlite3.connect(db_fn) as conn:
            c = conn.cursor()

            triplet = c.execute(
                'SELECT track_id, word, count FROM lyrics').fetchall()

            tracks = c.execute('SELECT DISTINCT track_id FROM lyrics').fetchall()
            track_hash = OrderedDict([(v[0],k) for k, v in enumerate(tracks)])
            words = c.execute('SELECT DISTINCT word FROM lyrics').fetchall()
            word_hash = OrderedDict([(v[0],k) for k, v in enumerate(words)])

        # (n_items, n_words)
        A = triplet2sparse(triplet, track_hash, word_hash)

        return A.tocsr(), track_hash, word_hash


class MSDArtist(MFTask):
    """"""
    def __init__(self, n_components, db_fn, n_iter, alg='plsa'):
        """"""
        super(MSDArtist, self).__init__(n_components, db_fn, n_iter, alg)
        A, track_hash_a, artist_hash = self.read(db_fn['artist'])
        T, track_hash_t, tag_hash = LastFMTag.read(db_fn['tag'])

        # filter artist matrix
        keep_dim_track = [track_hash_a[t] for t in track_hash_t.keys()]
        A_t = A[keep_dim_track]
        keep_dim_artist = np.array(A_t.sum(axis=0) > 0).ravel()
        A_t = A_t[:, keep_dim_artist] # artist-track subset which has tags

        self.A = A_t.T.dot(T) # (n_artist, n_tags)
        self.doc_hash = track_hash_t
        self.term_hash = tag_hash

        self.artist_id = np.array(np.argmax(A_t, axis=1)).ravel()
        self.artist_hash = artist_hash
        self.artists = np.array(artist_hash.keys())[keep_dim_artist]

    @classmethod
    def read(cls, db_fn):
        """"""
        # Load Artist data
        with sqlite3.connect(db_fn) as conn:
            c = conn.cursor()
            artist = [r[1] for r in c.execute(
                      'SELECT * FROM artists').fetchall()]
            triplet = [
                (r[0], artist[r[1]], 1) for r
                in c.execute('SELECT track_key, artist_id FROM tracks')]

            artist_filter = set(map(lambda x:x[1], triplet))
            artist_hash = OrderedDict(
                [(v, k) for k, v in enumerate(artist_filter)])
            tracks = set(map(lambda x:str(x[0]), triplet))
            track_hash = OrderedDict([(v, k) for k, v in enumerate(tracks)])

        A = triplet2sparse(triplet, track_hash, artist_hash)
        return A.tocsr(), track_hash, artist_hash

    @property
    def data(self):
        """"""
        if self.U is None:
            U_reassign = None
        else:
            U_reassign = self.U[self.artist_id]

        return {
            'item_factors':U_reassign,
            'term_factors':self.V,
            'tids':self.doc_hash.keys(),
            'tags':self.term_hash.keys(),
            'artists':self.artists
        }


class MSDTempo(BaseInternalTask):
    """"""
    def __init__(self, n_components, db_fn, n_iter, alg='plsa'):
        """"""
        super(MSDTempo, self).__init__(n_components, db_fn, n_iter, alg)
        self.A, self.doc_hash = self.read(db_fn)
        self.gmm = GaussianMixture(self.k, max_iter=n_iter)

    @classmethod
    def read(cls, db_fn):
        """"""
        data = joblib.load(db_fn)
        A = data['item_factors']
        tids = data['tids']
        tids_hash = OrderedDict([(v,k) for k,v in enumerate(tids)])
        return A, tids_hash

    def process(self):
        """"""
        self.gmm.fit(self.A)
        self.U = self.gmm.predict_proba(self.A)

