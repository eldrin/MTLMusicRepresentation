import os
import numpy as np
from scipy import sparse as sp
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans

from preprocess.processor.internal_task import MSDArtist
from utils.misc import pmap, load_audio

from tqdm import tqdm
import cPickle as pkl
import sox

A, t_hash, a_hash = MSDArtist.read('/mnt/msd/user_song_preference.db')
tids = t_hash.keys()
artists = a_hash.keys()

path_map = pkl.load(open('/mnt/msd/MSD_to_path.pkl'))
song_root = '/mnt/msd/songs/'

def get_mfcc(fn):
    m = None
    try:
        sr = 22050
        y, _ = load_audio(fn, sr=sr)
        if y is None: raise ValueError
        y = y.mean(axis=0)
        m = librosa.feature.mfcc(y, sr, n_mfcc=23)
    except Exception as e:
        print(e)
    finally:
        return m

def get_sup_vec(gmm):
    mu = gmm.means_.flatten()
    sd = gmm.covariances_.flatten()
    return np.concatenate([mu, sd])

res = []
for a, artist in tqdm(zip(A.T, artists), ncols=80):

    fns = []
    for tid in tqdm(sp.find(a)[1], ncols=80):
        if tids[tid] in path_map:
            fn = os.path.join(song_root,path_map[tids[tid]])
            if os.path.exists(fn):
                fns.append(fn)
            else:
                continue
        else:
            continue

    M = filter(
        lambda x:x is not None,
        pmap(get_mfcc, fns, n_jobs=16)
    )

    if len(M)==0:
        res.append((artist, None))
        continue

    mfccs = np.concatenate(M,axis=-1).T
    if mfccs.shape[0] > 10000:
        mfccs = mfccs[np.random.choice(mfccs.shape[0], 10000, replace=False)]
    elif mfccs.shape[0] < 1000:
        res.append((artist, None))
        continue

    # (n_steps, n_bins)
    mdl = GaussianMixture(n_components=4, covariance_type='diag')
    # mdl = MiniBatchKMeans(n_clusters=8)
    res.append(
        (artist, mdl.fit(mfccs))
    )

X = np.array(
    map(get_sup_vec,
        map(
            lambda x:x[1],
            filter(
                lambda x:x[1] is not None, res)
        )
    )
)


