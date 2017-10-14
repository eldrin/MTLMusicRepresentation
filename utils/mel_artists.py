import numpy as np
import cPickle as pkl

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer

from tqdm import tqdm
from preprocess.prepare_task import TASK_PROCESS
from preprocess.processor.internal_task import MSDArtist

K = 1024 + 512


def get_feat(x):
    x = x[0].mean(axis=0)
    dx = x[1:] - x[:-1]
    y = np.concatenate([x[:-1], dx], axis=1)
    return y


def sample_feat(y, n_samples=K):
    if n_samples > y.shape[0]:
        n_samples = y.shape[0]
    z = y[np.random.choice(y.shape[0], n_samples, replace=False)]
    return z


def train_kmeans(fns):
    # make universal frame model
    kms = MiniBatchKMeans(n_clusters=K)

    for f in tqdm(fns, ncols=80):
        x = np.load(f)
        y = get_feat(x)
        kms.partial_fit(y)

    return kms


def main():
    # fns = pkl.load(open('/mnt/msd/MSD_MTL_SUB_MEL_FNS.pkl'))
    # kms = train_kmeans(fns)

    from sklearn.externals import joblib
    kms = joblib.load('data/artist_mel_kmeans.dat.gz')

    # prepare data
    tids = pkl.load(open('/mnt/msd/MSD_MTL_SUB_MEL_TIDS.pkl'))
    # tid_set = set(tids)

    # load artist precessor
    msdartist = MSDArtist(
        50, TASK_PROCESS['artist']['db_fn'], 100, tids)
    msdartist._prepare_data()

    # load 'raw' song-artist map
    B, track_hash, artist_hash = MSDArtist.read(
        '/mnt/msd/user_song_preference.db')

    # filter out targets with tids input
    # C = B[map(lambda x:x[1],
    #           filter(lambda x:x[0] in tid_set, track_hash.items()))]
    # now, C contains only target track's artist info
    cnt = CountVectorizer(vocabulary=map(str, range(K)))
    inv_doc_hash = {v: k for k, v in msdartist.doc_hash.iteritems()}
    artists = set(msdartist.artist_id)
    artist_mel_bof = {}
    for artist in tqdm(artists, ncols=80):
        songs = np.where(msdartist.artist_id == artist)[0]
        Z = []
        for s in songs:
            tid = inv_doc_hash[s]
            fn = '/mnt/msdmel/{}.npy'.format(tid)
            x = np.load(fn)
            y = get_feat(x)
            Z.extend(kms.predict(y).tolist())
        c = cnt.transform(map(str, Z)).sum(axis=0)
        artist_mel_bof[artist] = np.array(c).ravel() / float(c.sum())

    return msdartist, kms, B, track_hash, artist_hash, artist_mel_bof


if __name__ == "__main__":
    msdartist, kms, B, track_hash, artist_hash, artist_mel_bof = main()
