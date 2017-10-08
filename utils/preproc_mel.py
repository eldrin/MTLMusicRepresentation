import os
import numpy as np
from functools import partial

import theano
from lasagne import layers as L
import cPickle as pkl

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from model.custom_layer import STFTLayer, MelSpecLayer
from preprocess.prepare_task import get_intersection
from utils.misc import load_audio, pmap

from tqdm import tqdm

SONG_ROOT = '/mnt/msd/songs/'
MEL_ROOT = '/mnt/msdmel/'
SR = 22050

# build CudaMel
l_in = L.InputLayer((None, 2, None))
l_stft = STFTLayer(
    L.ReshapeLayer(l_in, ([0], [1], [2], 1)),
                   n_ch=2, n_fft=1024, hop_size=256, log_amplitude=False)
l_mel = MelSpecLayer(l_stft, sr=22050, n_fft=1024, log_amplitude=True)
out = L.get_output(l_mel, deterministic=True)
melspec = theano.function([l_in.input_var], out)

def prepare_scaler():
    tids = get_intersection()
    path_map = pkl.load(open('/mnt/msd/MSD_to_path.pkl'))
    path = filter(lambda x:x[1] is not None,
                  map(lambda t:
                      (t,
                       os.path.join(SONG_ROOT, path_map[t])
                       if t in path_map else None),
                      tids)
                 )

    # init StandardScaler
    sclr = StandardScaler()

    fns = []
    for i in tqdm(range(0, len(path), 16)):
        p = path[slice(i, i+16)]
        Y = map(lambda x:x[0],
                pmap(partial(load_audio, sr=SR),
                     map(lambda x:x[1], p), n_jobs=8))
        Y = filter(lambda x:x[1] is not None, zip(p, Y))
        # y, sr = load_audio(path[slc], sr=SR)

        for tid, y in Y:
            fns.append(tid)
            M = melspec(y.astype(np.float32)[None, :, :])
            for m in M[0]:
                sclr.partial_fit(m)

    joblib.dump(sclr, '/mnt/msd/meta_data/sclr_dbscale_mel.dat.gz')
    joblib.dump(fns, '/mnt/msd/meta_data/intersection_tids.dat.gz')

def save_mels(fns):
    for i in tqdm(range(0, len(fns), 8), ncols=80):
        p = fns[slice(i, i+16)]
        Y = map(lambda x:x[0],
                pmap(partial(load_audio, sr=SR),
                     map(lambda x:x[1], p), n_jobs=8))
        Y = filter(lambda x:x[1] is not None, zip(p, Y))

        for tid, y in Y:
            M = melspec(y.astype(np.float32)[None, :, :])
            np.save(os.path.join(MEL_ROOT, tid[0]+'.npy'), M)


if __name__ == "__main__":
    fns = joblib.load('/mnt/msd/meta_data/intersection_tids.dat.gz')
    save_mels(fns)
