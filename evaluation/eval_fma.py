import os
import sys
sys.path.append('/home/ubuntu/workbench/End2EndMusic/')

import numpy as np
import pandas as pd

from sklearn.externals import joblib
import librosa

import h5py

from utils import preemphasis,load_check_point,prepare_X
from model import build_2dconv_clf_deep, get_debug_funcs

from collections import OrderedDict
from tqdm import tqdm

from lasagne.nonlinearities import softmax

GENRES = OrderedDict(
    [
        ('Blues',0),
        ('Classical',1),
        ('Country',2),
        ('Easy Listening',3),
        ('Electronic',4),
        ('Experimental',5),
        ('Folk',6),
        ('Hip-Hop',7),
        ('Instrumental',8),
        ('International',9),
        ('Jazz',10),
        ('Old-Time / Historic',11),
        ('Pop',12),
        ('Rock',13),
        ('Soul-RnB',14),
        ('Spoken',15)
    ]
)

def prepare_data_small(model_fn,dset_fn,out_act,n_out=200,hop_sz=2):
    """ Extract feature from GTZAN and save X,Y to h5 """

    # load CNN model
    net = build_2dconv_clf_deep(
        (None,2,513,431),
        out_act=out_act,
        n_out=n_out
    )
    net['out'] = load_check_point(
        net['out'],model_fn
    )
    funcs = get_debug_funcs(net)

    # load preprocessor 
    preproc = joblib.load(
        '/mnt/msd/meta_data/sclr_log10_stft.dat.gz'
    )

    # open file to write & create dataset
    h5_fn = os.path.join(
        '/mnt/bulk2/datasets/FMA/',
        dset_fn
    )
    hf = h5py.File(h5_fn,'w')

    feature = hf.create_dataset(
        'X',shape=(25000,512)
    )
    target = hf.create_dataset(
        'y',shape=(25000,16)
    )
    labels = hf.create_dataset(
        'labels',
        data=np.array(
            GENRES.keys(),
            dtype='S'
        )
    )

    # open meta data
    meta_data = pd.DataFrame.from_csv(
        '/mnt/bulk2/datasets/FMA/fma_metadata/tracks.csv',
        header=1,
        sep=','
    )

    db_root = '/mnt/bulk2/datasets/FMA/fma_medium/'
    rnd_idx = np.random.choice(25000,25000,replace=False)
    i = 0
    for root,dirs,files in os.walk(db_root):
        for f in tqdm(files):
            if f != 'checksums' and f != 'README.txt':
                fn = os.path.join(root,f)
                track_id = int(f.split('.')[0])
                genre = meta_data.loc[str(int(track_id))]['genre_top']

                x,sr = librosa.load(fn,mono=False,sr=22050)

                if x.ndim==1:
                    x = np.repeat(x[None,:],2,axis=0)

                if x.shape[1] / sr < 5:
                    continue

                X = []
                for j in xrange(0,30,hop_sz):
                    x_chunk = x[:,sr*j:sr*(j+5)]

                    if x_chunk.shape[-1] < sr*5:
                        continue

                    # X.append(
                    #     prepare_X(x_chunk,preproc)[0]
                    # )
                    X.append(
                        funcs['features'](
                            prepare_X(x_chunk,preproc)
                        )[-1]
                    )

                # XX = funcs['features'](np.array(X))[-1]
                # feature[rnd_idx[i]] = XX.mean(axis=0)
                feature[rnd_idx[i]] = np.mean(X,axis=0)
                target[rnd_idx[i],GENRES[genre]] = 1
                i+=1

if __name__ == '__main__':

    model = '/mnt/bulk2/models/2DConvTempoGMM_300k_param.npz'
    out_fn = 'medium_top_genre_tempogmm_dataset.h5'
    prepare_data_small(
        model_fn=model, dset_fn=out_fn, out_act=softmax, n_out=20
    )
