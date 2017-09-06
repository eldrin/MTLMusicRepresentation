import os
import sys
sys.path.append('/home/ubuntu/workbench/End2EndMusic/')

from itertools import chain,repeat

import numpy as np

from sklearn.externals import joblib
import librosa

from lasagne.nonlinearities import softmax

import h5py

from utils import preemphasis,load_check_point,prepare_X
from model import build_2dconv_clf_deep, get_debug_funcs

from collections import OrderedDict
from tqdm import tqdm

GENRES = OrderedDict(
    [
        ('ChaChaCha',('ChaChaCha')),
        ('Jive',('Jive')),
        ('Quickstep',('Quickstep')),
        ('Rumba',('Rumba-American','Rumba-International','Rumba-Misc')),
        ('Samba',('Samba')),
        ('Tango',('Tango')),
        ('VienneseWaltz',('VienneseWaltz')),
        ('Waltz',('Waltz')),
    ]
)

DB_ROOT = '/mnt/bulk2/datasets/Ballroom/BallroomData/'

def prepare_data(model_fn,dset_fn,out_act,n_out=200,hop_sz=1):
    """ Extract feature from GTZAN and save X,Y to h5 """

    # set up some parameters
    win_sz = 1024
    hop_sz = 256
    sr = 22050
    length = 2.5

    remaining = int(sr*length) % hop_sz
    sig_len = int(sr*length) - remaining


    # load CNN model
    net = build_2dconv_clf_deep(
        (None,2,sig_len),
        out_act=out_act,
        n_out=n_out
    )
    net['out'] = load_check_point(
        net['out'],model_fn
    )
    funcs = get_debug_funcs(net)

    # get dataset meta info
    n_files = 0
    fns = []
    genres = GENRES.values()
    dirs = chain.from_iterable(
        repeat(x,1) if isinstance(x,str) else x
        for x in genres
    )

    for d in dirs:
        for root,_,files in os.walk(os.path.join(DB_ROOT,d)):
            n_files += len(files)
            fns.extend(
                [
                    (
                        os.path.join(root,fn),
                        GENRES.keys().index(d.split('-')[0])
                    )
                    for fn in files
                ]
            )

    n_dim = 512 * 2 # fc dim * 2 (avg / std)
    n_cls = len(GENRES)
    print(
        'num samples : {:d}'.format(n_files),
        'feat dim : {:d}'.format(n_dim),
        'num classes : {:d}'.format(n_cls)
    )

    # open file to write & create dataset
    h5_fn = os.path.join(
        '/mnt/bulk2/datasets/Ballroom/',
        dset_fn
    )
    hf = h5py.File(h5_fn,'w')

    feature = hf.create_dataset(
        'X',shape=(n_files,n_dim)
    )
    target = hf.create_dataset(
        'y',shape=(n_files,n_cls)
    )
    output = hf.create_dataset(
        'Z',shape=(n_files,n_out)
    )
    labels = hf.create_dataset(
        'labels',
        data=np.array(
            GENRES.keys(),
            dtype='S'
        )
    )

    # rnd_idx = np.random.choice(n_files,n_files,replace=False)
    rnd_idx = xrange(n_files)
    i = 0
    for f,genre in tqdm(fns):
        fn = os.path.join(root,f)

        x,sr = librosa.load(fn,sr=sr)
        x = np.repeat(x[None,:],2,axis=0)

        X = []
        Y = []
        for j in xrange(0,30,hop_sz):
            start = int(sr*j)
            slc = slice(start,start + sig_len)
            x_chunk = x[:,slc][None,:,:]

            if x_chunk.shape[-1] < sig_len:
                continue

            # X_ = prepare_X(x_chunk,preproc)
            X_ = x_chunk

            # X.append(
            #     prepare_X(x_chunk,preproc)[0]
            # )
            X.append(funcs['features'](X_)[-1])
            Y.append(funcs['predict'](X_))

        # XX = funcs['features'](np.array(X))[-1]
        # feature[rnd_idx[i],:n_dim/2] = np.array(XX).mean(axis=0)
        # feature[rnd_idx[i],n_dim/2:] = np.array(XX).std(axis=0)
        feature[rnd_idx[i],:n_dim/2] = np.mean(X,axis=0)
        feature[rnd_idx[i],n_dim/2:] = np.std(X,axis=0)

        target[rnd_idx[i],genre] = 1

        output[rnd_idx[i]] = np.mean(Y,axis=0)

        i+=1

    hf.create_dataset(
        'fns',
        data=np.array(
            map(lambda x:x[0],fns),
            dtype='S'
        )
    )

if __name__ == '__main__':

    # model = '/mnt/bulk2/models/2DConvTopicNMF_50k_param.npz'
    model = '/mnt/bulk2/models/2DConvTempoGMM_short_14k_param.npz'
    out_fn = 'tempogmm_dataset_14k.h5'
    prepare_data(model_fn=model, dset_fn=out_fn, out_act=softmax, n_out=20)
