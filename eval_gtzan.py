import os
import sys
sys.path.append('/home/ubuntu/workbench/End2EndMusic/')

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
        ('blues',0),
        ('classical',1),
        ('country',2),
        ('disco',3),
        ('hiphop',4),
        ('jazz',5),
        ('metal',6),
        ('pop',7),
        ('reggae',8),
        ('rock',9)
    ]
)

def prepare_data(model_fn,dset_fn,out_act,n_out=200,hop_sz=1):
    """ Extract feature from GTZAN and save X,Y to h5 """

    # set up some parameters
    win_sz = 1024
    hop_sz = 256
    sr = 22050
    length = 2.5

    remaining = int(sr*length)%hop_sz
    sig_len = int(sr*length)-remaining


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

    # open file to write & create dataset
    h5_fn = os.path.join(
        '/mnt/bulk2/datasets/GTZAN/',
        dset_fn
    )
    hf = h5py.File(h5_fn,'w')

    feature = hf.create_dataset(
        'X',shape=(1000,512)
    )
    target = hf.create_dataset(
        'y',shape=(1000,10)
    )
    prob = hf.create_dataset(
        'Z',shape=(1000,n_out)
    )
    labels = hf.create_dataset(
        'labels',
        data=np.array(
            GENRES.keys(),
            dtype='S'
        )
    )

    db_root = '/mnt/bulk2/datasets/GTZAN/genres/'
    # rnd_idx = np.random.choice(1000,1000,replace=False)
    rnd_idx = xrange(1000)
    i = 0
    fns = []
    for root,dirs,files in os.walk(db_root):
        fns.extend(files)
        for f in tqdm(files):
            fn = os.path.join(root,f)
            genre = f.split('.')[0]

            x,sr = librosa.load(fn,sr=sr)
            x = np.repeat(x[None,:],2,axis=0)

            X,Y = [],[]
            for j in xrange(0,30,hop_sz):
                start = int(sr*j)
                slc = slice(start,start + sig_len)
                x_chunk = x[:,slc][None,:,:]

                if x_chunk.shape[-1] < sig_len:
                    continue

                X_ = x_chunk

                # X.append(
                #     prepare_X(x_chunk,preproc)[0]
                # )
                X.append(funcs['features'](X_)[-1])
                Y.append(funcs['predict'](X_))

            # XX = funcs['features'](np.array(X))[-1]
            # feature[rnd_idx[i]] = XX.mean(axis=0)
            feature[rnd_idx[i]] = np.mean(X,axis=0)
            target[rnd_idx[i],GENRES[genre]] = 1
            prob[rnd_idx[i]] = np.mean(Y,axis=0)
            i+=1

    hf.create_dataset(
        'fns',
        data=np.array(
            fns,dtype='S'
        )
    )

if __name__ == '__main__':

    # model = '/mnt/bulk2/models/2DConvTopicNMF_45k_param.npz'
    # out_fn = 'nmf_dataset.h5'
    model = '/mnt/bulk2/models/2DConvTempoGMM_short_34k_param.npz'
    out_fn = 'tempogmm_dataset.h5'
    prepare_data(model_fn=model, out_act=softmax, dset_fn=out_fn, n_out=20)
