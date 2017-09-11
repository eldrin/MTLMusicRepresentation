import os
import tempfile
import subprocess
from functools import partial

import sox
from tqdm import tqdm

from misc import load_audio, pmap

import fire

def get_file_info(fn):
    """"""
    err = None
    sr, dur, mono = None, None, None
    try:
        sr = sox.file_info.sample_rate(fn)
        dur = sox.file_info.duration(fn)
        n_ch = sox.file_info.duration(fn)
        mono = True if n_ch == 2 else False
    except sox.SoxiError as sie:
        err = sie
    except sox.SoxError as se:
        err = se
    except KeyboardInterrupt as kbe:
        print('[ERROR] User interrupted process')
        raise kbe
    except Exception as e:
        err = e
    finally:
        return {
            'sr':sr, 'dur':dur, 'mono':mono, 'error':err
        }

def _checker(inputs, audio_root):
    tid, f = inputs
    fn = os.path.join(audio_root, f)
    return tid, get_file_info(fn)

def check_all_info(audio_root, path_map):
    """"""
    audio_info = pmap(
        partial(_checker, audio_root=audio_root),
        path_map.items(),
        n_jobs=4, verbose=True
    )
    return audio_info


def _resample_n_save(fn, sr, out_fn=None):
    """"""
    if out_fn is None:
        out_fn = fn

    cmd = ['sox', fn, '-r', '{:d}'.format(sr), out_fn]
    subprocess.call(cmd)


def resample_all_22k(fns):
    """"""
    pmap(
        partial(_resample_n_save, sr=22050),
        fns, n_jobs=8, verbose=True
    )
