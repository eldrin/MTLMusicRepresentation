import os

import sox
from tqdm import tqdm

from misc import load_audio, pmap

def get_file_info(fn):
    """"""
    err = None
    sr, dur, mono = None, None, None
    try:
        sr = sox.file_info.sample_rate(fn)
        dur = sox.file_info.duration(fn)
        n_ch = sox.file_info.duration(fn)
        mono = False if n_ch == 2 else True
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

def check_all_info(audio_root, path_map):
    """"""
    # audio_info = {}
    # for tid, f in tqdm(path_map.iteritems()):
    #     fn = os.path.join(audio_root, f)
    #     audio_info[tid] = get_file_info(fn)
    def _checker(tid, f):
        fn = os.path.join(audio_root, f)
        return tids, get_file_info(fn)

    audio_info = pmap(
        _checker, tqdm(path_map.iteritems()),
        n_jobs=8
    )
    return audio_info
