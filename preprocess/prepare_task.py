import os
import traceback

from utils.misc import get_py_logger
from helper import *
from processor.internal_task import *
import fire

TASK_PROCESS = {
    'tag': {'db_fn': '/mnt/msd/lastfm_tags.db',
            'process': LastFMTag},
    'pref': {'db_fn': '/mnt/msd/user_song_preference.db',
             'process': MSDTaste},
    'bpm': {'db_fn': '/mnt/msd/meta_data/msd_tempo.dat.gz',
            'process': MSDTempo},
    'lyrics': {'db_fn':'/mnt/msd/meta_data/mxm_dataset.db',
               'process': MXMLyrics},
    'artist': {'db_fn': {'artist':'/mnt/msd/user_song_preference.db',
                         'tag':'/mnt/msd/lastfm_tags.db'},
               'process': MSDArtist},
    'year': {'db_fn':'/mnt/msd/tracks_per_year.txt',
             'process':MSDYear},
    'cdr_tag': {'db_fn': '/mnt/msd/meta_data/CDR_genre.pkl',
                'process': CDRGenre},
}


def subsample_n_split(fn, tids, subsample=0.05, split=0.1):
    """"""
    n_subsample = int(len(tids) * subsample)
    tids_sub = np.random.choice(tids, n_subsample, replace=False)

    n_valid = int(len(tids_sub) * split)
    joblib.dump(
        {'train':tids_sub[n_valid:],'valid':tids_sub[:n_valid]}, fn)

def get_intersection():
    """"""
    # load all tids from each internal tasks...
    tids = {}
    for task, info in TASK_PROCESS.iteritems():
        print('load {} info...'.format(task))
        if task == 'artist':
            db_fn = info['db_fn']['artist']
        else:
            db_fn = info['db_fn']
        tids[task] = set(info['process'].read(db_fn)[1].keys())

    # return intersection
    return list(set.intersection(*tids.values()))

def prepare_task(task, out_fn, subset_fn=None, k=50, n_iter=100, subsample=0.05, split=0.1):
    """"""
    global TASK_PROCESS
    # prepare logger
    basename = os.path.basename(out_fn).split('.')[0]
    dirname = os.path.dirname(out_fn)

    log_fn = os.path.join(dirname, basename) + '.log'
    logger = get_py_logger(log_fn)

    split_fn = os.path.join(dirname, basename) + '.split'

    if subset_fn is not None:
        # list of tid (string)
        tids_sub = pkl.load(open(subset_fn))
    else:
        tids_sub = subset_fn

    # process
    try:
        task_proc = TASK_PROCESS[task]
        logger.info('Initiate...')
        processor = task_proc['process'](
            n_components=k, db_fn=task_proc['db_fn'], n_iter=n_iter,
            tids=tids_sub)

        logger.info('Proccess data...')
        processor.process()

        logger.info('Saving data...')
        processor.save(out_fn)
        subsample_n_split(
            split_fn, processor.doc_hash.keys(), subsample, split)

        logger.info('Task preparation complete!')

    except Exception as e:
        logger.error('{}'.format(e))
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    fire.Fire(prepare_task)
