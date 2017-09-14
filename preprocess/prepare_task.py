import os

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
               'process': MSDArtist}
}


def subsample_n_split(fn, tids, subsample=0.05, split=0.1):
    """"""
    n_subsample = int(len(tids) * subsample)
    tids_sub = np.random.choice(tids, n_subsample, replace=False)

    n_valid = int(len(tids_sub) * split)
    joblib.dump(
        {'train':tids_sub[n_valid:],'valid':tids_sub[:n_valid]}, fn)


def prepare_task(task, out_fn, k=50, n_iter=100, subsample=0.05, split=0.1):
    """"""
    global TASK_PROCESS
    # prepare logger
    basename = os.path.basename(out_fn).split('.')[0]
    dirname = os.path.dirname(out_fn)

    log_fn = os.path.join(dirname, basename) + '.log'
    logger = get_py_logger(log_fn)

    split_fn = os.path.join(dirname, basename) + '.split'

    # process
    try:
        task_proc = TASK_PROCESS[task]
        logger.info('Initiate...')
        processor = task_proc['process'](
            n_components=k, db_fn=task_proc['db_fn'], n_iter=n_iter)

        logger.info('Proccess data...')
        processor.process()

        logger.info('Saving data...')
        processor.save(out_fn)
        subsample_n_split(
            split_fn, processor.doc_hash.keys(), subsample, split)

        logger.info('Task preparation complete!')

    except Exception as e:
        logger.error('{}'.format(e))
        raise e

if __name__ == "__main__":
    fire.Fire(prepare_task)
