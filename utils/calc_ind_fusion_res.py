import os
import pandas as pd
import subprocess
import fire
import numpy as np


id_map = {
    0: 'self',
    1: 'year',
    2: 'bpm',
    3: 'pref',
    4: 'tag',
    5: 'lyrics',
    6: 'cdr_tag',
    7: 'artist_tag'
}

metric_map = {
    'GTZAN': 'Accuracy',
    'Ballroom': 'Accuracy',
    'IRMAS_SUB': 'Accuracy',
    'FMA_SUB': 'Accuracy',
    'EmoVal': 'R2',
    'EmoAro': 'R2',
    'ThisIsMyJam': 'Ap@20'
}


def get_res(run_id, task_ids, dset, name=None):
    """"""
    root = '/mnt/bulk2/exp_res/features/'
    fn = [
        os.path.join(root, 'S{:d}'.format(task_id),
                     'conv_2d_{}_50_intersc_runS{:d}_{}_feature.h5'.format(
                          id_map[task_id], task_id, dset))
        for task_id in task_ids
    ]
    fn = '-'.join(fn)

    out_root = 'data/results/no_share/'
    if name is None:
        out_fn = 'run{:d}_no_share_res_{}_{}.txt'.format(
            int(run_id),
            '_'.join([id_map[task_id] for task_id in task_ids]), dset)
    else:
        out_fn = name
    out_path = os.path.join(out_root, out_fn)

    cmd = ['python', '-m',
           'evaluation.evaluate',
           '--out-dir', os.path.join(out_root, 'raw'),
           '--n-trial', '5',
           'external', fn]

    out = subprocess.check_output(cmd).splitlines()
    out = filter(lambda l: metric_map[dset] in l, out)
    out = map(lambda l: l.replace('%', ''), out)
    out = map(
        lambda l: float(l.replace('{}: '.format(metric_map[dset]), '')), out)
    out = pd.DataFrame(data={metric_map[dset]: out}).to_csv(out_path)


def read_plan(fn):
    """"""
    df = pd.read_csv(fn, sep='\t', header=None, index_col=None)
    return df


def main(plan_fn, dset, runs=None):
    """"""
    if runs is None:
        runs = [i for i in range(80) if i != 61 and i != 72]
    else:
        if isinstance(runs, (list, tuple)):
            assert any([isinstance(r, (int, float)) for r in runs])
        elif isinstance(runs, (float, int)):
            runs = [runs]
        elif isinstance(runs, str):
            assert isinstance(float(runs), float)
        else:
            raise ValueError(
                '[ERROR] only available number of list of the numbers!')

    d = read_plan(plan_fn)
    for i, run in zip(runs, d.iloc[runs].as_matrix()):
        tasks = np.where(run)[0]
        get_res(i, tasks, dset)


if __name__ == "__main__":
    # fire.Fire(get_res)
    fire.Fire(main)
