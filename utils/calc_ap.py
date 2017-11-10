import os
import subprocess
import pandas as pd
from tqdm import tqdm
import fire


def get_rec_ap(n):
    """"""
    root = '/mnt/bulk2/exp_res/features/{}/'.format(n)
    for root, dirs, files in os.walk(root):
        fn = filter(lambda f: 'ThisIsMyJam' in f, files)[0]
    fn = os.path.join(root, fn)

    if not os.path.exists(fn):
        raise IOError('[ERROR] no file for {}'.format(n))

    out_root = 'data/results/rec_ap/'
    out_fn = os.path.join(out_root, '{}.txt'.format(n))

    cmd = ['python', '-m',
           'evaluation.evaluate',
           '--out-dir', out_root,
           '--n-trial', '5',
           'external', fn]

    out = subprocess.check_output(cmd).splitlines()
    out = filter(lambda l: 'Ap' in l, out)
    out = map(lambda l: float(l.replace('Ap@10: ', '')), out)
    out = pd.DataFrame(data={'Ap@10': out}).to_csv(out_fn)


def main():
    """"""
    targets = [str(j) for j in range(62, 80)]
    targets += ['S{:d}'.format(i) for i in range(8)]

    for t in tqdm(targets, ncols=80):
        get_rec_ap(t)


if __name__ == "__main__":
    """"""
    fire.Fire(main)
