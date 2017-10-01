#!/usr/bin/python
import os
import time
import json
from itertools import combinations

import pandas as pd

from evaluator.external import MLEvaluator, RecSysEvaluator
from helper import print_cm

import h5py

import fire

class Evaluate(object):
    """"""
    def __init__(self, out_dir=None, comb_lim=2, preproc=None, n_jobs=-1,
                 keep_metrics=['accuracy_score', 'recall@40_score', 'r2_score']):
        """"""
        if out_dir is None:
            out_dir = os.getcwd()

        self.out_dir = out_dir
        self.preproc = preproc
        self.comb_lim = comb_lim
        self.n_jobs = -1
        self.keep_metrics = keep_metrics

    def internal(self, path):
        """"""
        pass

    def external(self, path):
        """"""
        # parsing path
        # path can be configuration file, file path, or directory
        # config version and filepath version will be developped first
        if os.path.isdir(path):
            # evaluation all files in dir
            raise NotImplementedError(
                '[ERROR] directory input is not yet supported!')
        else:
            if os.path.splitext(path)[-1] == '.json':

                # config file case
                keep_res = {}
                config = json.load(open(path))
                for task, paths in config.iteritems():
                    if task == 'root':
                        continue
                    else:
                        keep_res[task] = {}
                        avail_feats = filter(
                            lambda x:x[1] is not None, paths.items())

                        for r in xrange(1, self.comb_lim+1):
                            for comb in combinations(avail_feats, r):

                                # make each combination's file name id
                                comb_key = '-'.join(map(lambda x:x[0], comb))
                                out_fn = comb_key + '-{}.txt'.format(task)

                                # and file path list
                                fns = map(
                                    lambda x:
                                    os.path.join(config['root'], x[1]),
                                    comb
                                )
                                # instantiate evaluator and evaluate
                                evaluator = self._get_evaluator(task, fns)
                                res = evaluator.evaluate()

                                for metric in self.keep_metrics:
                                    keep_res[task][comb_key] = \
                                    {metric:res[metric]}

                                # save & print
                                self._save(out_fn, self._print(res))

                # save summary in json
                json.dump(keep_res,
                          open(os.path.join(
                              self.out_dir, 'summary.json'), 'w'))

                # save summary in individual csv for metrics
                for metric in self.keep_metrics:
                    score = {}
                    for task, scores in keep_res.iteritems():
                        score[task] = {}
                        for int_task, int_scores in scores.iteritems():
                            score[task][int_task] = int_scores[metric]

                    pd.DataFrame(score).to_csv(
                        os.path.join(
                            self.out_dir,
                            'summary.{}.csv'.format(metric)))

            elif os.path.splitext(path)[-1] == '.h5': # feature files case

                # prepare path
                path = path.split('-')
                out_fn = '-'.join(
                    [os.path.splitext(os.path.basename(fn))[0] for fn in path])
                out_fn += '{}.txt'

                # instantiate evaluator & evaluate
                # TODO: fix this temporary workaround
                with h5py.File(path[0]) as hf:
                    if hf.attrs['type'] == 'recommendation':
                        task = 'recsys'
                    else:
                        task = 'no_recsys'

                evaluator = self._get_evaluator(task, path)
                res = evaluator.evaluate()

                # save & print
                self._save(out_fn, self._print(res))

            else:
                raise ValueError(
                    '[ERROR] only confing (json) and feature (h5) files are\
                    supported!')

    def _save(self, fn, lines):
        """"""
        path = os.path.join(self.out_dir, fn)
        with open(path, 'w') as f:
            f.write(lines)

    @staticmethod
    def _print(res):
        """"""

        lines = ''
        if res['classification_report'] is not None:
            lines += '=================  Classification Report =================='
            lines += '\n'
            lines += res['classification_report']
            lines += '\n'
            lines += '====================  Confusion Matrix ===================='
            lines += print_cm(*res['confusion_matrix'])
            lines += '\n'

        score_key = filter(lambda k: 'score' in k, res.keys())[0]
        score_name = score_key.split('_score')[0]
        if score_name == 'accuracy':
            lines += '{}: {:.2%}'.format(score_name.title(), res[score_key])
        else:
            lines += '{}: {:.4f}'.format(score_name.title(), res[score_key])

        lines += '\n'
        lines += 'Time spent: {:.2f} (sec)'.format(res['time'])
        print
        print(lines)
        return lines

    def _get_evaluator(self, task, fns):
        """"""
        if task == 'recsys':
            evaluator = RecSysEvaluator(
                fns, self.preproc, self.n_jobs)
        else:
            evaluator = MLEvaluator(
                fns, self.preproc, self.n_jobs)

        return evaluator

if __name__ == "__main__":
    fire.Fire(Evaluate)
