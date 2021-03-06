#!/usr/bin/python

import os
import json
from itertools import combinations

import pandas as pd

from evaluator.external import MLEvaluator, RecSysEvaluator
from helper import print_cm

import h5py

import fire


class Evaluate(object):
    """"""
    def __init__(
            self, out_dir=None, out_fn=None, comb_lim=2, preproc=None, n_trial=1, n_jobs=-1,
            keep_metrics=['accuracy_score', 'recall@40_score', 'r2_score']):
        """"""
        if out_dir is None:
            out_dir = os.getcwd()

        self.out_dir = out_dir
        self.out_fn = out_fn
        self.preproc = preproc
        self.comb_lim = comb_lim
        self.n_jobs = n_jobs
        self.n_trial = n_trial
        self.keep_metrics = keep_metrics

    def internal(self, path):
        """"""
        pass

    def external(self, path):
        """"""
        # parsing path
        # path can be configuration file, file path, or directory
        # config version and filepath version will be developped first

        # feature file directory case ====================================
        if os.path.isdir(path):
            keep_scores = {
                'classification': 'accuracy_score',
                'regression': 'r2_score',
                'recommendation': 'ap'
            }
            # get file names
            fns = {}
            for root, dirs, files in os.walk(path):
                fns = dict(map(
                    lambda f:
                    (self._get_ext_task_name(f), os.path.join(root, f)),
                    files
                ))

            # evaluate all feature h5py in the dir
            res = {}
            for task, fn in fns.iteritems():
                res[task] = []
                with h5py.File(fn) as hf:
                    task_type = hf.attrs['type']

                for i in range(self.n_trial):
                    evaluator = self._get_evaluator(task_type, [fn])
                    r = filter(
                        lambda x: keep_scores[task_type] in x[0],
                        evaluator.evaluate().items()
                    )[0][1]
                    if task_type == 'classification':
                        res[task].append(r * 100)
                    else:
                        res[task].append(r)

            # save result as txt file
            if self.out_fn is None:
                out_fn = 'result.txt'
            else:
                out_fn = self.out_fn

            out_fn = os.path.join(self.out_dir, out_fn)
            pd.DataFrame(res).to_csv(out_fn)

        # evlauation config file case ====================================
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
                            lambda x: x[1] is not None, paths.items())

                        for r in xrange(1, self.comb_lim+1):
                            for comb in combinations(avail_feats, r):

                                # make each combination's file name id
                                comb_key = '-'.join(map(lambda x: x[0], comb))
                                out_fn = comb_key + '-{}.txt'.format(task)

                                # and file path list
                                fns = map(
                                    lambda x:
                                    os.path.join(config['root'], x[1]),
                                    comb
                                )

                                # instantiate evaluator and evaluate
                                evaluator = self._get_evaluator(task, fns)

                                # multiple times
                                res = [
                                    evaluator.evaluate()
                                    for i in range(self.n_trial)
                                ]
                                res_df = pd.DataFrame(res)
                                scores = res_df.filter(like='score')

                                # get stat and put in container
                                for metric in self.keep_metrics:
                                    if metric in scores:
                                        keep_res[task][comb_key] = \
                                            {
                                                metric: {
                                                    'avg': scores[metric].mean(),
                                                    'std': scores[metric].std()
                                                }
                                            }

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
                            score[task][int_task+'.avg'] = int_scores[metric]['avg']
                            score[task][int_task+'.std'] = int_scores[metric]['std']

                    pd.DataFrame(score).to_csv(
                        os.path.join(
                            self.out_dir,
                            'summary.{}.csv'.format(metric)))

            # single feature file case ====================================
            elif os.path.splitext(path)[-1] == '.h5':
                # prepare path
                path = path.split('-')
                out_fn = '-'.join(
                    [os.path.splitext(os.path.basename(fn))[0] for fn in path])
                out_fn += '{}.txt'

                # TODO: currently ad-hoc. need to fix it later
                with h5py.File(path[0]) as hf:
                    task = hf.attrs['type']
                for i in range(self.n_trial):
                    evaluator = self._get_evaluator(task, path)
                    res = evaluator.evaluate()

                    # save & print
                    out_str = self._print(res)
                    # self._save(out_fn, out_str)
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

        score_key = filter(lambda k: 'score' in k, res.keys())

        for score in score_key:
            score_name = score.split('_score')[0]
            if score_name == 'accuracy':
                lines += '{}: {:.2%}'.format(score_name.title(),
                                             res[score])
            else:
                lines += '{}: {:.4f}'.format(score_name.title(),
                                             res[score])
            lines += '\n'

        lines += 'Time spent: {:.2f} (sec)'.format(res['time'])
        print
        print(lines)
        return lines

    def _get_evaluator(self, task, fns):
        """"""
        if task == 'recommendation':
            evaluator = RecSysEvaluator(
                fns, self.preproc, self.n_jobs)

        elif task == 'regression':
            if self.preproc is None:
                print(
                    '[WARNING] found no preprocessor. \
                    set z-scaler for default...'
                )
                _preproc = self.preproc
                self.preproc = 'standardize'
                evaluator = MLEvaluator(
                    fns, self.preproc, self.n_jobs)
                self.preproc = _preproc
            else:
                evaluator = MLEvaluator(
                    fns, self.preproc, self.n_jobs)

        elif task == 'classification':
            evaluator = MLEvaluator(
                fns, self.preproc, self.n_jobs)
        else:
            raise ValueError('[ERROR] {} is not supported task!'.format(task))

        return evaluator

    def _get_ext_task_name(self, fn):
        """"""
        ext_tasks = ['GTZAN', 'Ballroom', 'EmoAroStatic', 'EmoValStatic',
                     'FMA_SUB', 'IRMAS_SUB', 'ThisIsMyJam']
        for task in ext_tasks:
            if task in fn:
                return task
        return None


if __name__ == "__main__":
    fire.Fire(Evaluate)
