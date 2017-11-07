from itertools import chain
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureBlockSelector(BaseEstimator, TransformerMixin):

    def __init__(self, blocks):
        self.blocks = blocks
        self.keep_dims = list(chain.from_iterable(
            [range(i*32, (i+1)*32) for i in blocks]))

    def transform(self, X):
        """"""
        if X.shape[-1] != 160:
            raise ValueError('[ERROR] only supports 160 dimension!')
        return X[:, self.keep_dims]

    def fit(self, X, y=None):
        """"""
        return self


def print_cm(cm, labels, hide_zeroes=False,
             hide_diagonal=False, hide_threshold=None):
    """by zachguo's github"""
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    line = '\n'
    line += "    " + empty_cell
    for label in labels:
        line += "%{0}s".format(columnwidth) % label
    line += '\n'
    # Print rows
    for i, label1 in enumerate(labels):
        line += "    %{0}s".format(columnwidth) % label1
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            line += cell
        line += '\n'
    return line
