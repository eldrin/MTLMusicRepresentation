import numpy as np

def sample_matcher_idx(req_tids, all_tids):
    """"""
    res = []
    n_positive = len(req_tids) / 2
    for p1 in req_tids[:n_positive]:
        res.append((p1, p1, 1))

    neg_cands = req_tids[n_positive:]
    for n1, swap in zip(neg_cands, np.random.choice(2, len(neg_cands))):
        n2 = np.random.choice(len(all_tids))
        while n2 == n1:
            n2 = np.random.choice(len(all_tids))
        if swap:
            res.append((n2, n1, 0))
        else:
            res.append((n1, n2, 0))
    return res

