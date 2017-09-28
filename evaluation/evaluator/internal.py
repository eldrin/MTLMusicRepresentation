import cPickle as pkl

from sklearn.externals import joblib
import namedtupled

from model.model import Model
from utils.misc import get_in_shape

# Evaluator for all internal tasks
class InternalTaskEvaluator:
    """"""
    def __init__(self, fn):
        """"""
        # load configuration for model
        model_state = joblib.load(fn)
        self.config = namedtupled.map(model_state['config'])

        # variable set up
        self.tasks = self.config.task
        self.sr = self.config.hyper_parameters.sample_rate
        self.hop_sz = hop_sz # in second
        self.in_shape = get_in_shape(self.config)

        # load valid id for each task
        split = namedtupled.reduce(self.config.paths.meta_data.splits)
        self.valid_ids = {
            k:joblib.load(v)['valid']
            for k,v in split.iteritems()
        }
        self.path_map = pkl.load(open(self.config.paths.path_map))

        # load model builder
        self.model = Model(config, feature_layer)


# Task specific evaluation helpers
def evaluate_tag(song_ids, model, path_map, config, top_k=20):
    """"""
    # variable set up
    tasks = config.task
    sr = config.hyper_parameters.sample_rate
    hop_sz = hop_sz # in second
    in_shape = get_in_shape(self.config)

    # load tag factor
    tag_factor_model = joblib.load(
        os.path.join(
            config.paths.meta_data.root,
            config.paths.meta_data.targets.tag
        )
    )

    V = tag_factor_model['tag_factors']
    U = tag_factor_model['item_factors']
    tids_hash = {
        tid:j for j,tid in enumerate(tag_factor_model['tids'])
    }

    # inference song level
    for song_id in song_ids:
        y = U[tids_hash[song_id]]
        o, c, f = _get_feature_and_prob(
            path_map[song_id], y, model, hop_sz, in_shape[-1])

        pred = o.dot(V)
        pred_tag_ix = np.argsort(pred)[-top_k:][::-1]
        # true_tag_ix = something
        # TODO: finish this up
