import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, classification_report

import fire

def easy_eval_clf(X, y, verbose=True):
	""""""
	# init model (SVC)
	# clf = SVC(kernel='linear')
	clf = SGDClassifier(n_jobs=-1)

	if X.shape[1] < X.shape[0] * 2:
		preproc = StandardScaler()
	else: # empirical dimension reduction for extreme cases
		preproc = PCA(n_components=X.shape[0] * 2, whiten=True)
	
	pl = Pipeline([('preproc', preproc), ('clf', clf)])

	# fire cross validation
	y_ = cross_val_predict(pl, X, y, cv=10)

	# simple evaluation
	acc = accuracy_score(y, y_)
	cr = classification_report(y, y_)

	if verbose:
		print(cr)
		print
		print('Accuracy : {:.2%}'.format(acc))
	return acc

class EasyEval:
	""""""
	def __init__(self, n_trial=1):
		""""""
		self.n_trial = n_trial

	def eval(self):
		for n in xrange(self.n_trial):
			pass # do evaluation here

if __name__ == '__main__':
	pass

