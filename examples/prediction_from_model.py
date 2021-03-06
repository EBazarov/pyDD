# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import os
from pydd.utils import os_utils
from pydd.models import MLP
from pydd.connectors import SVMConnector, ArrayConnector
from pydd.solver import GenericSolver
from sklearn import datasets, preprocessing, model_selection, metrics


# Parameters
test_size = 0.2
seed = 1337
sname = 'predict_from_model'
model_repo = os.path.abspath('trained_model')
params = {'host': 'localhost', 'port': 8085, 'nclasses': 10, 'layers': [100, 100]}

# We make sure model repo does not exist
if os.path.exists(model_repo):
    os_utils._remove_dirs([model_repo])
os_utils._create_dirs([model_repo])

# create dataset
X, Y = datasets.load_digits(return_X_y=True)
X = preprocessing.StandardScaler().fit_transform(X)
xtr, xte, ytr, yte = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

# create and save train.svm and test.svm
tr_f = os.path.abspath('x_train.svm')
te_f = os.path.abspath('x_test.svm')
datasets.dump_svmlight_file(xtr, ytr, tr_f)
datasets.dump_svmlight_file(xte, yte, te_f)

# create connectors
xtr_svm, xte_svm = SVMConnector(tr_f), SVMConnector(te_f)
optimizer = GenericSolver(solver_type='SGD', iterations=500, base_lr=0.1, snapshot=100)

# train model and delete model
clf = MLP(sname=sname, repository=model_repo, **params)
clf.fit(xtr_svm, validation_data=[xte_svm, xtr_svm], solver=optimizer)
del clf

params = {'host': 'localhost', 'port': 8085, 'nclasses': 10, 'finetuning': True, 'template': None}
clf = MLP(sname=sname, repository=model_repo, **params)

# predict and show metrics
ytr_pred, yte_pred = clf.predict(xtr_svm), clf.predict(xte_svm)
report = metrics.classification_report(yte, yte_pred)
print(report)

# remove create files and folders
os_utils._remove_files([tr_f, te_f])
os_utils._remove_dirs([model_repo])