# -*- coding: utf-8 -*-
"""
@author: YaYaB <https://github.com/YaYaB>
@brief:
"""
import os
import numpy as np
import time
from pydd.solver import GenericSolver
from pydd.models import MLP
from pydd.connectors import SVMConnector
from pydd.utils import os_utils
from sklearn import datasets, metrics, model_selection, preprocessing

# Parameters
seed = 1337
np.random.seed(seed)  # for reproducibility
n_classes = 10
repository = "/tmp/pydd_test"
if not os.path.exists(repository):
    os.makedirs(repository)
params = {"repository": repository,"port": 8085, "nclasses": n_classes, "gpu": True}
split_params = {"test_size": 0.2, "random_state": seed}

# create dataset
X, y = datasets.load_digits(n_class=n_classes, return_X_y=True)
X = preprocessing.StandardScaler().fit_transform(X)
xtr, xte, ytr, yte = model_selection.train_test_split(X, y, **split_params)

# create and save train.svm and test.svm
tr_f = os.path.abspath('x_train.svm')
te_f = os.path.abspath('x_test.svm')
datasets.dump_svmlight_file(xtr, ytr, tr_f)
datasets.dump_svmlight_file(xte, yte, te_f)

# Define models and class weights
clf = MLP(**params)

solver = GenericSolver(iterations=500, solver_type="SGD", base_lr=0.01, gamma=0.1, stepsize=30, momentum=0.9, snapshot=200)
# one class weight value for each class
class_weights = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1]

train_data, test_data = SVMConnector(path=tr_f), SVMConnector(path=te_f)

logs = clf.fit(train_data, validation_data=[test_data], solver=solver, class_weights=class_weights, batch_size=128)

params = {"repository": repository, "port": 8085, "nclasses": n_classes, "gpu": True, "resume": True}
clf = MLP(**params)
solver = GenericSolver(iterations=1000, solver_type="SGD", base_lr=0.01, gamma=0.1, stepsize=30, momentum=0.9)
logs = clf.fit(train_data, validation_data=[test_data], solver=solver, class_weights=class_weights, batch_size=128)

yte_pred = clf.predict(test_data)
report = metrics.classification_report(yte, yte_pred)
print(report)

# remove saved svm files
os_utils._remove_files([tr_f, te_f])

