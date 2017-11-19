# -*- coding: UTF-8 -*-
__author__ = "Josh Montague"
__license__ = "MIT License"

#
# This module defines a number of helper functions.
# It is a modification of the original file by Josh Montague
#

from datetime import datetime
import logging
import sys
import gzip
import pandas as pd
import numpy as np
import random
from collections import defaultdict

# set up a logger
util_logr = logging.getLogger(__name__)
util_logr.setLevel(logging.DEBUG)
util_sh = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')
util_sh.setFormatter(formatter)
util_logr.addHandler(util_sh)


def short_name(model):
    """Return a simplified name for this model. A bit brittle."""
    # for a single model, this will work
    name = model.__class__.__name__
    try:
        if hasattr(model, 'steps'):
            # pipeline
            name = '-'.join([pair[0] for pair in model.steps])
        elif hasattr(model, 'best_estimator_'):
            if hasattr(model.estimator, 'steps'):
                # gridsearchcv
                name = 'gscv_' + '-'.join([x[0] for x in model.estimator.steps])
            elif hasattr(model.estimator, 'estimators'):
                # votingclassifier
                name = 'gscv_vc_' + '-'.join([x[0] for x in model.estimator.estimators])
        elif hasattr(model, 'base_estimator_'):
            # bagging
            name = 'bag_' + short_name(model.base_estimator)
    except AttributeError as e:
        util_logr.info('utils.short_name() couldnt generate quality name')
        # for a single model, this will work
        name = model.__class__.__name__
        util_logr.info('falling back to generic name={}'.format(name))
    return name


def create_submission(predictions, sub_name, data):
    users = data['user'].tolist()
    items = data['item'].tolist()
    now = datetime.utcnow().strftime('%Y-%m-%dT%H%M%S')
    submission_name = '-'.join(sub_name.split())

    with open('submissions/{}_{}.txt'.format(now, submission_name), 'w') as f:
        f.write('userID-businessID,prediction\n')
        for u, i, p in list(zip(users, items, predictions)):
            f.write(str(u) + "-" + str(i) + "," + str(p) + "\n")
    return True


def read_gz(f):
    for l in gzip.open(f):
        yield eval(l)


def load_train_data(model):
    X, y = [], []
    for l in read_gz("data/train.json.gz"):
        user, business = l['userID'], l['businessID']
        X.append((user, business))
        if model == 'Visit':
            y.append(1)
        else:
            y.append(l['rating'])
    return pd.DataFrame(X, columns=['user', 'item']), np.array(y)


def load_test_data(model):
    test = []
    for l in open("submissions/pairs_{}.txt".format(model)):
        if not l.startswith("userID"):
            u, i = l.strip().split('-')
            test.append((u, i))

    return pd.DataFrame(test, columns=['user', 'item'])


def get_categories(business):
    business_cat = defaultdict(set)
    b_set = set(business)

    for l in read_gz("data/train.json.gz"):
        bid, categories = l['businessID'], l['categories']
        if bid in b_set:
            business_cat[bid].update(categories)

    return business_cat


def sample_negatives(X, y):
    users, items = X['user'].tolist(), X['item'].tolist()
    pair_set = set(list(zip(users, items)))

    n = len(y)

    u_set, i_set = list(set(users)), list(set(items))
    u_n, i_n = len(u_set), len(i_set)

    X_new = list(zip(users,items))
    y_new = list(y) + [0] * n
    while n > 0:
        u = random.randint(0, u_n - 1)
        i = random.randint(0, i_n - 1)
        if (users[u], items[i]) not in pair_set:
            X_new.append((users[u], items[i]))
            n -= 1

    shuffled = list(zip(X_new, y_new))
    random.shuffle(shuffled)
    X_nnew, y_new = zip(*shuffled)
    X_new = [list(elem) for elem in X_nnew]
    return pd.DataFrame(X_new, columns=['user', 'item']), np.array(y_new)
