import utils
import pandas as pd
import numpy as np
import scipy.sparse as sp

from collections import defaultdict


def enrich_category(X):
    items = X['item'].tolist()
    cat_dict = utils.get_categories(items)

    cat = [cat_dict[i] for i in items]
    X.loc[:, 'category'] = pd.Series(cat).values

    return X


def to_user_item_matrix(X):
    grouped = X.groupby(X['user']).aggregate(lambda x: set(tuple(x)))
    return grouped


def enrich_category_count(X):
    items = X['item'].tolist()
    users = X['user'].tolist()
    cat_dict = utils.get_categories(items)
    nested = list(cat_dict.values())
    cat = list(set([item for sublist in nested for item in sublist]))

    print(cat)
    print(len(cat))
    user_cat_count = defaultdict(lambda : defaultdict(int))

    for u,i in list(zip(users,items)):
        i_cat = cat_dict[i]
        for cat in i_cat:
            cat_dict[u][cat] += 1

    for u, i in list(zip(users, items)):
        cat_vec = np.zeros(len(cat_dict))
        i_cat = cat_dict[i]


