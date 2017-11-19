import logging
import numpy as np
import pandas as pd
import utils
from sklearn.preprocessing import LabelBinarizer
from sklearn_pandas import DataFrameMapper

# set up a logger, at least for the ImportError
model_logr = logging.getLogger(__name__)
model_logr.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')

from sklearn.preprocessing import FunctionTransformer
from custom_estimators.multi_labels import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from sklearn.pipeline import Pipeline


def enrich_category(X):
    items = X['item'].tolist()
    cat_dict = utils.get_categories(items)

    cat = [cat_dict[i] for i in items]
    X.loc[:, 'category'] = pd.Series(cat).values

    return X


experiment_dict = \
    {
        'expt_1': {
            'note': 'default logistic regression',
            'name': 'default logistic regression',
            'pl': Pipeline([('categorize', FunctionTransformer(enrich_category, validate=False)),
                            ('mapper1', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                         ('item', LabelBinarizer(sparse_output=True)),
                                                         ('category', MultiLabelBinarizer(sparse_output=True))],
                                                        input_df=True, sparse=True)),
                            ('lgst', LogisticRegression(solver='lbfgs'))])
        }

    }
