import logging
import numpy as np

# set up a logger, at least for the ImportError
from custom_estimators.multi_labels import MultiLabelBinarizer
from custom_estimators.preprocessors import enrich_category

model_logr = logging.getLogger(__name__)
model_logr.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelBinarizer, FunctionTransformer
from sklearn_pandas import DataFrameMapper
from fastFM import als, sgd

from sklearn.pipeline import Pipeline

experiment_dict = \
    {
        'expt_1': {
            'note': 'default Linear regression with regularization',
            'name': 'default Linear regression with regularization',
            'pl': Pipeline([('mapper', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                        ('item', LabelBinarizer(sparse_output=True))],
                                                       input_df=True, sparse=True, )),
                            ('ridge', Ridge())])
        },
        'expt_2': {
            'note': 'default Linear regression with regularization',
            'name': 'default Linear regression with regularization',
            'pl': Pipeline([('mapper', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                        ('item', LabelBinarizer(sparse_output=True))],
                                                       input_df=True, sparse=True, )),
                            ('ridge', Ridge())])
        },

        'expt_3': {
            'note': 'default linear regression with regularization and grid search cv',
            'name': 'linear regression with reg and cv',
            'pl': GridSearchCV(Pipeline([('mapper', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                                     ('item', LabelBinarizer(sparse_output=True))],
                                                                    input_df=True, sparse=True, )),
                                         ('ridge', Ridge())]),
                               param_grid={'ridge__alpha': np.arange(4.0, 5.0, 0.05)}, scoring='neg_mean_squared_error')
        },
        'expt_4': {
            'note': 'latent factor model',
            'name': 'latent factor model with als',
            'pl': Pipeline([('mapper', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                        ('item', LabelBinarizer(sparse_output=True))],
                                                       input_df=True, sparse=True, )),
                            ('als', als.FMRegression(n_iter=100, rank=2, l2_reg=5))])
        },
        'expt_5': {
            'note': 'latent factor model',
            'name': 'latent factor model',
            'pl': GridSearchCV(Pipeline([('mapper', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                        ('item', LabelBinarizer(sparse_output=True))],
                                                       input_df=True, sparse=True, )),
                            ('als', als.FMRegression(n_iter=100, rank=3, l2_reg=5))]),
                            param_grid={'als__rank': [3,5,10]}, scoring='neg_mean_squared_error')
        },
        'expt_6': {
            'note': 'latent factor model with category',
            'name': 'latent factor model',
            'pl': Pipeline([('categorize', FunctionTransformer(enrich_category, validate=False)),
                            ('mapper', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                         ('item', LabelBinarizer(sparse_output=True)),
                                                         ('category', MultiLabelBinarizer(sparse_output=True))],
                                                        input_df=True, sparse=True)),
                            ('als', als.FMRegression(n_iter=100, rank=5, l2_reg=5))])
        },

        'expt_7': {
            'note': 'latent factor model with category',
            'name': 'FM with categories',
            'pl': Pipeline([('categorize', FunctionTransformer(enrich_category, validate=False)),
                            ('mapper', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                        ('item', LabelBinarizer(sparse_output=True)),
                                                        ('category', MultiLabelBinarizer(sparse_output=True))],
                                                       input_df=True, sparse=True)),
                            ('als', als.FMRegression(n_iter=100, rank=3, l2_reg=20))])
        },
        'expt_8': {
            'note': 'ridge with categories with gridsearch on alpha',
            'name': 'ridge with categories with gridsearch on alpha',
            'pl': GridSearchCV(Pipeline([('categorize', FunctionTransformer(enrich_category, validate=False)),
                            ('mapper', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                        ('item', LabelBinarizer(sparse_output=True)),
                                                        ('category', MultiLabelBinarizer(sparse_output=True))],
                                                       input_df=True, sparse=True)),
                            ('ridge', Ridge())]),
                               param_grid={'ridge__alpha': np.arange(4.0, 4.4, 0.1)}, scoring='neg_mean_squared_error')
        },
        'expt_9': {
            'note': 'default linear regression with regularization and grid search cv',
            'name': 'linear regression with reg and cv',
            'pl': GridSearchCV(Pipeline([('mapper', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                                     ('item', LabelBinarizer(sparse_output=True))],
                                                                    input_df=True, sparse=True, )),
                                         ('ridge', Ridge(solver='svd'))]),
                               param_grid={'ridge__solver': ['svd', 'lsqr', 'sparse_cg', 'sag']}, scoring='neg_mean_squared_error')
        },
        'expt_10': {
            'note': 'ridge with categories',
            'name': 'ridge with categories',
            'pl':Pipeline([('categorize', FunctionTransformer(enrich_category, validate=False)),
                                         ('mapper', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                                     ('item', LabelBinarizer(sparse_output=True)),
                                                                     ('category',
                                                                      MultiLabelBinarizer(sparse_output=True))],
                                                                    input_df=True, sparse=True)),
                                         ('ridge', Ridge(alpha=4.2))])
        },
    }
