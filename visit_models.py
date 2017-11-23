import logging
import numpy as np
import pandas as pd
import utils
from fastFM import als, sgd, mcmc

# set up a logger, at least for the ImportError
from custom_estimators.preprocessors import enrich_category, to_user_item_matrix

model_logr = logging.getLogger(__name__)
model_logr.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import FunctionTransformer
from custom_estimators.multi_labels import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier

from sklearn.pipeline import Pipeline, FeatureUnion

mapper_1 = DataFrameMapper([('user', LabelBinarizer(sparse_output=True))],
                        input_df=True, sparse=True)

mapper_2 = DataFrameMapper([('item', LabelBinarizer(sparse_output=True))],
                        input_df=True, sparse=True)

experiment_dict = \
    {
        'expt_1': {
            'note': 'default logistic regression',
            'name': 'default logistic regression',
            'pl': Pipeline([('categorize', FunctionTransformer(enrich_category, validate=False)),
                            ('mapper', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                         ('item', LabelBinarizer(sparse_output=True)),
                                                         ('category', MultiLabelBinarizer(sparse_output=True))],
                                                        input_df=True, sparse=True)),
                            ('lgst', LogisticRegression(solver='lbfgs'))])
        },
        'expt_2': {
            'note': 'naive latent factor model',
            'name': 'naive latent factor model',
            'pl': Pipeline([('mapper', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                        ('item', LabelBinarizer(sparse_output=True))],
                                                        input_df=True, sparse=True, )),
                            ('sgd', sgd.FMClassification(n_iter=100, rank=3))])
        },
        'expt_3': {
            'note': 'naive latent factor model',
            'name': 'naive latent factor model',
            'pl': GridSearchCV(Pipeline([('mapper', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                        ('item', LabelBinarizer(sparse_output=True))],
                                                       input_df=True, sparse=True, )),
                                        ('als', sgd.FMClassification(n_iter=100, rank=2))]),
                            param_grid={'als__rank': [2,10,50]}, scoring='accuracy')
        },
        'expt_4': {
            'note': 'naive latent factor model',
            'name': 'naive latent factor model',
            'pl': GridSearchCV(Pipeline([('mapper', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                                     ('item', LabelBinarizer(sparse_output=True))],
                                                                    input_df=True, sparse=True, )),
                                         ('als', sgd.FMClassification(n_iter=100, rank=2))]),
                               param_grid={'als__n_iter': [100, 200, 500]}, scoring='accuracy')
        },
        'expt_5': {
            'note': 'naive latent factor model',
            'name': 'naive latent factor model',
            'pl': GridSearchCV(Pipeline([('mapper', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                                     ('item', LabelBinarizer(sparse_output=True))],
                                                                    input_df=True, sparse=True, )),
                                         ('als', sgd.FMClassification(n_iter=100, rank=2,l2_reg=1))]),
                               param_grid={'als__l2_reg': [1,5,10,50]}, scoring='accuracy')
        },
        'expt_6': {
            'note': 'fm with category',
            'name': 'fm visit with category',
            'pl': Pipeline([('categorize', FunctionTransformer(enrich_category, validate=False)),
                            ('mapper', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                        ('item', LabelBinarizer(sparse_output=True)),
                                                        ('category', MultiLabelBinarizer(sparse_output=True))],
                                                       input_df=True, sparse=True)),
                            ('als', als.FMClassification(n_iter=100, rank=3))])
        },
        'expt_7': {
            'note': 'naive latent factor model with feature union',
            'name': 'naive latent factor model with feature union',
            'pl': Pipeline([('feature', FeatureUnion([("user", mapper_1),("item", mapper_2)])),
                            ('als', als.FMClassification(n_iter=100, rank=4))])
        },
        'expt_8': {
            'note': 'ridge classifier',
            'name': 'ridge classifier',
            'pl': Pipeline([('categorize', FunctionTransformer(enrich_category, validate=False)),
                            ('mapper', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                        ('item', LabelBinarizer(sparse_output=True)),
                                                        ('category', MultiLabelBinarizer(sparse_output=True))],
                                                       input_df=True, sparse=True)),
                            ('lgst', RidgeClassifier())])
        },
        'expt_9': {
            'note': 'fm with category',
            'name': 'fm visit with category',
            'pl': Pipeline([('categorize', FunctionTransformer(enrich_category, validate=False)),
                            ('mapper', DataFrameMapper([('user', LabelBinarizer(sparse_output=True)),
                                                        ('item', LabelBinarizer(sparse_output=True)),
                                                        ('category', MultiLabelBinarizer(sparse_output=True))],
                                                       input_df=True, sparse=True))])
        },


    }
