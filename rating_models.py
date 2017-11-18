import logging
import numpy as np

# set up a logger, at least for the ImportError
model_logr = logging.getLogger(__name__)
model_logr.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn_pandas import DataFrameMapper

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
                               param_grid={'ridge__alpha': np.arange(4.0, 5.0, 0.1)}, scoring='neg_mean_squared_error')
        }

    }
