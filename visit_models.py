import logging
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn_pandas import DataFrameMapper

# set up a logger, at least for the ImportError
model_logr = logging.getLogger(__name__)
model_logr.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline



experiment_dict = \
    {

    }
