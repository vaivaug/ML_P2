from sklearn.feature_selection import f_classif, chi2, mutual_info_classif

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import matplotlib.pyplot as plt


def get_univariate_f_test_features(data):

    data_target = data['output']
    data_features = data.drop(columns='output')

    # Univariate feature selection with F-test for feature scoring
    # We use the default selection function to select the k
    # most significant features
    selector = SelectKBest(f_classif, k=20)
    selector.fit(data_features, data_target)
    cols = selector.get_support(indices=True)

    data_selected_features = data_features.iloc[:, cols]
    print(data_selected_features)
    return data_selected_features

