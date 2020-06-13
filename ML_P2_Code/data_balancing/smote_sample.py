"""
Contains a function to form balanced training set using SMOTE
"""
from imblearn.over_sampling import SMOTE


def get_smote_data(train):
    """ over-sample rows using SMOTE technique
    @param train: pandas dataframe storing data used for training
    @return: train: pandas dataframe storing balanced train data
    """

    sm = SMOTE()

    # resample the dataset
    train, output = sm.fit_sample(train.drop(columns='output'),
                                  train['output'])

    train['output'] = output

    return train
