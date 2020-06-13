"""
Contains a function to form balanced training set by over-sampling seal samples
"""
import pandas as pd


def get_over_sampled_seal_data_binary(train):
    """ Oversample minority class i.e. seal samples
    @param train: pandas dataframe storing the data used for training
    @return: train: pandas dataframe storing the data used for training containing all 'background' samples
        from initial train data plus the same amount of 'seal' samples some of which are repeated multiple times
    """

    # create two datasets containing only 'seal' and only 'background' samples
    seal = train[train['output'] == 'seal']
    background = train[train['output'] == 'background']

    # change 'seal' rows to a oversampled set of 'seal' samples
    seal = seal.sample(n=len(background), replace=True)

    # combine both sets
    train = pd.concat([seal, background])

    # shuffle the order of training samples
    train = train.sample(n=len(train)).reset_index(drop=True)

    return train


def get_over_sampled_seal_data_multi(train):
    """ over sample 4 minority classes (4 different seal classes)
    :param train: pandas dataframe containing imbalanced 5 class data
    :return: train: balanced pandas dataframe
    """
    class_names_list = train.output.unique()

    # the biggest class
    background = train[train['output'] == 'background']

    balanced_train = pd.DataFrame()

    for name in class_names_list:
        # create a dataset containing samples only from class 'name'
        class_samples = train[train['output'] == name]
        # change 'name' rows to a oversampled set of 'name' samples
        class_samples = class_samples.sample(n=len(background), replace=True)
        balanced_train = balanced_train.append(class_samples)

    # shuffle the order of training samples
    balanced_train = balanced_train.sample(n=len(balanced_train)).reset_index(drop=True)

    return balanced_train
