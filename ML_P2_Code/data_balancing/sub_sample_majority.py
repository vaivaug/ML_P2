"""
Contains a function to form balanced training set by sub-sampling 'background' samples
"""
import pandas as pd


def get_sub_sampled_background_data_binary(train):
    """ Downsample majority class i.e. 'background' images
    @param train: pandas dataframe storing the training data
    @return: train: pandas dataframe storing the training data containing all 'seal' samples
        from initial train data plus the same amount of 'background' samples
    """

    # create two datasets containing only 'seal' and only 'background' samples
    seal = train[train['output'] == 'seal']
    background = train[train['output'] == 'background']

    # change 'background' set to a downsampled set of 'background' samples
    background = background.sample(n=len(seal), replace=False)

    # combine both sets
    train = pd.concat([seal, background])

    # shuffle the order of training samples
    train = train.sample(n=len(train)).reset_index(drop=True)

    return train


def get_sub_sampled_data_multi(train):
    """ Downsample majority class i.e. 'background' images
    @param train: pandas dataframe storing the training data
    @return: train: pandas dataframe storing the training data containing all different seal samples
        from initial train data plus the same amount of 'background' samples
    """
    class_names_list = train.output.unique()

    # the smallest class
    juvenile = train[train['output'] == 'juvenile']

    balanced_train = pd.DataFrame()

    for name in class_names_list:
        # create a dataset containing samples only from class 'name'
        class_samples = train[train['output'] == name]

        # change 'name' rows to a sub-sampled set of 'name' samples
        class_samples = class_samples.sample(n=len(juvenile), replace=False)
        balanced_train = balanced_train.append(class_samples)

    # shuffle the order of training samples
    balanced_train = balanced_train.sample(n=len(balanced_train)).reset_index(drop=True)

    return balanced_train


def get_mixed_balancing_data(train):
    """ Downsample majority class i.e. 'background' images
        Over-sample different seal images
        @param train: pandas dataframe storing the training data
        @return: train: pandas dataframe storing the training data containing all different seal samples
            from initial train data plus the same amount of 'background' samples
        """

    class_names_list = train.output.unique()
    balanced_train = pd.DataFrame()

    # the biggest class of seals
    whitecoat = train[train['output'] == 'whitecoat']

    # sub-sample the biggest class i.e. 'background'
    background = train[train['output'] == 'background']
    background = background.sample(n=len(whitecoat), replace=False)

    balanced_train = balanced_train.append(background)

    for name in class_names_list:
        if name == 'background':
            continue
        # create a dataset containing samples only from class 'name'
        class_samples = train[train['output'] == name]

        # change 'name' rows to oversampled set of 'name' samples
        class_samples = class_samples.sample(n=len(whitecoat), replace=True)
        balanced_train = balanced_train.append(class_samples)

    # shuffle the order of training samples
    balanced_train = balanced_train.sample(n=len(balanced_train)).reset_index(drop=True)

    return balanced_train
