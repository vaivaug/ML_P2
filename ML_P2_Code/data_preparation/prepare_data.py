import pandas as pd


def get_clean_dataframes(filedir):
    """
    :param filedir: String value, file path to the 'binary' or 'multi' folder
    :return: clean input data stored in pandas dataframe

    """
    # the given file path has X_test, X_train, Y_train files
    X_train = pd.read_csv(filedir + '/X_train.csv', header=None)
    Y_train = pd.read_csv(filedir + '/Y_train.csv', header=None)
    X_test = pd.read_csv(filedir + '/X_test.csv', header=None)

    return X_train, Y_train, X_test


