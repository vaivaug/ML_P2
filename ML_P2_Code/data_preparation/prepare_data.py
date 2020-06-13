import pandas as pd

pixels_in_image = 60*60


def get_clean_dataframes(filedir):
    """
    :param filedir: String value, file path to the 'binary' or 'multi' folder
    :return: clean input data stored in pandas dataframe

    Read and clean the data
    """
    # the given file path has X_test, X_train, Y_train files
    X_train = pd.read_csv(filedir + '/X_train.csv', header=None)
    Y_train = pd.read_csv(filedir + '/Y_train.csv', header=None)
    X_test = pd.read_csv(filedir + '/X_test.csv', header=None)

    X_train = get_X_df_with_column_names(X_train)
    X_test = get_X_df_with_column_names(X_test)
    Y_train.columns = ['output']

    if check_X_all_numeric_values(X_train) and check_X_all_numeric_values(X_test):
        print('All values are numeric')

    # drop rows if the colour histograms are incorrect
    # no rows are dropped in binary case and in multi case
    X_train, Y_train = get_df_correct_colours(X_train, Y_train)

    return X_train, Y_train, X_test


def get_X_df_with_column_names(data_df):
    """
    :param data_df: dataset with no column names
    :return: dataset with column names
    """

    # create list of integer values for gradient column names
    columns = list(range(1, 901))

    # create list of column names for random normal distribution variables
    random_var_normal_dist = [f'nd_variable_{i}' for i in range(1, 17)]
    # append this list to 'columns' list
    columns.extend(random_var_normal_dist)

    # create list of column names for colour histograms. Join into one list
    colour_histograms = list(f'red_{i}' for i in range(1, 17))
    colour_histograms.extend(f'green_{i}' for i in range(1, 17))
    colour_histograms.extend(f'blue_{i}' for i in range(1, 17))

    columns.extend(colour_histograms)

    data_df.columns = columns
    return data_df


def check_X_all_numeric_values(data_df):
    """
    :param data_df: data in pandas dataframe
    :return: boolean value, 'True' if all values are numeric

    Check that all values in the given dataframe are numeric
    """

    for (column_name, column_data) in data_df.iteritems():
        if not pd.to_numeric(data_df[column_name], errors='coerce').notnull().all():
            return False

    return True


def get_df_correct_colours(data_X, data_Y=None):
    """Drop rows which have incorrect number of pixels for colur bins
    """
    # 16 bins for all 3 colours have to be non-negative
    # all 3 groups of 16 bins add to pixels_in_image

    for index, row in data_X.iterrows():
        if not colours_correct(row):
            data_X.drop(index, inplace=True)
            if data_Y is not None:
                data_Y.drop(index, inplace=True)

    return data_X, data_Y


def colours_correct(row):
    """
    :param row: one row of X data
    :return: boolean value, True if all colour histogram values are positive and add up to pixels_in_image.
             return False otherwise.
    """
    global pixels_in_image

    sum_red_bins = 0
    sum_green_bins = 0
    sum_blue_bins = 0

    for i in range(1, 17):

        # check all index i values are positive
        if row['red_{}'.format(i)] < 0 or row['green_{}'.format(i)] < 0 or row['blue_{}'.format(i)] < 0:
            return False

        # update the bins sum for all 3 colours
        sum_red_bins += row['red_{}'.format(i)]
        sum_green_bins += row['green_{}'.format(i)]
        sum_blue_bins += row['blue_{}'.format(i)]

    if sum_red_bins != pixels_in_image or sum_green_bins != pixels_in_image or sum_blue_bins != pixels_in_image:
        return False

    return True


def get_number_of_classes(data_df):

    df = data_df.groupby('output')['output'].nunique()
    return df.size

'''
def get_no_rows_output_nan(X_train, Y_train):
    print('length before nan are removed:', len(X_train))

    X_train['output'] = Y_train
    # drop rows where 'output' is NaN
    X_train = X_train.dropna(subset=['output'], inplace=False)
    Y_train = X_train['output']
    Y_train.columns = ['output']
    X_train = X_train.drop(columns='output')
    print('after nan are removed: ', len(X_train))
    print(X_train)
    print(Y_train)
    return X_train, Y_train
'''