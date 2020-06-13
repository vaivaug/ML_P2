from sklearn import preprocessing
from pandas import DataFrame


def get_standardized_data(data):
    # save the column names and output column
    column_names_list = list(data.columns)
    stored_output = data['output']

    # standardize data without using the output column
    standardized_data = preprocessing.scale(data.drop(columns='output'))
    data = DataFrame(standardized_data)

    # add the output column to the dataset again
    data['output'] = stored_output
    data.columns = column_names_list
    return data
