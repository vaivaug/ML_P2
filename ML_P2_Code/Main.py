from data_preparation.prepare_data import get_clean_dataframes

binary_filedir = '../binary'
multi_filedir = '../multi'

# clean data
X_train, Y_train, X_test = get_clean_dataframes(binary_filedir)

# do not use X_test data after this point till the very end


print(X_train.iloc[[0]].values.tolist())
print(Y_train)