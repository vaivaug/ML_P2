
def get_train_validation_test(X_train, Y_train):

    X_train['output'] = Y_train['output']
    train = X_train.sample(frac=1, random_state=1).reset_index(drop=True)

    # keep random_state variable to always form the same set
    # Keep 30% of the data to form test and validation sets
    test_validation = train.sample(frac=0.3, random_state=0).reset_index(drop=True)
    print('length of test and validation data together: ', len(test_validation))

    # test_validation data is split into half for test and validation sets
    test = test_validation.sample(frac=0.5, random_state=1).reset_index(drop=True)
    validation = test_validation.drop(test.index).reset_index(drop=True)

    # The other 70% of data is used for training
    train = train.drop(test_validation.index).reset_index(drop=True)
    print('length of training data: ', len(train))

    return train, validation, test
