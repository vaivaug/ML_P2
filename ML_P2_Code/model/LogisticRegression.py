from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier


def get_predicted_cross_validation_LR_binary(data, threshold, solver='lbfgs'):
    """Create Logistic Regression model on the train data. Calculate probability of sample being a 'seal'
    Use cross-validation
    @param data: pandas dataframe containing all the training data
    @param threshold: threshold value
    @param solver: type of solver for Logistic Regression
    @return: predicted_output: list of predicted values for each row in the data set
             prediction_probs: list of probabilities between 0 and 1 for each row in the data set
    """

    # logistic regression
    model = LogisticRegression(solver=solver, max_iter=1000)

    predicted_probs = cross_val_predict(model, data.drop(columns='output'), data['output'],
                                        cv=3, method='predict_proba')
    predicted_probs = predicted_probs[:, 1]

    # classify samples into two classes depending on the probabilities
    predicted_output = np.where(predicted_probs > threshold, 'seal', 'background')

    print('predicted: ', predicted_output)
    print('actual: ', data['output'])

    return predicted_output, data['output'], predicted_probs


def get_predicted_validation_LR_binary(train, validation, threshold=0.5, solver='lbfgs'):
    """Create Logistic Regression model on the train data. Calculate probability of sample being a 'seal'
    Evaluate using validation set
    @param train: pandas dataframe containing all the training data
    @param threshold: threshold value
    @param solver: type of solver for Logistic Regression
    @return: predicted_output: list of predicted values for each row in the data set
             prediction_probs: list of probabilities between 0 and 1 for each row in the data set
    """

    # logistic regression
    model = LogisticRegression(solver=solver, max_iter=1000)

    model.fit(train.drop(columns='output'), train['output'])

    predicted_probs = model.predict_proba(validation.drop(columns='output'))

    predicted_probs = predicted_probs[:, 1]

    # classify samples into two classes depending on the probabilities
    predicted_output = np.where(predicted_probs > threshold, 'seal', 'background')

    print('predicted: ', predicted_output)
    print('actual: ', validation['output'])

    return predicted_output, validation['output'], predicted_probs


def get_predicted_cross_validation_LR_multi(data, solver='lbfgs'):
    """Create Logistic Regression model on the train data.
    Use cross-validation
    @param data: pandas dataframe containing all the training data
    @param solver: type of solver for Logistic Regression
    @return: predicted_output: list of predicted values for each row in the data set
             prediction_probs: list of probabilities between 0 and 1 for each row in the data set
    """

    # logistic regression
    model = OneVsRestClassifier(LogisticRegression(solver=solver, max_iter=1000))

    predicted_output = cross_val_predict(model, data.drop(columns='output'), data['output'],
                                         cv=3, method='predict')

    return predicted_output, data['output']


def get_predicted_validation_LR_multi(train, validation, solver='lbfgs'):
    """Create Logistic Regression model on the train data.
    Evaluate using validation set
    @param train: pandas dataframe containing all the training data
    @param solver: type of solver for Logistic Regression
    @return: predicted_output: list of predicted values for each row in the data set
             prediction_probs: list of probabilities between 0 and 1 for each row in the data set
    """

    # logistic regression
    model = OneVsRestClassifier(LogisticRegression(solver=solver, max_iter=1000))

    model.fit(train.drop(columns='output'), train['output'])

    predicted_output = model.predict(validation.drop(columns='output'))

    print('predicted: ', predicted_output)
    print('actual: ', validation['output'])

    return predicted_output, validation['output']
