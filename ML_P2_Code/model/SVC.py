from sklearn import svm
from sklearn.model_selection import cross_val_predict
import numpy as np
from sklearn.multiclass import OneVsRestClassifier


def get_predicted_cross_validation_SVC_binary(data, threshold):
    """Train the SVC classifier and calculate probabilities of class being a seal
    Use cross-validation
    :param data: training data
    :param threshold: threshold value
    :return: predicted_output: dataframe for predicted classes
             data['output']: actual output for the given training data
             predicted_probabilities: list of probabilities, used to plot AUC
    """

    model = svm.SVC(probability=True, max_iter=1000)
    predicted_probs = cross_val_predict(model, data.drop(columns='output'), data['output'],
                                        cv=3, method='predict_proba')
    predicted_probs = predicted_probs[:, 1]

    # classify samples into two classes depending on the probabilities
    predicted_output = np.where(predicted_probs > threshold, 'seal', 'background')

    return predicted_output, data['output'], predicted_probs


def get_predicted_validation_SVC_binary(train, validation, threshold):
    """Train the SVC classifier and calculate probabilities of class being a seal
       Predict on validation data
    :param train: training data
    :param threshold: threshold value
    :param validation: validation dataset
    :return: predicted_output: dataframe for predicted classes
                 validation['output']: actual output for the given test data
                 predicted_probabilities: list of probabilities, used to plot AUC
    """
    model = svm.SVC(probability=True, max_iter=1000)
    model.fit(train.drop(columns='output'), train['output'])

    predicted_probs = model.predict_proba(validation.drop(columns='output'))
    predicted_probs = predicted_probs[:, 1]

    # classify samples into two classes depending on the probabilities
    predicted_output = np.where(predicted_probs > threshold, 'seal', 'background')

    return predicted_output, validation['output'], predicted_probs


def get_predicted_test_SVC_binary(train, X_test, threshold):
    """This class is used for final evaluation of the model on unseen data
    """

    model = svm.SVC(probability=True, max_iter=1000)
    model.fit(train.drop(columns='output'), train['output'])

    predicted_probs = model.predict_proba(X_test)
    predicted_probs = predicted_probs[:, 1]

    # classify samples into two classes depending on the probabilities
    predicted_output = np.where(predicted_probs > threshold, 'seal', 'background')

    return predicted_output


def get_predicted_cross_validation_SVC_multi(data):
    """Train the model to predict the output in a multiclass problem
    Use cross-validation
    :param data: training data
    :return: predicted and actual outputs
    """

    # SVC model
    model = OneVsRestClassifier(svm.SVC(max_iter=1000))

    predicted_output = cross_val_predict(model, data.drop(columns='output'), data['output'],
                                         cv=3, method='predict')

    return predicted_output, data['output']


def get_predicted_validation_SVC_multi(train, validation):
    """Train the model to predict the output in a multiclass problem
       Predict on validation set
    :param data: training data
    :return: predicted and actual outputs
    """
    model = OneVsRestClassifier(svm.SVC(max_iter=1000))
    model.fit(train.drop(columns='output'), train['output'])

    predicted_output = model.predict(validation.drop(columns='output'))

    return predicted_output, validation['output']


def get_predicted_test_SVC_multi(train, X_test):
    """This function is used at the end to evaluate model performance on X_test data
    """

    model = OneVsRestClassifier(svm.SVC(max_iter=2000))

    model.fit(train.drop(columns='output'), train['output'])
    predicted_output = model.predict(X_test)

    return predicted_output



