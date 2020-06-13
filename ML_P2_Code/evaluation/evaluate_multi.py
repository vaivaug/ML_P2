from sklearn import metrics
import matplotlib.pyplot as auc_plt
from math import sqrt
import numpy as np
import pandas as pd
import pylab as pl


def evaluate_multi(data_output, predicted_output):

    # true positive rate
    recall = metrics.recall_score(data_output, predicted_output, average="macro")
    print('recall: ', recall)
    precision = metrics.precision_score(data_output, predicted_output, average="macro")
    print('precision: ', precision)
    accuracy = metrics.accuracy_score(data_output, predicted_output)
    print('accuracy: ', accuracy)

