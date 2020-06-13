from sklearn import metrics
import matplotlib.pyplot as auc_plt
from math import sqrt
import numpy as np
import pandas as pd
import pylab as pl


def plot_AUC_LR(data_output, predicted_output, prediction_probs, balancing_type, solver, threshold):

    fig, ax = auc_plt.subplots()

    # true positive rate
    recall = metrics.recall_score(data_output, predicted_output, labels=['seal', 'background'], pos_label='seal')
    precision = metrics.precision_score(data_output, predicted_output, labels=['seal', 'background'], pos_label='seal')
    accuracy = metrics.accuracy_score(data_output, predicted_output)
    print('accuracy: ', accuracy)

    # no skill prediction
    no_skill_probs = [0 for _ in range(len(data_output))]

    # calculate AUC score
    model_auc = metrics.roc_auc_score(data_output, prediction_probs)
    print('auc: ', model_auc)

    # calculate roc curves
    no_skills_false_pos_rate, no_skill_true_pos_rate, no_skill_thresholds = metrics.roc_curve(data_output,
                                                                                              no_skill_probs,
                                                                                              pos_label='seal')
    model_false_positive_rate, model_true_pos_rate, model_thresholds = metrics.roc_curve(data_output,
                                                                                         prediction_probs,
                                                                                         pos_label='seal')
    # string to be outputted in a text box
    experiment_params = '\n'.join((
        'Balancing type:  {}'.format(balancing_type),
        'Solver:  {}'.format(solver),
        'Threshold:  {}'.format(threshold),
        'Recall:  %.3f' % (recall,),
        'Precision:  %.3f' % (precision,),
        'AUC: %.3f' % (model_auc,),
        'Accuracy: %.3f' % (accuracy,),
    ))

    # plot the auc curves
    ax.plot(model_false_positive_rate, model_true_pos_rate, marker='.', label='Logistic')
    ax.plot(no_skills_false_pos_rate, no_skill_true_pos_rate, linestyle='--', label='No skills')
    props = dict(boxstyle='round', facecolor='none', alpha=0.7)

    # place a text box in the bottom right
    x, y = get_text_coordinates(balancing_type)
    ax.text(x, y, experiment_params, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # axis labels
    auc_plt.xlabel('False Positive Rate')
    auc_plt.ylabel('True Positive Rate')
    auc_plt.title('Area Under the ROC Curve')
    auc_plt.legend(bbox_to_anchor=(0.97, 0.38), loc='lower right')

    plt_fig = auc_plt.gcf()
    plt_fig.tight_layout()
    plt_fig.savefig('plots/auc_LR_{}_{}_{}.png'.format(balancing_type, solver, threshold))


def get_text_coordinates(balancing_type):

    if balancing_type == 'SMOTE':
        return 0.61, 0.36
    elif balancing_type == 'over-sample-seals':
        return 0.465, 0.36
    else:
        return 0.43, 0.36


def plot_AUC_SVC(data_output, predicted_output, prediction_probs, balancing_type, threshold):

    fig, ax = auc_plt.subplots()

    # true positive rate
    recall = metrics.recall_score(data_output, predicted_output, labels=['seal', 'background'], pos_label='seal')
    precision = metrics.precision_score(data_output, predicted_output, labels=['seal', 'background'], pos_label='seal')
    accuracy = metrics.accuracy_score(data_output, predicted_output)
    print('accuracy: ', accuracy)

    # no skill prediction
    no_skill_probs = [0 for _ in range(len(data_output))]

    # calculate AUC score
    model_auc = metrics.roc_auc_score(data_output, prediction_probs)
    print('auc: ', model_auc)

    # calculate roc curves
    no_skills_false_pos_rate, no_skill_true_pos_rate, no_skill_thresholds = metrics.roc_curve(data_output,
                                                                                              no_skill_probs,
                                                                                              pos_label='seal')
    model_false_positive_rate, model_true_pos_rate, model_thresholds = metrics.roc_curve(data_output,
                                                                                         prediction_probs,
                                                                                         pos_label='seal')
    # string to be outputted in a text box
    experiment_params = '\n'.join((
        'Balancing type:  {}'.format(balancing_type),
        'Threshold:  {}'.format(threshold),
        'Recall:  %.3f' % (recall,),
        'Precision:  %.3f' % (precision,),
        'AUC: %.3f' % (model_auc,),
        'Accuracy: %.3f' % (accuracy,),
    ))

    # plot the auc curves
    ax.plot(model_false_positive_rate, model_true_pos_rate, marker='.', label='SVC')
    ax.plot(no_skills_false_pos_rate, no_skill_true_pos_rate, linestyle='--', label='No skills')
    props = dict(boxstyle='round', facecolor='none', alpha=0.7)

    # place a text box in the bottom right
    x, y = get_text_coordinates(balancing_type)
    ax.text(x, y, experiment_params, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # axis labels
    auc_plt.xlabel('False Positive Rate')
    auc_plt.ylabel('True Positive Rate')
    auc_plt.title('Area Under the ROC Curve')
    auc_plt.legend(bbox_to_anchor=(0.97, 0.38), loc='lower right')

    plt_fig = auc_plt.gcf()
    plt_fig.tight_layout()
    plt_fig.savefig('plots/auc_SVC_{}_{}.png'.format(balancing_type, threshold))


