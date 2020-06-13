import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def target_value_counts(data):
    print(data['output'].value_counts())


def describe_data_by_class_binary(data):
    print(data.describe())
    # create two datasets containing only 'seal' and only 'background' samples
    seal = data[data['output'] == 'seal']
    background = data[data['output'] == 'background']

    print('seal: ')
    print(seal.describe())
    print('background: ')
    print(background.describe())


def plot_gradient_counts(data):

    class_names_list = data.output.unique()

    for name in class_names_list:
        # create a dataset containing samples only from class 'name'
        class_samples = data[data['output'] == name]
        plot_gradient_counts_one_class(class_samples, "Value counts of the first 900 columns"
                                                      " for '{}' samples".format(name))


def plot_gradient_counts_one_class(data, title):

    ranges_list = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]

    all_values_list = []
    for i in range(1, 901):
        all_values_list.extend(data[i].tolist())

    gradient_counts_list = []
    for i in ranges_list:
        print('i: ', i)
        count = 0
        # count all elements in the given ranges
        for element in all_values_list:
            if i < element <= i+0.2:
                count += 1

        print('count: ', count)
        gradient_counts_list.append(count)

    ind = np.arange(10)  # the x locations for the groups
    width = 0.9  # the width of the bars: can also be len(x) sequence
    plt.bar(ind, gradient_counts_list, width)

    plt.ylabel('value counts')
    plt.xlabel('value ranges')
    plt.title(title)
    plt.xticks(ind, ('(-1, -0.8]', '(-0.8, -0.6]', '(-0.6, -0.4]', '(-0.4, -0.2]', '(-0.2, 0]',
                     '(0, 0.2]', '(0.2, 0.4]', '(0.4, 0.6]', '(0.6, 0.8]', '(0.8, 1]'))
    plt.show()


def plot_normal_distribution_counts(data):

    class_names_list = data.output.unique()

    for name in class_names_list:
        # create a dataset containing samples only from class 'name'
        class_samples = data[data['output'] == name]
        plot_normal_distribution_counts_one_class(class_samples, "Value counts of the 16 normal distribution columns"
                                                      " for '{}' samples".format(name))


def plot_normal_distribution_counts_one_class(data, title):

    all_values_list = []
    for i in range(1, 17):
        all_values_list.extend(data['nd_variable_{}'.format(i)].tolist())

    nd_counts_list = []
    for i in range(-4, 4):
        print('i: ', i)
        count = 0
        # count all elements in the given ranges
        for element in all_values_list:
            if i < element <= i + 1:
                count += 1

        print('count: ', count)
        nd_counts_list.append(count)

    ind = np.arange(8)  # the x locations for the groups
    width = 0.9  # the width of the bars: can also be len(x) sequence
    bar = plt.bar(ind, nd_counts_list, width)

    plt.ylabel('value counts')
    plt.xlabel('value ranges')
    plt.title(title)
    plt.xticks(ind, ('(-4, -3]', '(-3, -2]', '(-2, -1]', '(-1, 0]', '(0, 1]', '(1, 2]', '(2, 3]', '(3, 4]'))
    plt.show()


def plot_colour_counts(data):

    class_names_list = data.output.unique()

    for name in class_names_list:
        # create a dataset containing samples only from class 'name'
        class_samples = data[data['output'] == name]
        plot_colour_counts_one_class(class_samples, name, 'red')
        plot_colour_counts_one_class(class_samples, name, 'green')
        plot_colour_counts_one_class(class_samples, name, 'blue')


def plot_colour_counts_one_class(data, class_name, colour_name):

    sums_per_bins = []
    for i in range(1, 17):
        sums_per_bins.append(data['{}_{}'.format(colour_name, i)].sum()/len(data))

    ind = np.arange(16)  # the x locations for the groups
    width = 0.9  # the width of the bars: can also be len(x) sequence
    bar = plt.bar(ind, sums_per_bins, width)

    plt.ylabel('value counts')
    plt.xlabel('value ranges')
    plt.title('Sum of number of pixels in each bin for {} samples, colour "{}"'.format(class_name, colour_name))
    plt.xticks(ind, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'))
    plt.show()



