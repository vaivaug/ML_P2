import matplotlib.pyplot as plt
import seaborn as sns


def plot_colour_correlation(data):

    data = data[get_list_of_colour_bins(['red'])]
    print(data)
    # Correlation Matrix Heatmap
    f, ax = plt.subplots(figsize=(10, 6))

    corr = data.corr()
    hm = sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f',
                     linewidths=.05)
    f.subplots_adjust(top=0.93)
    t = f.suptitle('Colours Correlation Heatmap', fontsize=14)
    sns.heatmap(corr, annot=False, cmap=plt.cm.Reds)
    plt.show()


def get_list_of_colour_bins(colours=None):

    if colours is None:
        colours = ['red', 'green', 'blue']

    list_of_colours = []

    for colour in colours:
        for i in range(1, 17):
            list_of_colours.append(colour+'_'+str(i))

    return list_of_colours
