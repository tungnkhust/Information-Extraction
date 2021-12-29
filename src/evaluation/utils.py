from typing import List
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import os

ROOT_PATH = sys.path[1]


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          save_dir=None):


    """Function to plot confusion matrics.

    :param cm: confusion_matrix: function in sklearn.
    :param target_names: list of classes.
    :param cmap: str or matplotlib Colormap: Colormap recognized by matplotlib.
    :param normalize: normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
    :param save_dir: str: directory address to save.
    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label. Metrics: accuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass))
    if save_dir is None:
        save_dir = os.path.join(ROOT_PATH, "report")

    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    plt.savefig((save_dir + '/{}.png'.format(title.replace(' ', '-'))))


class Column:
    def __init__(self, key, value=None):
        self.key = key
        if value is None:
            value = []
        self.value = value
        self.max_seq = self.max_line()

    def __getitem__(self, i):
        return self.value[i]

    def max_line(self):
        if len(self.value) == 0:
            return len(self.key)
        return max(max([len(str(v)) for v in self.value]), len(self.key))

    def print_item(self, i):
        return str(self.value[i]) + ' '*(self.max_seq-len(str(self.value[i])))

    def print_key(self):
        return str(self.key) + ' ' * (self.max_seq - len(str(self.key)))