# %%

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


def load(name_list):
    path = "C:/Users/Chiak/Desktop/research/FYP/vqnn/results"
    table = {}
    for name in name_list:
        df = pd.read_csv("{}/{}/results.csv".format(path, name))
        table[name] = df
    return table


def plot(name_dict, titles):
    for key, value in name_dict.items():
        plt.plot(100-value['val_error1'], label=key)
    if titles is not None:
        plt.title(titles[0])
    else:
        plt.title("val_prec@1")
    plt.legend()
    plt.show()
    for key, value in name_dict.items():
        plt.plot(100-value['val_error5'], label=key)
    if titles is not None:
        plt.title(titles[1])
    else:
        plt.title("val_prec@5")
    plt.legend()
    plt.show()


def load_plot(name_list, titles=None):
    table = load(name_list)
    plot(table, titles)


# %%
# cifar100, resnet18, Adam, lr=5e-3
dataset = 'cifar100'
network = 'resnet18'
optimizer = 'Adam'
methods = ['_identical_{}'.format(optimizer), 'q_BC_{}'.format(optimizer), 'q_BNN_{}'.format(optimizer), '_VQ_{}'.format(
    optimizer), '_VQ_VQ{}_rate4'.format(optimizer), '_VQ_VQ{}_rate10'.format(optimizer)]
names = ["{}/{}{}".format(dataset, network, x) for x in methods]
print(names)
load_plot(names)

# %%
# cifar10, resnet18, Adam, lr=5e-3
dataset = 'cifar10'
network = 'resnet18'
optimizer = 'Adam'
methods = ['_identical_{}'.format(optimizer), 'q_BC_{}'.format(optimizer), 'q_BNN_{}'.format(optimizer), '_VQ_{}'.format(
    optimizer), '_VQ_VQ{}_rate4'.format(optimizer), '_VQ_VQ{}_rate10'.format(optimizer)]
names = ["{}/{}{}".format(dataset, network, x) for x in methods]
print(names)
load_plot(names)


# %%
# mnist, logistic, SGD, lr=1e-3
dataset = 'mnist'
network = 'logistic'
optimizer = 'Adam'
methods = ['identical_{}'.format(optimizer), 'BC_{}'.format(optimizer), 'VQ_{}'.format(
    optimizer), 'VQ_VQ{}_rate4'.format(optimizer)]

names = ["{}/{}_{}".format(dataset, network, x) for x in methods]
print(names)
load_plot(names)


# %%
# mnist, logistic, SGD, lr=1e-3
dataset = 'rcv1'
network = 'logistic'
optimizer = 'SGD'
# methods = ['identical_{}'.format(optimizer), 'BC_{}'.format(optimizer), 'VQ_{}'.format(
# optimizer), 'VQ_VQ{}_rate4'.format(optimizer)]
methods = ['identical_{}'.format(optimizer), 'BC_{}'.format(
    optimizer), 'VQ_{}'.format(optimizer), 'VQ_VQ{}_rate4'.format(optimizer)]
names = ["{}/{}_{}".format(dataset, network, x) for x in methods]
print(names)
load_plot(names, ['f1', 'acc'])

# %%
