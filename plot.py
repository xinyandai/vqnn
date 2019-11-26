import numpy as np
import matplotlib.pyplot as plt


fontsize=44
ticksize=40
legendsize=30
plt.style.use('seaborn-white')
plot_accuracy = True

train_loss = 0
train_acc1 = 1
train_acc5 = 2
valid_loss = 3
valid_acc1 = 4
valid_acc5 = 5


def from_text(location):
    data = []
    for i in open("results/%s/log.txt" % location):
        if i.startswith(" Epoch: "):
            split = i.split("\t")
            data.append([float(s.split(" ")[-2])
                         for s in split[1:]])
    return np.array(data)


def _plot_setting():
    plt.xlabel('# Epochs', fontsize=fontsize)
    plt.ylabel('Accuracy' if plot_accuracy else "Loss", fontsize=fontsize)
    plt.yticks(fontsize=ticksize)
    plt.xticks(fontsize=ticksize)

    plt.legend(loc='lower right' if plot_accuracy else 'upper right', fontsize=legendsize)

    plt.show()


def plot_one(location, label, color, linestyle, marker=""):
    try:
        data = from_text(location)
        y = data[:, valid_acc1]
        print(y)
        x = np.arange(1, len(y)+1).astype(np.float64)
        plt.plot(x, y, color, label=label, linestyle=linestyle, marker=marker)
    except Exception as e:
        print(e)


def plot_cifar100():
    plot_one('resnet18_cifar100_identical', 'Full Precision', 'black', '--', ' ')
    plot_one('resnet18_cifar100_bnn', 'BNN', 'blue', '-.', ' ')
    plot_one('resnet18_cifar100_vq_dim4_k16', 'VQNN(d=4, k=16)', 'red', '-', ' ')
    _plot_setting()


def plot_cifar10():
    plot_one('resnet18_cifar10_identical', 'Full Precision', 'black', '--', ' ')
    plot_one('resnet18_cifar10_bnn', 'BNN', 'blue', '-.', ' ')
    plot_one('resnet18_cifar10_vq_dim4_k16', 'VQNN', 'red', '-', ' ')
    _plot_setting()


def plot_varies_codebook():
    plot_one('resnet18_cifar10_identical', 'Full Precision', 'black', '--', ' ')
    plot_one('resnet18_cifar10_vq_dim4_k16', 'VQNN(d=4, k=16)', 'red', '--', ' ')
    plot_one('resnet18_cifar10_vq_dim8_k16', 'VQNN(d=8, k=16)', 'blue', '-.', ' ')
    plot_one('resnet18_cifar10_vq_dim16_k256', 'VQNN(d=16, k=256)', 'gray', '-', ' ')
    plot_one('resnet18_cifar10_vq_dim32_k256', 'VQNN(d=32, k=256)', 'yellow', '-', ' ')
    _plot_setting()


def plot_straight():
    plot_one('resnet18_cifar10_identical', 'Full Precision', 'black', '--', ' ')
    plot_one('resnet18_cifar10_vq_dim4_k16_no_org', 'VQNN-straight', 'blue', '-', ' ')
    plot_one('resnet18_cifar10_vq_dim4_k16', 'VQNN', 'red', '--', ' ')
    _plot_setting()


plot_cifar10()
