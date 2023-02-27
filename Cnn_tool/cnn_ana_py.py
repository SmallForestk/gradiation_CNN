import numpy as np
import matplotlib.pyplot as plt

def sb_histgram(y_pred, y):
    data = np.stack([y, y_pred], 0)
    print(data.shape)
    positive = data[:, data[0]==1]
    negative = data[:, data[0]==0]
    print(positive.shape, negative.shape)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    ax.hist(positive[1], bins=100, alpha=0.5, label="pi")
    ax.hist(negative[1], bins=100, alpha=0.5, label="kaon")
    ax.set_xlim(0, 1.)
    ax.legend()
    fig.show()

def plot_loss_accuracy(tloss, vloss, taccuracy, vaccuracy):
    _, axes = plt.subplots(2, 1, figsize=(12, 12))
    axes[0].plot(tloss, marker="o", label="train loss")
    axes[0].plot(vloss, marker="o", label="validation loss")
    axes[0].grid()
    axes[0].legend()
    axes[0].set_xlabel("epoch")
    axes[0].set_title("Loss", fontsize=18)

    axes[1].plot(taccuracy, marker="o", label="train accuracy")
    axes[1].plot(vaccuracy, marker="o", label="validation accuracy")
    axes[1].legend()
    axes[1].grid()
    axes[1].set_ylim(0.7, 1.)
    axes[1].set_xlabel("epoch")
    axes[1].set_title("Accuray", fontsize=18)
    plt.show()

def plot_roc(y_pred, y):
    threshold = np.linspace(0, 1, 1000)
    tpr = []
    fpr = []
    for i in range(1000):
        anser = (y_pred >= threshold[i]).astype(int)
        data = np.stack([y, anser], 0)
        signal = data[:, data[0]==1]
        fn = (signal[:, signal[1]==0].shape)[1]
        tp = (signal[:, signal[1]==1].shape)[1]
        backgraund = data[:, data[0]==0]
        fp = (backgraund[:, backgraund[1]==1].shape)[1]
        tn = (backgraund[:, backgraund[1]==0].shape)[1]
        tpr.append(tp/(tp+fn))
        fpr.append(fp/(fp+tn))
    tpr = np.array(tpr)
    fpr = np.array(fpr)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(tpr, 1-fpr, label="ROC curve")
    ax.set_title("ROC curve", fontsize=18)
    ax.grid()
    ax.set_xlabel("True Positive Rate(TPR)", fontsize=18)
    ax.set_xlim(0., 1.1)
    ax.set_ylabel("False Positive Rate(FPR)", fontsize=18)
    ax.set_ylim(0., 1.1)
    plt.show()

def test_accuracy(y_out, y_true, threshold=0.5):
    nd_threshold = np.full(y_out.shape, threshold)
    result = y_out >= nd_threshold
    result = result.astype(np.int32)
    nd_correct = result == y_true
    correct = np.sum(nd_correct.astype(np.int32))
    correct /= y_out.size
    return correct