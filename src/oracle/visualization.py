import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score

cm = plt.get_cmap('gist_rainbow')

def plot_confusion_matrix(y_true, y_pred, labels, title=None, img_file=None):

    # Only keep source where a true label exists
    idx = np.where(y_true!=None)[0]
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    
    n_class = len(labels)
    font = {'size'   : 25}
    plt.rc('font', **font)
    
    cm = np.round(confusion_matrix(y_true, y_pred, labels=labels, normalize='true'),2)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    disp.im_.colorbar.remove()
    
    fig = disp.figure_
    if n_class > 10:
        plt.xticks(rotation=90)
        plt.yticks(rotation=45)
    
    fig.set_figwidth(18)
    fig.set_figheight(18)
    
    for label in disp.text_.ravel():
        if n_class > 10:
            label.set_fontsize(12)
        elif n_class <= 10 and n_class > 3:
            disp.ax_.tick_params(axis='both', labelsize=40)
            label.set_fontsize('xx-large')
        else:
            disp.ax_.tick_params(axis='both', labelsize=40)
            label.set_fontsize('xx-large')
    
    if title:
        disp.ax_.set_xlabel("Predicted Label", fontsize=60)
        disp.ax_.set_ylabel("True Label", fontsize=60)
        disp.ax_.set_title(title, fontsize=60)
    
    plt.tight_layout()

    if img_file:
        plt.savefig(img_file)

def plot_roc_curves(probs_true, probs_pred, labels, title=None, img_file=None):

    # Only keep source where a true label exists
    idx = np.where(np.sum(probs_true, axis=1)!=0)[0]
    probs_true = probs_true[idx,:]
    probs_pred = probs_pred[idx,:]

    chance = np.arange(-0.001,1.01,0.001)
    if probs_pred.shape[1] <10:
        plt.figure(figsize=(12,12))
    else:
        plt.figure(figsize=(12,16))
    plt.plot(chance, chance, '--', color='black')

    color_arr=[cm(1.*i/probs_true.shape[1]) for i in range(probs_true.shape[1])]

    n_classes = probs_true.shape[1]   
    fpr_all = np.zeros((n_classes, len(chance)))
    tpr_all = np.zeros((n_classes, len(chance)))
    macro_auc = 0

    for i, label in enumerate(labels):

        score = roc_auc_score(probs_true[:, i], probs_pred[:, i])
        fpr, tpr, _ = roc_curve(probs_true[:, i], probs_pred[:, i])

        macro_auc += score
        fpr_all[i, :] = chance
        tpr_all[i, :] = np.interp(chance, fpr, tpr)

        plt.plot(fpr, tpr, label=f"{label} (AUC = {score:.2f})", color=color_arr[i])

    macro_auc = macro_auc/probs_true.shape[1]
    fpr_macro = np.mean(fpr_all, axis=0)
    tpr_macro = np.mean(tpr_all, axis=0)

    plt.plot(fpr_macro, tpr_macro, linestyle=':', linewidth=4 , label=f"Macro avg (AUC = {macro_auc:.2f})", color='red')

    plt.xlabel('False Positive Rate', fontsize=40)
    plt.ylabel('True Positive Rate', fontsize=40)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2, fontsize = 20)
    plt.title(title, fontsize=40)
    plt.tight_layout()
    plt.gca().set_aspect('equal')

    if img_file:
        plt.savefig(img_file)