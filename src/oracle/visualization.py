import umap

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.animation as animation
import matplotlib.patches as mpatches

from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score

cm = plt.get_cmap('gist_rainbow')

def plot_confusion_matrix(y_true, y_pred, labels, normalize='true', title=None, img_file=None):

    plt.close('all')
    plt.style.use(['default'])


    # Only keep source where a true label exists
    idx = np.where(y_true!=None)[0]
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    
    n_class = len(labels)
    font = {'size'   : 25}
    plt.rc('font', **font)
    
    cm = np.round(confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize),2)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    disp.im_.colorbar.remove()
    
    fig = disp.figure_
    if n_class > 7:
        plt.xticks(rotation=90)
        plt.yticks(rotation=45)
    
    fig.set_figwidth(18)
    fig.set_figheight(18)
    
    for label in disp.text_.ravel():
        if n_class > 7:
            label.set_fontsize(12)
        elif n_class <= 7 and n_class > 3:
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

    plt.close()

def plot_plain_cf(y_true, y_pred, labels, normalize='true', title=None, img_file=None):

    plt.close('all')
    plt.style.use(['default'])

    # Only keep source where a true label exists
    idx = np.where(y_true!=None)[0]
    y_true = y_true[idx]
    y_pred = y_pred[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    
    cm = np.round(confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize),2)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, include_values=False, colorbar=False, cmap='Blues')

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel('')
    ax.set_ylabel('')

    # 🔹 Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)    
    
    plt.tight_layout()

    if img_file:
        plt.savefig(img_file)

    plt.close()

def plot_roc_curves(probs_true, probs_pred, labels, title=None, img_file=None):

    plt.close('all')
    plt.style.use(['default'])

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

    plt.close()


def plot_train_val_history(train_loss_history, val_loss_history, file_name):

    window_size = 10

    rolling_train = []
    rolling_val = []
    s = []

    for i in range(len(train_loss_history) - window_size):

        rolling_train.append(np.mean(train_loss_history[i:i+window_size]))
        rolling_val.append(np.mean(val_loss_history[i:i+window_size]))
        s.append(i) #s.append(i + window_size)

    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(5, 7), layout="constrained")

    axs[0].plot(list(range(len(train_loss_history))), np.log(train_loss_history), label='Train Loss', color='C0', alpha=0.5)
    axs[0].plot(s, np.log(rolling_train), label='Rolling Avg Train Loss', color='C1')

    # axs[0].set_ylabel("Mean log loss", fontsize='x-large')
    # axs[0].legend()
    axs[0].set_xticks([])

    axs[1].plot(list(range(len(val_loss_history))), np.log(val_loss_history), label='Validation Loss', color='C0', alpha=0.5)
    axs[1].plot(s, np.log(rolling_val), label='Rolling Avg Validation Loss', color='C1')

    # axs[1].set_xlabel("Epoch", fontsize='x-large')
    # axs[1].set_ylabel("Mean log loss", fontsize='x-large')
    # axs[1].legend()

    # axs[0].set_ylim(-3.4, -1)
    # axs[1].set_ylim(-3.4, -1)

    plt.savefig(file_name)
    plt.close()


def plot_class_wise_performance_over_all_phases(metric, metrics_dictionary, model_dir=None):

    plt.close('all')
    plt.style.use(['default'])

    for depth in metrics_dictionary:

        df = metrics_dictionary[depth]

        for c, row in df.iterrows():

            if c not in ['accuracy','macro avg','weighted avg']:

                days, value = row.index, row.values

                plt.plot(days, value, label=c, marker = 'o')

        plt.xlabel("Days from first detection", fontsize='xx-large')
        plt.ylabel(f"{metric}", fontsize='xx-large')

        plt.grid()
        plt.tight_layout()
        plt.legend(loc='lower right')
        plt.xscale('log')
        plt.xticks(days, days)

        if model_dir == None:
            plt.show()
        else:
            plt.savefig(f"{model_dir}/plots/depth{depth}/class_wise_{metric}.pdf")

        plt.close()



def plot_average_performance_over_all_phases(metric, metrics_dictionary, model_dir=None):

    plt.close('all')
    plt.style.use(['default'])

    for depth in metrics_dictionary:

        df = metrics_dictionary[depth]

        for c, row in df.iterrows():

            if c in ['macro avg']: # Put things like ['macro avg','weighted avg']

                days, value = row.index, row.values

                plt.plot(days, value, label=f"{c}(Depth={depth})", marker = 'o')
                print(metric, days, value)

    plt.xlabel("Days from first detection", fontsize='xx-large')
    plt.ylabel(f"{metric}", fontsize='xx-large')

    plt.grid()
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.xscale('log')
    plt.xticks(days, days)

    if model_dir == None:
        plt.show()
    else:
        plt.savefig(f"{model_dir}/plots/average_{metric}.pdf")

    plt.close()

def plot_umap(embeddings, classes, bts_classes, d, model_dir=None):

    plt.close('all')
    plt.style.use(['default'])

    reducer = umap.UMAP(random_state=42)
    umap_embedding = reducer.fit_transform(embeddings)

    x = umap_embedding[:, 0]
    y = umap_embedding[:, 1]

    for c in np.unique(classes):

        idx_class = np.where(np.asarray(classes)==c)[0]
        plt.scatter(x[idx_class], y[idx_class], label=c, marker='.')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2, fontsize = 12)
    
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    plt.tight_layout()

    plt.title(f"Trigger+{d} days")

    if model_dir != None:
            plt.savefig(f"{model_dir}/plots/umap/umap_trigger+{d}.pdf") 
    else:
        plt.show()

    plt.close()

    df = pd.DataFrame(umap_embedding, columns=['umap1','umap2'])
    df['class'] = classes
    df['bts_class'] = bts_classes
    fig = px.scatter(df, x='umap1', y='umap2', color=f"class", hover_data=['class', 'bts_class'])#, cmap='viridis', marker=markers[i])
    fig.write_html(f"{model_dir}/plots/umap/umap_trigger+{d}.html")

        

def plot_embeddings_umaps(df, days, class_list, model_dir=None):

    plt.close('all')
    plt.style.use(['default'])

    embeddings = df.to_numpy()

    reducer = umap.UMAP(random_state=42)
    umap_embedding = reducer.fit_transform(embeddings)

    for d in np.unique(days):

        idx = np.where(days==d)[0]
        x = umap_embedding[idx, 0]
        y = umap_embedding[idx, 1]
        classes = class_list[idx]

        for c in np.unique(classes):

            idx_class = np.where(np.asarray(classes)==c)[0]
            plt.scatter(x[idx_class], y[idx_class], label=c, marker='.')

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2, fontsize = 12)
        
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')

        plt.tight_layout()

        plt.title(f"Trigger+{d} days")

        if model_dir != None:
            if "Anomaly" in class_list:
                plt.savefig(f"{model_dir}/plots/umap_AD/umap_trigger+{d}.pdf")
            else:
                plt.savefig(f"{model_dir}/plots/umap/umap_trigger+{d}.pdf") 
        else:
            plt.show()

        plt.close()

        

        