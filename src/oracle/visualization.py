"""
Module for visualization functions in the ORACLE framework.
"""
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
    """
    Plot a confusion matrix using the given true and predicted labels and display it with matplotlib.

    Parameters:
        y_true (array-like): Array of true labels.
        y_pred (array-like): Array of predicted labels corresponding to y_true.
        labels (list): List of label names to be used in the confusion matrix.
        normalize (str, optional): Normalization mode for the confusion matrix. Default is 'true'.
                                   Accepted values are typically 'true', 'pred', or 'all'.
        title (str, optional): Title of the plot; if provided, it will be set on the plot.
        img_file (str, optional): File path to save the generated plot image. If None, the plot is not saved.

    Returns:
        None

    Note:
        - The function filters out any entries where the true label is None.
        - It adjusts the figure size and text properties based on the number of classes.
        - The function closes all previous matplotlib figures at the start and closes the plot at the end.
    """

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
    """
    Plot a plain confusion matrix visualization based on true and predicted labels.
    This function computes and displays a confusion matrix for the provided true and
    predicted labels using a predefined style. It only considers entries in y_true that
    are not None. The resulting confusion matrix is displayed without tick marks or spines,
    and can optionally be saved to an image file.

    Parameters:
        y_true (array-like): Array of true labels. Only the elements that are not None are used.
        y_pred (array-like): Array of predicted labels corresponding to y_true.
        labels (array-like): The set of labels to index the confusion matrix.
        normalize (str, optional): Normalization method for the confusion matrix (e.g., 'true').
                                   Defaults to 'true'.
        title (str, optional): Title of the plot. (Currently not utilized in the function.)
        img_file (str, optional): If provided, the plot is saved to this file path.

    Returns:
        None
    """


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
    """
    Plot ROC curves for each class and compute the macro-average ROC curve.

    Parameters:
        probs_true (ndarray): A 2D array of shape (n_samples, n_classes) containing the ground truth binary labels for each class.
                              Rows with all-zero values (indicating missing true labels) are removed before plotting.
        probs_pred (ndarray): A 2D array of shape (n_samples, n_classes) containing the predicted probabilities for each class.
        labels (list of str): A list of class labels corresponding to the columns in probs_true and probs_pred.
        title (str, optional): Title of the ROC plot. Defaults to None.
        img_file (str, optional): File path to save the plot image. If None, the plot is not saved to a file.

    Returns:
        None

    Note:
        - The ROC curves are plotted with an equal aspect ratio, and a legend is included showing the AUC for each class
          along with the macro-average AUC.
    """

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
    """
    Plot the training and validation loss curves along with their rolling averages on a logarithmic scale.

    Parameters:
        train_loss_history (list or array-like): The history of training loss values.
        val_loss_history (list or array-like): The history of validation loss values.
        file_name (str): The file path where the generated plot will be saved.
    """

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
    """
    Plots class-wise performance over all phases for a given metric.

    This function iterates over each depth level present in the metrics_dictionary,
    extracts class-specific metric values (ignoring summary rows such as 'accuracy',
    'macro avg', and 'weighted avg'), and plots these values against the days from 
    the first detection. The x-axis is set to a logarithmic scale.

    Parameters:
        metric (str): The name of the performance metric to be displayed on the y-axis.
        metrics_dictionary (dict): A dictionary where each key represents a depth level and 
            its corresponding value is a pandas DataFrame. The DataFrame should have its rows 
            indexed by class names (with some entries like 'accuracy', 'macro avg', and 
            'weighted avg' to be skipped) and columns representing days from the first detection.
        model_dir (str or None, optional): The directory path where the plot PDFs will be saved.
            If provided, each plot is saved as 'class_wise_{metric}.pdf' in a subdirectory.
    """

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
    """
    Plot the average performance over all phases for the specified metric.

    This function iterates over a dictionary of metrics grouped by different depths. For each depth,
    it extracts the row corresponding to a specified metric (e.g., 'macro avg') from a pandas DataFrame,
    plots the metric values against the days from first detection on a logarithmic x-scale, and either
    displays the plot interactively or saves it to a file in the specified directory.

    Parameters:
        metric (str): The performance metric to be plotted. This is used to label the y-axis and is printed along with the plot data.
        metrics_dictionary (dict): A dictionary where each key corresponds to a depth level and each value is a pandas DataFrame.
            The DataFrame should contain metric rows (e.g., 'macro avg') with its index representing days from first detection and
            the row values representing the metric values.
        model_dir (str, optional): Directory in which to save the generated plot as a PDF file under the subdirectory 'plots'.
            If None, the plot is displayed interactively using plt.show(). Defaults to None.
            
    Note:
        - All existing matplotlib figures are closed at the beginning to prevent overlapping.
        - The plot style is set to 'default'.
        - The x-axis uses a logarithmic scale and its ticks are set based on the days from first detection.
        - A legend is added to the lower right of the plot.
    """

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

def plot_umap(embeddings, classes, bts_classes, id, d, model_dir=None):
    """
    Plot UMAP projection of embeddings.
    This function computes a 2D UMAP projection from high-dimensional embeddings and generates both a 
    static scatter plot using matplotlib and an interactive scatter plot using plotly. Points in the plots 
    are colored based on the provided classes. If a model directory is specified, the plots are saved 
    to disk; otherwise, the static plot is displayed.

    Parameters:
        embeddings (array-like): High-dimensional feature data (e.g., a numpy array) with shape (n_samples, n_features).
        classes (array-like): Class labels for each embedding, used for color-coding the points in the plot.
        id (array-like): Unique identifiers for each source, used for hover information in the interactive plot.
        bts_classes (array-like): Additional class information for tooltips in the interactive plot.
        d (int or str): Identifier (e.g., number of days or a delay parameter) used in the plot title and file names.
        model_dir (str, optional): Directory where the plots will be saved. If None, the static plot is shown instead.

    Returns:
        None

    Raises:
        Exceptions from UMAP, matplotlib, or plotly if issues occur during the transformation or plotting process.
    """


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
    df['raw_class'] = bts_classes
    df['id'] = id
    fig = px.scatter(df, x='umap1', y='umap2', color=f"class", hover_data=['class', 'raw_class', 'id'])#, cmap='viridis', marker=markers[i])
    fig.write_html(f"{model_dir}/plots/umap/umap_trigger+{d}.html")

        

        