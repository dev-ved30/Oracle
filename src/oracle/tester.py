"""
Module for testing hierarchical models in the ORACLE framework."""

import torch

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from tqdm import tqdm
from pathlib import Path
from functools import reduce    

from oracle.visualization import *
class Tester:
    """
    Top-level class providing testing functionalities for hierarchical classification models."""
    
    def setup_testing(self, model_dir, device):
        """
        Sets up the testing environment by configuring the model directory and device used for testing.

        Parameters:
            model_dir (str): The directory path where the model files are stored.
            device (torch.device or str): The device on which the model will run (e.g., CPU or GPU).

        Returns:
            None
        """

        self.model_dir = model_dir
        self.device = device

    def create_loss_history_plot(self):
        """
        Create and save a plot of the training and validation loss history.

        Note:
            - The numpy files must exist in the specified directory.
            - The 'plot_train_val_history' function must be properly defined and accessible.
        """

        # Load the train and validation loss history
        train_loss_history = np.load(f"{self.model_dir}/train_loss_history.npy")
        val_loss_history = np.load(f"{self.model_dir}/val_loss_history.npy")
        
        # Save the plot
        file_name = f"{self.model_dir}/loss.pdf"
        plot_train_val_history(train_loss_history, val_loss_history, file_name)

    def create_metric_phase_plots(self):
        """
        Generates phase plots for key evaluation metrics across all experimental phases.
        This method iterates over a predefined list of metrics ('f1-score', 'precision', 'recall'),
        retrieving the corresponding metric values across different phases by invoking the 
        get_metric_over_all_phases method. For each metric, it then generates two types of plots:
            1. Class-wise performance over all phases using plot_class_wise_performance_over_all_phases.
            2. Level-averaged performance over all phases using plot_average_performance_over_all_phases.
        The plots are saved to the directory specified by the model_dir attribute.
        """
        
        for metric in ['f1-score','precision','recall']:
            metrics_dictionary = self.get_metric_over_all_phases(metric)
            plot_class_wise_performance_over_all_phases(metric, metrics_dictionary, self.model_dir)
            plot_average_performance_over_all_phases(metric, metrics_dictionary, self.model_dir)

    def create_classification_report(self, y_true, y_pred, file_name=None):
        """
        Generates a classification report comparing true and predicted labels, and optionally writes the report to a CSV file.
        This method first filters the input arrays to include only entries with a non-None true label. It then computes
        the classification report using scikit-learn's classification_report function. If a file name is provided, it also
        exports the detailed report as a CSV file.

        Parameters:
            y_true (array-like): Array of true labels. Only entries where the label is not None will be considered.
            y_pred (array-like): Array of predicted labels, corresponding to y_true.
            file_name (str, optional): The file path where the CSV report will be saved. If None, the CSV file is not generated.

        Returns:
            str: A text summary of the classification report.
        """
        
        # Only keep source where a true label exists
        idx = np.where(y_true!=None)[0]
        y_true = y_true[idx]
        y_pred = y_pred[idx]

        report = classification_report(y_true, y_pred)

        if file_name:
            report_dict = classification_report(y_true, y_pred, output_dict=True)
            pd.DataFrame(report_dict).transpose().to_csv(file_name, index_label='Class')
        return report

    def get_metric_over_all_phases(self, metric):
        """
        Calculates and aggregates the specified metric (f1-score, precision, or recall) across all non-root taxonomy depths.

        Parameters:
            metric (str): The name of the metric to process. Must be one of ['f1-score', 'precision', 'recall'].

        Returns:
            dict: A dictionary where each key is a taxonomy depth (int) and each value is a pandas DataFrame containing the day-wise aggregated metric data.

        Raises:
            AssertionError: If the provided metric is not one of the accepted values.
        """
        
        # Make sure metric is valid
        assert metric in ['f1-score','precision','recall']

        nodes_by_depth = self.taxonomy.get_nodes_by_depth()

        metrics_dictionary = {}

        for depth in nodes_by_depth:

            if depth != 0:

                reports_dir = Path(f"{self.model_dir}/reports/depth{depth}/")
                reports = list(reports_dir.glob("*.csv"))
                reports.sort()
                
                day_wise_metrics = []

                for report in reports:

                    day = int(str(report).split("+")[1].split('.')[0])
                    df = pd.read_csv(report, index_col='Class')[[metric]]
                    df = df.rename(columns={metric:day})
                    day_wise_metrics.append(df)

                combined_df = pd.concat(day_wise_metrics, axis=1, join='inner')
                combined_df = combined_df.reindex(sorted(combined_df.columns), axis=1)
                metrics_dictionary[depth] = combined_df

        return metrics_dictionary
    
    def make_embeddings_for_AD(self, test_loader, d):
        """
        Generate latent space embeddings for anomaly detection (AD) analysis and save the results.
        This method processes the test dataset by running model inference, extracting latent embeddings,
        and gathering the corresponding class labels and identifiers. It then creates a UMAP plot of the
        embeddings and saves both the plot and a CSV file containing the combined embedding data.

        Parameters:
            test_loader (iterable): A DataLoader or iterable over the test dataset where each batch is a
                                    dictionary with keys 'label', 'raw_label', 'id', and any tensor data
                                    needed for embedding generation.
            d (int or float): A parameter indicating the number of days (or a similar metric) used in naming
                              the output files and plots.
        """

        self.eval()
        nodes_by_depth = self.taxonomy.get_nodes_by_depth()

        true_classes = []
        raw_classes = []
        ids = []
        combined_embeddings = []

        print(f'==========\nStarting Analysis for Trigger + {d} days...')

        # Run inference on the test set and combine the output dataframes
        for batch in tqdm(test_loader, desc='Testing'):

            # Move everything to the device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}


            embeddings = pd.DataFrame(self.get_latent_space_embeddings(batch).detach().cpu())

            true_classes += batch['label'].tolist()
            raw_classes += batch['raw_label'].tolist()
            ids += batch['id'].tolist()

            combined_embeddings.append(embeddings)
        
        true_classes = np.array(true_classes)
        combined_embeddings = pd.concat(combined_embeddings, ignore_index=True)

        Path(f"{self.model_dir}/plots/umap").mkdir(parents=True, exist_ok=True)
        plot_umap(combined_embeddings.to_numpy(), true_classes, raw_classes, d, model_dir=self.model_dir)

        combined_embeddings['class'] = true_classes
        combined_embeddings['raw_class'] = raw_classes
        combined_embeddings['id'] = ids

        Path(f"{self.model_dir}/embeddings").mkdir(parents=True, exist_ok=True)
        combined_embeddings.to_csv(f"{self.model_dir}/embeddings/embeddings+{d}.csv", index=False)

    def merge_performance_tables(self, days):
        """
        Merge performance tables for specified days and print LaTeX formatted results.
              
        Parameters:
            days (iterable): A collection (e.g., list) of identifiers representing different report days.

        Returns:
            None

        Side Effects:
            Outputs a LaTeX formatted table to standard output.
        """

        levels = ['1','2']

        for level in levels:

            data_frames = []

            for d in days:

                df = pd.read_csv(f"{self.model_dir}/reports/depth{level}/report_trigger+{d}.csv", index_col=0)
                df.drop(columns=["support"], inplace=True)
                df.index.name = 'Class'
                df.rename(columns={'precision': '$p_{' + f"{d}" + '}$'}, inplace=True)
                df.rename(columns={'recall': '$r_{' + f"{d}" + '}$'}, inplace=True)
                df.rename(columns={'f1-score': '$f1_{' + f"{d}" + '}$'}, inplace=True)
                data_frames.append(df)

            df_merged = reduce(lambda  left,right: pd.merge(left,right, how='left',on='Class', sort=False), data_frames)
            df_merged = df_merged.loc[:, df_merged.columns.str.contains("f1")]
            print(df_merged.to_latex(float_format="%.2f"))

    def run_all_analysis(self, test_loader, d):
        """
        Run analysis on the test set and generate evaluation plots and reports.
        This method sets the model to evaluation mode and iterates over the test_loader to perform
        inference. It aggregates the predicted class probabilities and the corresponding true labels,
        translating them into a hierarchical format based on the taxonomy provided. For each depth level
        (excluding the root level), it computes:
            - The recovery of true labels for the corresponding hierarchy level.
            - Confusion matrices for recall and precision, saving the plots as PDF files.
            - ROC curves for the predicted probabilities, saving the plots as PDF files.
            - A classification report that is both printed on the console and saved as a CSV file.

        Parameters:
                test_loader (iterable): An iterable (e.g., DataLoader) that yields batches of test data, where each
                                                                batch is a dictionary containing tensors (and other values) including the key 'label'.
                d (int): An integer representing the number of days used in the trigger, incorporated into the naming of output files.

        Returns:
                None
        """

        self.eval()
        nodes_by_depth = self.taxonomy.get_nodes_by_depth()

        true_classes = []
        combined_pred_df = []
        combined_true_df = []

        print(f'==========\nStarting Analysis for Trigger + {d} days...')

        # Run inference on the test set and combine the output dataframes
        for batch in tqdm(test_loader, desc='Testing'):

            # Move everything to the device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Run inference and get the predictions df
            pred_df = self.predict_class_probabilities_df(batch)

            # Make dataframe for true labels
            true_df = self.taxonomy.get_hierarchical_one_hot_encoding(batch['label'])
            true_df = pd.DataFrame(true_df, columns=pred_df.columns)

            true_classes += batch['label'].tolist()
            combined_pred_df.append(pred_df)
            combined_true_df.append(true_df)
        
        true_classes = np.array(true_classes)
        combined_pred_df = pd.concat(combined_pred_df, ignore_index=True)
        combined_true_df = pd.concat(combined_true_df, ignore_index=True)

        # Run the analysis on the combined dataframe for each level
        for depth in nodes_by_depth:
            
            # Skip the root node since it will always have a probability of 1
            if depth != 0:

                print(f'----------\nDEPTH: {depth}')

                # Make dirs for plots and reports
                Path(f"{self.model_dir}/plots/depth{depth}").mkdir(parents=True, exist_ok=True)
                Path(f"{self.model_dir}/reports/depth{depth}").mkdir(parents=True, exist_ok=True)

                # Get all the nodes at depth 
                nodes = nodes_by_depth[depth]

                # Only select the classes at the appropriate depth
                level_pred_df = combined_pred_df[nodes]
                level_pred_classes = nodes[np.argmax(level_pred_df.to_numpy(), axis=1)]

                level_true_df = combined_true_df[nodes]

                # Compute the true class at the appropriate level
                true_paths = self.taxonomy.get_paths(true_classes)

                # For all the true classes, grab the label from the correct level
                level_true_classes = []
                for i in range(len(true_paths)):
                    try:
                        level_true_classes.append(true_paths[i][depth]) 
                    except IndexError:
                        # For some objects, a label may not exist at finer levels
                        level_true_classes.append(None)
                
                # Make the recall confusion matrix plot
                cf_title = f"Trigger+{d} days"
                Path(f"{self.model_dir}/plots/depth{depth}/cf_recall").mkdir(parents=True, exist_ok=True)
                cf_img_file = f"{self.model_dir}/plots/depth{depth}/cf_recall/cf_trigger+{d}.pdf"
                plot_confusion_matrix(np.array(level_true_classes), np.array(level_pred_classes), nodes, title=cf_title, img_file=cf_img_file)

                # Make the precision confusion matrix plot
                cf_title = f"Trigger+{d} days"
                Path(f"{self.model_dir}/plots/depth{depth}/cf_precision").mkdir(parents=True, exist_ok=True)
                cf_img_file = f"{self.model_dir}/plots/depth{depth}/cf_precision/cf_trigger+{d}.pdf"
                plot_confusion_matrix(np.array(level_true_classes), np.array(level_pred_classes), nodes, normalize='pred', title=cf_title, img_file=cf_img_file)

                # cf_title = f"Trigger+{d} days"
                # Path(f"{self.model_dir}/plots/depth{depth}/cf_plain").mkdir(parents=True, exist_ok=True)
                # cf_img_file = f"{self.model_dir}/plots/depth{depth}/cf_plain/cf_trigger+{d}.png"
                # plot_plain_cf(np.array(level_true_classes), np.array(level_pred_classes), nodes, title=cf_title, img_file=cf_img_file)

                # Make the ROC plot
                roc_title = f"Trigger+{d} days"
                Path(f"{self.model_dir}/plots/depth{depth}/roc").mkdir(parents=True, exist_ok=True)
                roc_img_file = f"{self.model_dir}/plots/depth{depth}/roc/roc_trigger+{d}.pdf"
                plot_roc_curves(level_true_df.to_numpy(), level_pred_df.to_numpy(), nodes, title=roc_title, img_file=roc_img_file)

                # Make classification report
                report_file = f"{self.model_dir}/reports/depth{depth}/report_trigger+{d}.csv"
                report = self.create_classification_report(np.array(level_true_classes), np.array(level_pred_classes), report_file)
                print(report)








