import torch
import joblib
import umap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from tqdm import tqdm
from pathlib import Path    

from oracle.visualization import plot_confusion_matrix, plot_roc_curves, plot_train_val_history, plot_class_wise_performance_over_all_phases, plot_average_performance_over_all_phases

class Tester:
    
    def setup_testing(self, model_dir, device):

        self.model_dir = model_dir
        self.one_hot_encoder = joblib.load(f"{self.model_dir}/encoder.pkl")
        self.device = device

    def create_loss_history_plot(self):

        # Load the train and validation loss history
        train_loss_history = np.load(f"{self.model_dir}/train_loss_history.npy")
        val_loss_history = np.load(f"{self.model_dir}/val_loss_history.npy")
        
        # Save the plot
        file_name = f"{self.model_dir}/loss.pdf"
        plot_train_val_history(train_loss_history, val_loss_history, file_name)

    def create_metric_phase_plots(self):
        
        for metric in ['f1-score','precision','recall']:
            metrics_dictionary = self.get_metric_over_all_phases(metric)
            plot_class_wise_performance_over_all_phases(metric, metrics_dictionary, self.model_dir)
            plot_average_performance_over_all_phases(metric, metrics_dictionary, self.model_dir)


    def create_classification_report(self, y_true, y_pred, file_name=None):
        
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
        
        # Make sure metric is valid
        assert metric in ['f1-score','precision','recall']

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
    
    def get_umap_of_latent_space(self, embeddings):

        reducer = umap.UMAP()
        umap_embedding = reducer.fit_transform(embeddings.to_numpy())
        return umap_embedding
    
    def save_umap_plot(self, umap_embedding, trues, title, file):

        plt.close('all')
        plt.style.use(['default'])

        for c in np.unique(trues):

            idx = np.where(np.asarray(trues)==c)
            plt.scatter(umap_embedding[idx,0], umap_embedding[idx,1], label=c)

        plt.legend()
        plt.title(title)
        plt.savefig(file)
        plt.close()

    def run_all_analysis(self, test_loader, d):

        self.eval()

        true_classes = []
        combined_pred_df = []
        combined_true_df = []
        combined_embeddings = []

        print(f'==========\nStarting Analysis for Trigger + {d} days...')

        # Run inference on the test set and combine the output dataframes
        for batch in tqdm(test_loader, desc='Testing'):

            # Move everything to the device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Run inference and get the predictions df
            pred_df = self.predict_conditional_probabilities_df(batch)
            embeddings = self.get_latent_space_embeddings(batch).detach()
            embeddings = pd.DataFrame(embeddings)
            combined_embeddings.append(embeddings)

            # Make dataframe for true labels
            true_df = self.one_hot_encoder.transform(np.asarray(batch['label']).reshape(-1, 1)).toarray()
            true_df = pd.DataFrame(true_df, columns=pred_df.columns)

            true_classes += batch['label'].tolist()
            combined_pred_df.append(pred_df)
            combined_true_df.append(true_df)
        
        true_classes = np.array(true_classes)
        combined_pred_df = pd.concat(combined_pred_df, ignore_index=True)
        combined_true_df = pd.concat(combined_true_df, ignore_index=True)
        combined_embeddings = pd.concat(combined_embeddings, ignore_index=True)
        u_map_embeddings = self.get_umap_of_latent_space(combined_embeddings)


        # Make dirs for plots and reports
        Path(f"{self.model_dir}/plots/depthleaf").mkdir(parents=True, exist_ok=True)
        Path(f"{self.model_dir}/reports/depthleaf").mkdir(parents=True, exist_ok=True)

        nodes_by_depth = {
            'leaf': combined_pred_df.columns
        }
        depth = 'leaf'

        # Get all the nodes at depth 
        nodes = nodes_by_depth[depth]

        # Only select the classes at the appropriate depth
        level_pred_df = combined_pred_df
        level_pred_classes = nodes[np.argmax(level_pred_df.to_numpy(), axis=1)]

        level_true_df = combined_true_df
        level_true_classes = nodes[np.argmax(level_true_df.to_numpy(), axis=1)]
        
        # Make the confusion matrix plot
        cf_title = f"Trigger+{d} days"
        cf_img_file = f"{self.model_dir}/plots/depth{depth}/cf_trigger+{d}.pdf"
        plot_confusion_matrix(np.array(level_true_classes), np.array(level_pred_classes), nodes, title=cf_title, img_file=cf_img_file)

        umap_title = f"Trigger+{d} days"
        umap_img_file = f"{self.model_dir}/plots/depth{depth}/umap_trigger+{d}.pdf"
        self.save_umap_plot(u_map_embeddings, true_classes, umap_title, umap_img_file)

        # Make the ROC plot
        roc_title = f"Trigger+{d} days"
        roc_img_file = f"{self.model_dir}/plots/depth{depth}/roc_trigger+{d}.pdf"
        plot_roc_curves(level_true_df.to_numpy(), level_pred_df.to_numpy(), nodes, title=roc_title, img_file=roc_img_file)

        

        # Make classification report
        report_file = f"{self.model_dir}/reports/depth{depth}/report_trigger+{d}.csv"
        report = self.create_classification_report(np.array(level_true_classes), np.array(level_pred_classes), report_file)
        print(report)








