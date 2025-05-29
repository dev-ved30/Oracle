import torch

import numpy as np
import pandas as pd

from tqdm import tqdm

from oracle.visualization import plot_confusion_matrix, plot_roc_curves

class Tester:
    
    def setup_testing(self, model_dir, device):

        self.model_dir = model_dir
        self.device = device

    # def save_classification_report():

    #     classification_report(y_true, y_pred)

    def run_all_analysis(self, test_loader, d):

        self.eval()
        nodes_by_depth = self.taxonomy.get_nodes_by_depth()

        true_classes = []
        combined_pred_df = []
        combined_true_df = []

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
                
                # Make the confusion matrix plot
                cf_title = f"Trigger+{d} days"
                cf_img_file = f"{self.model_dir}/plots/cf_d{depth}_trigger+{d}.pdf"
                plot_confusion_matrix(np.array(level_true_classes), np.array(level_pred_classes), nodes, title=cf_title, img_file=cf_img_file)

                # Make the ROC plot
                roc_title = f"Trigger+{d} days"
                roc_img_file = f"{self.model_dir}/plots/roc_d{depth}_trigger+{d}.pdf"
                plot_roc_curves(level_true_df.to_numpy(), level_pred_df.to_numpy(), nodes, title=roc_title, img_file=roc_img_file)













