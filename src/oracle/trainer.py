"""
Module for training hierarchical models in the ORACLE framework."""
import time
import torch
import wandb

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.optim.lr_scheduler import ReduceLROnPlateau

from oracle.loss import WHXE_Loss


class EarlyStopper:
    """
    EarlyStopper class for monitoring validation loss and triggering early stopping during training.

    Attributes:
        patience (int): Number of consecutive epochs with insufficient improvement allowed before stopping early.
        min_delta (float): Minimum change in validation loss to be considered an improvement.
        counter (int): Counts the number of consecutive epochs without sufficient improvement.
        min_validation_loss (float): The lowest validation loss observed so far.
    """

    def __init__(self, patience=1, min_delta=0):
        """
        Initializes the trainer instance with early stopping parameters.
        Parameters:
            patience (int): Number of consecutive epochs without improvement to tolerate before triggering early stopping. Default is 1.
            min_delta (float): Minimum difference in validation loss to qualify as an improvement between epochs. Default is 0.
        Attributes:
            counter (int): Tracks the number of consecutive epochs without sufficient improvement.
            min_validation_loss (float): Records the best (lowest) validation loss observed, initially set to infinity.
        """
        
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """
        Checks if training should be stopped early based on the current validation loss.
        This method compares the provided validation loss against the minimum validation loss seen so far.
            - If the validation loss is less than the minimum, it updates the minimum value and resets the counter.
            - If the validation loss exceeds the minimum by more than a specified delta, the counter increments.
            - When the counter reaches or exceeds the patience threshold, the method indicates that early stopping is warranted.

        Parameters:
            validation_loss (float): The current validation loss from the evaluation phase.
            
        Returns:
            bool: True if early stopping criterion is met (i.e., the counter has reached the patience limit),
            otherwise False.
        """

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Trainer:
    """
    Top-level class providing training functionalities for hierarchical classification models."""

    def setup_training(self, alpha, gamma, lr, train_labels, val_labels, model_dir, device, wandb_run):
        """
        Set up the training components for the model.
        This method configures the training environment by initializing both the training and validation loss criteria,
        setting up the optimizer, scheduling learning rate adjustments, and configuring early stopping. It also assigns
        various parameters such as the device to use, the model directory for saving checkpoints, and the Weights & Biases
        run instance.

        Parameters:
            alpha (float): Hyperparameter for the training loss function to adjust the influence of the taxonomy.
            gamma (float): Hyperparameter for the loss function that modulates the weighting of different classes.
            lr (float): Learning rate for the optimizer.
            train_labels (iterable): Labels for the training dataset, used to compute class weights in the training loss.
            val_labels (iterable): Labels for the validation dataset, used to compute class weights in the validation loss.
            model_dir (str): Directory path where model checkpoints and related outputs will be stored.
            device (torch.device or str): The computing device (CPU or GPU) on which to run the model.
            wandb_run: Instance of the Weights & Biases run for logging training progress.
            
        Returns:
            None
        """

        self.alpha = alpha
        self.gamma = gamma
        
        # Set up criterion for training and validation. These need to be different because the class weights can be different
        self.train_criterion = WHXE_Loss(self.taxonomy, train_labels, self.alpha, self.gamma)
        self.val_criterion = WHXE_Loss(self.taxonomy, val_labels, self.alpha)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.model_dir = model_dir
        self.device = device
        self.wandb_run = wandb_run
        self.early_stopper = EarlyStopper(100, 1e-3)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=20, factor=0.8, threshold=lr/100)

    def train_one_epoch(self, train_loader):
        """
        Train the model for one epoch.

        This method performs a full training cycle over all batches provided by the `train_loader`.
        For each batch, it moves the batch data to the appropriate device, converts labels into a 
        hierarchical one-hot encoding using the taxonomy, performs a forward pass to compute logits,
        calculates the loss using the train criterion, and applies backpropagation along with an optimizer step.

        Parameters:
            train_loader (iterable): A data loader that yields batches of training data. Each batch is expected to be a
                dictionary where the key 'label' is used to obtain the correct one-hot encoding.

        Returns:
            float: The mean loss value computed over all batches during the epoch.
        """

        self.train()
        train_loss_values = []

        # Loop over all the batches in the data set
        for i, batch in enumerate(tqdm(train_loader, desc='Training')):

            # Move everything to the device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Get the label encodings
            label_encodings = torch.from_numpy(self.taxonomy.get_hierarchical_one_hot_encoding(batch['label'])).to(device=self.device)           

            # Forward pass
            logits = self(batch)

            loss = self.train_criterion(logits, label_encodings)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss_values.append(loss.item())

        return np.mean(train_loss_values)

    def validate_one_epoch(self, val_loader):
        """
        Performs validation for one epoch.

        This method sets the model to evaluation mode and iterates over the validation data loader,
        performing forward passes without gradient computation. It computes the loss using a specified
        validation criterion and aggregates the true and predicted labels for metric calculation.
        Additionally, it constructs a confusion matrix visualization and computes the macro F1 score.

        Args:
            val_loader (torch.utils.data.DataLoader): A data loader for the validation dataset. Each batch
                should be a dictionary with a 'label' key among others and may include tensors that need to be
                moved to the device.

        Returns:
            dict: A dictionary containing:
                - 'val_loss' (float): The average loss computed over all validation batches.
                - 'macro_f1' (float): The macro F1 score calculated using aggregated true and predicted labels.
                - 'cf' (wandb.Image): A wandb.Image object representing the confusion matrix plot.

        Note:
            - The method uses the taxonomy provided by self.taxonomy to filter and encode the labels hierarchically.
            - Leaf node indices are determined using the taxonomy's level order traversal.
            - All computations are performed with gradients disabled using torch.no_grad().
        """

        self.eval()
        val_loss_values = []
        all_true_labels = []
        all_pred_labels = []

        leaf_labels = self.taxonomy.get_leaf_nodes()
        leaf_mask = np.where(np.array([c in leaf_labels for c in self.taxonomy.get_level_order_traversal()])==True)[0]

        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc='Validation')):

                # Move everything to the device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                # Get the label encodings
                label_encodings = self.taxonomy.get_hierarchical_one_hot_encoding(batch['label'])
                label_encodings = torch.from_numpy(label_encodings).to(device=self.device)           

                # Forward pass
                logits = self(batch)
                loss = self.val_criterion(logits, label_encodings)
                val_loss_values.append(loss.item())

                # Record everything for computing F1, accuracy, etc.
                all_true_labels.append(np.argmax(label_encodings[:, leaf_mask].cpu().numpy(), axis=1))
                all_pred_labels.append(np.argmax(logits[:, leaf_mask].cpu().numpy(), axis=1))

        all_true_labels = np.concatenate(all_true_labels)
        all_pred_labels = np.concatenate(all_pred_labels)

        cf = confusion_matrix(all_true_labels, all_pred_labels, normalize='true')
        disp = ConfusionMatrixDisplay(cf, display_labels=leaf_labels)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", values_format=".2g", colorbar=False)

        statistics = {
            'val_loss': np.mean(val_loss_values),
            'macro_f1': f1_score(all_true_labels, all_pred_labels, average='macro'),
            'cf': wandb.Image(fig),
        }

        plt.close(fig)

        return statistics
    
    def log_data_in_wandb(self, train_loss_history, val_loss_history, f1_history, cf):
        """
        Logs training and validation metrics to Weights and Biases (wandb).

        Parameters:
            train_loss_history (List[float]): List of training loss values recorded per epoch.
            val_loss_history (List[float]): List of validation loss values recorded per epoch.
            f1_history (List[float]): List of f1 score values recorded per epoch.
            cf (Any): A configuration parameter or additional information to log.

        Returns:
            None
        """

        self.wandb_run.log(
            {
                'Train Loss': train_loss_history[-1],
                'Validation Loss': val_loss_history[-1], 
                'f1 score': f1_history[-1],
                'Min (Validation Loss)': min(val_loss_history),
                'Min (Train Loss)': min(train_loss_history),
                'Max (f1 score)': max(f1_history),
                'Learning rate': self.optimizer.param_groups[0]['lr'],
                'Last Epoch': len(train_loss_history),
                "cf": cf,
            }
        )

    def save_model_in_wandb(self):
        """
        Saves training artifacts to Weights & Biases (wandb) for experiment tracking.
        Each file is expected to reside in the directory specified by `self.model_dir`.
        """
        
        # Save artifacts to wandb
        print('Saving model to wandb')
        wandb.save(f"{self.model_dir}/train_loss_history.npy")
        wandb.save(f"{self.model_dir}/val_loss_history.npy")
        wandb.save(f"{self.model_dir}/f1_history.npy")
        wandb.save(f"{self.model_dir}/best_model_f1.pth")
        wandb.save(f"{self.model_dir}/best_model.pth")
        wandb.save(f"{self.model_dir}/train_args.csv")

    def save_loss_history(self, train_loss_history, val_loss_history, f1_history):
        """
        Saves the loss and F1 score histories as NumPy binary files in the model directory.

        Parameters:
            train_loss_history (list or array-like): A collection of training loss values.
            val_loss_history (list or array-like): A collection of validation loss values.
            f1_history (list or array-like): A collection of F1 score values.

        Each history is converted to a NumPy array before saving.
        """

        np.save(f"{self.model_dir}/train_loss_history.npy", np.array(train_loss_history))
        np.save(f"{self.model_dir}/val_loss_history.npy", np.array(val_loss_history))
        np.save(f"{self.model_dir}/f1_history.npy", np.array(f1_history))

    def fit(self, train_loader, val_loader, num_epochs=5):
        """
        Train the model for a specified number of epochs.

        This method moves the model to the designated device and iterates over the given number
        of epochs. During each epoch, it trains the model on the training data and evaluates it on
        the validation data. It records the training loss, validation loss, and macro F1 score, and
        saves the best models based on the lowest validation loss and highest F1 score. Additionally,
        the method logs metrics to an external service (e.g., Weights and Biases), updates the learning
        rate scheduler, saves the loss histories, and stops training early if an early stopping condition
        is met.

        Parameters:
            train_loader (DataLoader): DataLoader providing batches of training data.
            val_loader (DataLoader): DataLoader providing batches of validation data.
            num_epochs (int, optional): Number of epochs to train for. Defaults to 5.

        Returns:
            None
        """

        self.to(self.device)

        train_loss_history = []
        val_loss_history = []
        leaf_f1_history = []

        print(f"==========\nBEGINNING TRAINING\n")

        for epoch in range(num_epochs):

            print(f"----------\nStarting epoch {epoch+1}/{num_epochs}...")

            start_time = time.time()

            train_loss = self.train_one_epoch(train_loader)
            val_stats = self.validate_one_epoch(val_loader)
            val_loss, f1, cf = val_stats['val_loss'], val_stats['macro_f1'], val_stats['cf']

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            leaf_f1_history.append(f1)

            if np.isnan(train_loss) == True:
                print("Training loss was nan. Exiting the loop.")
                break

            print(f"Train Loss: {train_loss:.4f} (Best: {min(train_loss_history):.4f})\n"
                  f"Val Loss: {val_loss:.4f} (Best: {min(val_loss_history):.4f})\n"
                  f"Macro f1: {f1:.4f} (Best: {max(leaf_f1_history):.4f})")
            print(f"Time taken: {time.time() - start_time:.2f}s")

            # Log in weights and biases
            self.log_data_in_wandb(train_loss_history, val_loss_history, leaf_f1_history, cf)

            # Save the best model
            if len(train_loss_history) == 1 or val_loss == min(val_loss_history):
                print("Saving model...")
                torch.save(self.state_dict(), f'{self.model_dir}/best_model.pth')
            if len(leaf_f1_history) == 1 or f1 == max(leaf_f1_history):
                print("Saving model with highest f1 score...")
                torch.save(self.state_dict(), f'{self.model_dir}/best_model_f1.pth')

            # Update the learning rate scheduler state
            self.scheduler.step(val_loss)

            # Dump the train and val loss history
            self.save_loss_history(train_loss_history, val_loss_history, leaf_f1_history)

            # Check for early exit
            if self.early_stopper.early_stop(val_loss):
                print("Early stop condition met. Exiting the loop.")
                break