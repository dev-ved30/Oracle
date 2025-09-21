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

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Trainer:
    
    def setup_training(self, alpha, beta, lr, train_labels, val_labels, model_dir, device, wandb_run):

        self.alpha = alpha
        self.beta = beta
        
        # Set up criterion for training and validation. These need to be different because the class weights can be different
        self.train_criterion = WHXE_Loss(self.taxonomy, train_labels, self.alpha, self.beta)
        self.val_criterion = WHXE_Loss(self.taxonomy, val_labels, self.alpha)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.model_dir = model_dir
        self.device = device
        self.wandb_run = wandb_run
        self.early_stopper = EarlyStopper(100, 1e-3)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=20, factor=0.8, threshold=lr/100)

    def train_one_epoch(self, train_loader):

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
        
        # Save artifacts to wandb
        # Save artifacts to wandb
        print('Saving model to wandb')
        wandb.save(f"{self.model_dir}/train_loss_history.npy")
        wandb.save(f"{self.model_dir}/val_loss_history.npy")
        wandb.save(f"{self.model_dir}/f1_history.npy")
        wandb.save(f"{self.model_dir}/best_model_f1.pth")
        wandb.save(f"{self.model_dir}/best_model.pth")
        wandb.save(f"{self.model_dir}/train_args.csv")

    def save_loss_history(self, train_loss_history, val_loss_history, f1_history):

        np.save(f"{self.model_dir}/train_loss_history.npy", np.array(train_loss_history))
        np.save(f"{self.model_dir}/val_loss_history.npy", np.array(val_loss_history))
        np.save(f"{self.model_dir}/f1_history.npy", np.array(f1_history))

    def fit(self, train_loader, val_loader, num_epochs=5):

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