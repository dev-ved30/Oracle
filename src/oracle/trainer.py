import time
import torch
import wandb
import joblib

import numpy as np
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

def get_loss_weights(one_hot_encodings):

    # Total number of samples
    N = one_hot_encodings.shape[0]
    N_c = one_hot_encodings.shape[1]
    
    # Class counts (sum over rows)
    counts = np.sum(one_hot_encodings, axis=0)  # shape: (C,)
    
    # Inverse frequency: higher weight for rarer classes
    weights = N / (counts * N_c)

    weights = np.where(np.isinf(weights), 0, weights)

    # Convert to float32 1D tensor
    return torch.from_numpy(weights.astype(np.float32)).flatten()

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
    
    def setup_training(self, lr, train_labels, val_labels, model_dir, device, wandb_run):
        
        self.device = device
        self.model_dir = model_dir

        self.one_hot_encoder = OneHotEncoder()
        self.one_hot_encoder.fit(np.asarray(train_labels).reshape(-1, 1))
        joblib.dump(self.one_hot_encoder, f'{model_dir}/encoder.pkl')

        self.train_encodings = self.one_hot_encoder.transform(np.asarray(train_labels).reshape(-1, 1))
        self.val_encodings = self.one_hot_encoder.transform(np.asarray(val_labels).reshape(-1, 1))

        # Set up criterion for training and validation. These need to be different because the class weights can be different
        self.train_weights = get_loss_weights(self.train_encodings).to(device=self.device)
        self.val_weights = get_loss_weights(self.val_encodings).to(device=self.device)

        print(self.train_weights, self.val_weights)

        self.train_criterion = CrossEntropyLoss(weight=self.train_weights)
        self.val_criterion = CrossEntropyLoss(weight=self.val_weights)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.wandb_run = wandb_run
        self.early_stopper = EarlyStopper(25, 1e-3)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.8, threshold=lr/100)

    def train_one_epoch(self, train_loader):

        self.train()
        train_loss_values = []

        # Loop over all the batches in the data set
        for i, batch in enumerate(tqdm(train_loader, desc='Training')):

            # Move everything to the device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Get the label encodings
            label_encodings = torch.from_numpy(
                                                    self.one_hot_encoder.transform(np.asarray(batch['label']).reshape(-1, 1)).toarray()
                                                ).to(device=self.device)           

            # Forward pass
            logits = self(batch)

            loss = self.train_criterion(logits, label_encodings)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss_values.append(loss.item())

        return np.mean(train_loss_values)

    def validate(self, val_loader):

        self.eval()
        val_loss_values = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc='Validation')):

                # Move everything to the device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                # Get the label encodings
                label_encodings = torch.from_numpy(
                                                    self.one_hot_encoder.transform(np.asarray(batch['label']).reshape(-1, 1)).toarray()
                                                ).to(device=self.device)    
                # Forward pass
                logits = self(batch)

                loss = self.val_criterion(logits, label_encodings)
                val_loss_values.append(loss.item())

        return np.mean(val_loss_values)
    
    def log_data_in_wandb(self, train_loss_history, val_loss_history):

        self.wandb_run.log(
            {
                'Train Loss': train_loss_history[-1],
                'Validation Loss': val_loss_history[-1], 
                'Min (Validation Loss)': min(val_loss_history),
                'Min (Train Loss)': min(train_loss_history),
                'Learning rate': self.optimizer.param_groups[0]['lr'],
                'Last Epoch': len(train_loss_history),
            }
        )
    
    def save_model_in_wandb(self):
        
        # Save artifacts to wandb
        print('Saving model to wandb')
        wandb.save(f"{self.model_dir}/train_loss_history.npy")
        wandb.save(f"{self.model_dir}/val_loss_history.npy")
        wandb.save(f"{self.model_dir}/best_model.pth")
        wandb.save(f"{self.model_dir}/train_args.csv")

    def save_loss_history(self, train_loss_history, val_loss_history):

        np.save(f"{self.model_dir}/train_loss_history.npy", np.array(train_loss_history))
        np.save(f"{self.model_dir}/val_loss_history.npy", np.array(val_loss_history))

    def fit(self, train_loader, val_loader, num_epochs=5):

        self.to(self.device)

        train_loss_history = []
        val_loss_history = []

        print(f"==========\nBEGINNING TRAINING\n")

        for epoch in range(num_epochs):

            print(f"----------\nStarting epoch {epoch+1}/{num_epochs}...")

            start_time = time.time()

            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)

            if np.isnan(train_loss) == True:
                print("Training loss was nan. Exiting the loop.")
                break

            print(f"Train Loss: {train_loss:.4f} (Best: {min(train_loss_history):.4f})\n"
                  f"Val Loss: {val_loss:.4f} (Best: {min(val_loss_history):.4f})")
            print(f"Time taken: {time.time() - start_time:.2f}s")

            # Log in weights and biases
            self.log_data_in_wandb(train_loss_history, val_loss_history)

            # Save the best model
            if len(train_loss_history) == 1 or val_loss == min(val_loss_history):
                print("Saving model...")
                torch.save(self.state_dict(), f'{self.model_dir}/best_model.pth')

            # Update the learning rate scheduler state
            self.scheduler.step(val_loss)

            # Dump the train and val loss history
            self.save_loss_history(train_loss_history, val_loss_history)

            # Check for early exit
            if self.early_stopper.early_stop(val_loss):
                print("Early stop condition met. Exiting the loop.")
                break