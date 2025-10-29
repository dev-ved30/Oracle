import time
import torch

import numpy as np
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from oracle.loss import WHXE_Loss

class Trainer:
    
    def setup_training(self, alpha, lr, model_dir, device):
        
        self.criterion = WHXE_Loss(self.taxonomy, alpha)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.model_dir = model_dir
        self.device = device
        self.scheduler = ReduceLROnPlateau(self.optimizer)

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

            loss = self.criterion(logits, label_encodings)

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
                label_encodings = torch.from_numpy(self.taxonomy.get_hierarchical_one_hot_encoding(batch['label'])).to(device=self.device)           

                # Forward pass
                logits = self(batch)

                loss = self.criterion(logits, label_encodings)
                val_loss_values.append(loss.item())

        return np.mean(val_loss_values)

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

            # Save the best model
            if len(train_loss_history) == 1 or val_loss == min(val_loss_history):
                print("Saving model...")
                torch.save(self.state_dict(), f'{self.model_dir}/best_model.pth')

            self.scheduler.step(val_loss)

            # Dump the train and val loss history
            np.save(f"{self.model_dir}/train_loss_history.npy", np.array(train_loss_history))
            np.save(f"{self.model_dir}/val_loss_history.npy", np.array(val_loss_history))
