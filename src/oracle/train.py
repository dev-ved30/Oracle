import os
import time
import torch
import argparse

from tqdm import tqdm
from pathlib import Path    
from torch.utils.data import DataLoader

from oracle.loss import WHXE_Loss
from oracle.taxonomies import ORACLE_Taxonomy, BTS_Taxonomy
from oracle.architectures import *
from oracle.custom_datasets.ELAsTiCC import *
from oracle.custom_datasets.BTS import *

# <----- Defaults for training the models ----->
default_num_epochs = 100
default_batch_size = 1024
default_learning_rate = 1e-5
default_alpha = 0.0
default_max_n_per_class = int(1e7)
default_model_dir = Path('./models/test_model')

# <----- Config for the model ----->
model_choices = ["ORACLE1_ELAsTiCC", "ORACLE1-lite_ELAsTiCC", "ORACLE1-lite_BTS", "ORACLE2_swin_ELAsTiCC", "ORACLE2-lite_swin_ELAsTiCC", "ORACLE2_swin_BTS", "ORACLE2-lite_swin_BTS", "ORACLE2-pro_swin_BTS"]
default_model_type = "ORACLE1_ELAsTiCC"

# Switch device to GPU if available
#device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
torch.set_default_device(device)

def parse_args():
    '''
    Get commandline options
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',choices=model_choices, default=default_model_type, help='Type of model to train.')
    parser.add_argument('--num_epochs', type=int, default=default_num_epochs, help='Number of epochs to train the model for.')
    parser.add_argument('--batch_size', type=int, default=default_batch_size, help='Batch size used for training.')
    parser.add_argument('--lr', type=float, default=default_learning_rate, help='Learning rate used for training.')
    parser.add_argument('--max_n_per_class', type=int, default=default_max_n_per_class, help='Maximum number of samples for any class. This allows for balancing of datasets. ')
    parser.add_argument('--alpha', type=float, default=default_alpha, help='Alpha value used for the loss function. See Villar et al. (2024) for more information. [https://arxiv.org/abs/2312.02266]')
    parser.add_argument('--dir', type=Path, default=default_model_dir, help='Directory for saving the models and best model during training.')

    args = parser.parse_args()
    return args

def save_args_to_csv(args, filepath):

    df = pd.DataFrame([vars(args)])  # Wrap in list to make a single-row DataFrame
    df.to_csv(filepath, index=False)    

def run_training_loop(args):

    # Assign the arguments to variables
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    max_n_per_class = args.max_n_per_class
    alpha = args.alpha
    model_dir = args.dir
    model_choice = args.model

    # Create the model directory if it does not exist   
    model_dir.mkdir(parents=True, exist_ok=True)
    save_args_to_csv(args, f'{model_dir}/train_args.csv')

    # Keep generator on the CPU
    generator = torch.Generator(device=device)

    # Assign the taxonomy based on the choice
    if model_choice == "ORACLE1_ELAsTiCC":

        taxonomy = ORACLE_Taxonomy()
        model = ORACLE1(taxonomy)
        dataset = ELAsTiCC_LC_Dataset('data/ELAsTiCC/train.parquet', include_lc_plots=False, transform=truncate_ELAsTiCC_light_curve_fractionally, max_n_per_class=max_n_per_class)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_ELAsTiCC, generator=generator)

    elif model_choice == "ORACLE1-lite_ELAsTiCC":

        taxonomy = ORACLE_Taxonomy()
        model = ORACLE1_lite(taxonomy)
        dataset = ELAsTiCC_LC_Dataset('data/ELAsTiCC/train.parquet', include_lc_plots=False, transform=truncate_ELAsTiCC_light_curve_fractionally, max_n_per_class=max_n_per_class)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_ELAsTiCC, generator=generator)

    elif model_choice == "ORACLE1-lite_BTS":

        taxonomy = BTS_Taxonomy()
        model = ORACLE1_lite(taxonomy, ts_feature_dim=4)
        dataset = BTS_LC_Dataset(BTS_train_parquet_path, transform=truncate_BTS_light_curve_fractionally, max_n_per_class=max_n_per_class)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_BTS, generator=generator)

    elif model_choice == "ORACLE2-lite_swin_LSST":

        taxonomy = ORACLE_Taxonomy()
        model = ORACLE2_lite_swin(taxonomy)
        dataset = ELAsTiCC_LC_Dataset('data/ELAsTiCC/train.parquet', include_lc_plots=True, transform=truncate_ELAsTiCC_light_curve_fractionally, max_n_per_class=max_n_per_class)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_ELAsTiCC, generator=generator)

    elif model_choice == "ORACLE2-pro_swin_BTS":

        taxonomy = BTS_Taxonomy()
        model = ORACLE2_pro_swin(taxonomy)
        dataset = BTS_LC_Dataset(BTS_train_parquet_path, include_lc_plots=True, include_postage_stamps=True, transform=truncate_BTS_light_curve_fractionally, max_n_per_class=max_n_per_class)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_BTS, generator=generator)

    model = model.to(device)
    model.train()

    # Assign the loss function and optimizer
    loss_fn = WHXE_Loss(taxonomy, alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    avg_train_losses = []

    # Training loop
    for epoch in range(num_epochs):

        print(f"Epoch {epoch+1}/{num_epochs} started")
        start_time = time.time()

        train_loss_values = []

        # Loop over all the batches in the data set
        for i, batch in enumerate(tqdm(dataloader, desc='Training Epoch')):

            # Move everything to the device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Get the label encodings
            label_encodings = torch.from_numpy(taxonomy.get_hierarchical_one_hot_encoding(batch['label'])).to(device=device)           

            # Forward pass
            logits = model(batch)
            loss = loss_fn(logits, label_encodings)

            train_loss_values.append(loss.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # TODO: Add validation loop here and log the value of the loss
        avg_train_loss = np.mean(train_loss_values)
        avg_train_losses.append(avg_train_loss)
        print(f"Avg training loss: {float(avg_train_loss):.4f}")

        if np.isnan(avg_train_loss) == True:
            print("Training loss was nan. Exiting the loop.")
            break

        # TODO: Save the model. We can load the best model based on the validation loss for inference.
        torch.save(model.state_dict(), f'{model_dir}/model_epoch{epoch+1}.pth')
        print(f"Time taken: {time.time() - start_time:.2f}s\n=======\n")

def main():
    args = parse_args()
    run_training_loop(args)

if __name__=='__main__':

    main()