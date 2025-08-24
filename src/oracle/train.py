import torch
import wandb
import argparse

from pathlib import Path    
from torch.utils.data import DataLoader, ConcatDataset

from oracle.custom_datasets.ELAsTiCC import *
from oracle.custom_datasets.BTS import *
from oracle.custom_datasets.ZTF_sims import *
from oracle.presets import get_model, get_train_loader, get_val_loader

# <----- Defaults for training the models ----->
default_num_epochs = 100
default_batch_size = 1024
default_learning_rate = 1e-5
default_alpha = 0.0
default_max_n_per_class = None
default_model_dir = None
default_beta = 1

# <----- Config for the model ----->
model_choices = ["ORACLE1_ELAsTiCC", "ORACLE1-lite_ELAsTiCC", "ORACLE1-lite_BTS", "ORACLE1-lite_ZTFSims", "ORACLE1_BTS", "BTS"]
default_model_type = "BTS"

# Switch device to GPU if available
#device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
torch.set_default_device(device)

val_truncation_days = [2,8,16,32,128,256,1024]

def parse_args():
    '''
    Get commandline options
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('model',choices=model_choices, default=default_model_type, help='Type of model to train.')
    parser.add_argument('--num_epochs', type=int, default=default_num_epochs, help='Number of epochs to train the model for.')
    parser.add_argument('--batch_size', type=int, default=default_batch_size, help='Batch size used for training.')
    parser.add_argument('--lr', type=float, default=default_learning_rate, help='Learning rate used for training.')
    parser.add_argument('--max_n_per_class', type=int, default=default_max_n_per_class, help='Maximum number of samples for any class. This allows for balancing of datasets. ')
    parser.add_argument('--alpha', type=float, default=default_alpha, help='Alpha value used for the loss function. See Villar et al. (2024) for more information. [https://arxiv.org/abs/2312.02266]')
    parser.add_argument('--beta', type=float, default=default_beta, help='Exponent for the training weights.')
    parser.add_argument('--dir', type=Path, default=default_model_dir, help='Directory for saving the models and best model during training.')
    parser.add_argument('--load_weights', type=Path, default=None, help='Path to model which should be loaded before training stars.')

    args = parser.parse_args()
    return args

def save_args_to_csv(args, filepath):

    df = pd.DataFrame([vars(args)])  # Wrap in list to make a single-row DataFrame
    df.to_csv(filepath, index=False)

def get_wandb_run(args):

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="vedshah-email-northwestern-university",
        # Set the wandb project where this run will be logged.
        project="AD",
        # Track hyperparameters and run metadata.
        config={
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_n_per_class": args.max_n_per_class,
            "alpha": args.alpha,
            "beta": args.beta,
            "model_dir": args.dir,
            "model_choice": args.model,
            "pretrained_model_path": args.load_weights,
        },
    )
    return run    

def run_training_loop(args):

    # Assign the arguments to variables
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    max_n_per_class = args.max_n_per_class
    alpha = args.alpha
    beta = args.beta
    model_dir = args.dir
    model_choice = args.model
    pretrained_model_path = args.load_weights

    # Get the model
    model = get_model(model_choice)

    # Get the train and validation datasets
    train_dataloader, train_labels = get_train_loader(model_choice, batch_size, max_n_per_class, ['Anomaly'])
    val_dataloader, val_labels = get_val_loader(model_choice, batch_size, val_truncation_days, ['Anomaly'])

    # This is used to log data
    wandb_run = get_wandb_run(args)
    model_dir = Path(f'./models/{model_choice}/{wandb_run.name}')

    # Create the model directory if it does not exist   
    model_dir.mkdir(parents=True, exist_ok=True)
    save_args_to_csv(args, f'{model_dir}/train_args.csv')

    # Load pretrained model
    if pretrained_model_path != None:
        print(f"Loading pre-trained weights from {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device), strict=False)

    # Fit the model
    model = model.to(device)
    model.setup_training(alpha, beta, lr, train_labels, val_labels, model_dir, device, wandb_run)
    model.fit(train_dataloader, val_dataloader, num_epochs)

    # End the logging run with WandB and upload the model
    model.save_model_in_wandb()
    wandb_run.finish()

def main():
    args = parse_args()
    run_training_loop(args)

if __name__=='__main__':

    main()