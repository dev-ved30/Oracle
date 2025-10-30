"""
Interface for training models in the ORACLE framework.
"""
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
default_gamma = 1

# <----- Config for the model ----->
model_choices = ["BTS-lite", "BTS", "ZTF_Sims-lite", "ELAsTiCC", "ELAsTiCC-lite"]
default_model_type = "BTS"

# Switch device to GPU if available
#device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
torch.set_default_device(device)

val_truncation_days = 2 ** np.array(range(11))

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
    parser.add_argument('--gamma', type=float, default=default_gamma, help='Exponent for the training weights.')
    parser.add_argument('--dir', type=Path, default=default_model_dir, help='Directory for saving the models and best model during training.')
    parser.add_argument('--load_weights', type=Path, default=None, help='Path to model which should be loaded before training stars.')

    args = parser.parse_args()
    return args

def save_args_to_csv(args, filepath):
    """
    Save command-line arguments to a CSV file.

    This function converts the attributes of an object, typically parsed from command-line input, into a single-row pandas DataFrame, and saves it to a CSV file at the specified filepath.

    Parameters:
        args (object): An object containing attributes to be saved, often created using argparse.
        filepath (str): The file path (including filename) where the CSV file will be written.

    Returns:
        None
    """

    df = pd.DataFrame([vars(args)])  # Wrap in list to make a single-row DataFrame
    df.to_csv(filepath, index=False)

def get_wandb_run(args):
    """
    Initializes and returns a Weights & Biases (wandb) run with the specified configuration.

    Parameters:
        args: An object that must contain the following attributes:
            1. num_epochs (int): The number of training epochs.
            2. batch_size (int): The batch size to be used.
            3. lr (float): The learning rate for training.
            4. max_n_per_class (int): The maximum number of samples per class.
            5. alpha (float): A hyperparameter  used for controlling loss behavior.
            6. gamma (float): A hyperparameter used for weighting.
            7. dir (str): The directory path where the model should be saved.
            8. model (str): The identifier for the chosen model architecture.
            9. load_weights (str): The file path for the pretrained model weights, if any.

    Returns:
        A wandb run instance initialized with the given configuration, which logs metadata and hyperparameters.
    """

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="vedshah-email-northwestern-university",
        # Set the wandb project where this run will be logged.
        project="ORACLE",
        # Track hyperparameters and run metadata.
        config={
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_n_per_class": args.max_n_per_class,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "model_dir": args.dir,
            "model_choice": args.model,
            "pretrained_model_path": args.load_weights,
        },
    )
    return run    

def run_training_loop(args):
    """
    Runs the training loop for the model using the specified configuration and dataset loaders.

    This function performs the following steps:
        1. Extracts training configuration parameters (e.g., number of epochs, batch size, learning rate, model choice, etc.) from the `args` argument.
        2. Initializes the model based on the provided model choice.
        3. Retrieves the training and validation data loaders along with their corresponding labels.
        4. Initializes a logging run (using WandB) and sets up the directory for saving models and training arguments.
        5. Optionally loads a pretrained model's weights if a valid path is provided.
        6. Moves the model to the appropriate device, sets up the training configuration (including hyperparameters such as alpha and gamma), and begins model training.
        7. After training, saves the model to WandB and finalizes the logging run.

    Parameters:
        args (argparse.Namespace): An object containing all necessary configuration parameters and hyperparameters including:
            1. num_epochs (int): Number of epochs to train the model.
            2. batch_size (int): Size of the batches used in training and validation.
            3. lr (float): Learning rate for the optimizer.
            4. max_n_per_class (int): Maximum number of samples per class for the training data.
            5. alpha (float): Hyperparameter used during training (specific purpose defined by model's setup).
            6. gamma (float): Hyperparameter used during training (specific purpose defined by model's setup).
            7. dir (str): Directory path for saving the model and other related artifacts.
            8. model (str): Identifier to select which model architecture to use.
            9. load_weights (str or None): Path to pretrained model weights. If provided, these weights are loaded into the model.

    Returns:
        None
    """

    # Assign the arguments to variables
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    max_n_per_class = args.max_n_per_class
    max_n_per_class_val = int(np.ceil(max_n_per_class/len(val_truncation_days))) if max_n_per_class != None else None
    alpha = args.alpha
    gamma = args.gamma
    model_dir = args.dir
    model_choice = args.model
    pretrained_model_path = args.load_weights

    # Get the model
    model = get_model(model_choice)

    # Get the train and validation datasets
    train_dataloader, train_labels = get_train_loader(model_choice, batch_size, max_n_per_class, ['Anomaly'])
    val_dataloader, val_labels = get_val_loader(model_choice, batch_size, val_truncation_days, max_n_per_class_val, ['Anomaly'])

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
    model.setup_training(alpha, gamma, lr, train_labels, val_labels, model_dir, device, wandb_run)
    model.fit(train_dataloader, val_dataloader, num_epochs)

    # End the logging run with WandB and upload the model
    model.save_model_in_wandb()
    wandb_run.finish()

def main():
    args = parse_args()
    run_training_loop(args)

if __name__=='__main__':

    main()