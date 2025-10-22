"""
Interface for testing saved models in the ORACLE framework."""
import os
import time
import torch
import argparse

from tqdm import tqdm
from pathlib import Path    
from torch.utils.data import DataLoader

from oracle.loss import WHXE_Loss
from oracle.taxonomies import ORACLE_Taxonomy, BTS_Taxonomy
from oracle.constants import BTS_to_Astrophysical_mappings_AD
from oracle.architectures import *
from oracle.custom_datasets.ELAsTiCC import *
from oracle.custom_datasets.BTS import *
from oracle.custom_datasets.ZTF_sims import *
from oracle.presets import get_model, get_test_loaders

# <----- Defaults for training the models ----->
default_batch_size = 1024
default_max_n_per_class = None
default_model_dir = None
defaults_days_list = 2 ** np.array(range(11))

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
    parser.add_argument('dir', type=Path, help='Directory for saved model.')
    parser.add_argument('--batch_size', type=int, default=default_batch_size, help='Batch size used for test.')
    parser.add_argument('--max_n_per_class', type=int, default=default_max_n_per_class, help='Maximum number of samples for any class. This allows for balancing of datasets.')
    args = parser.parse_args()
    return args 

def run_testing_loop(args):
    """
    Runs the testing loop for a specified model using the provided arguments.

    This function performs the following steps:
        - Extracts key parameters (batch_size, max_n_per_class, and model directory) from the input arguments.
        - Reads the model choice from 'train_args.csv' located in the specified model directory.
        - Creates necessary subdirectories ('plots' and 'reports') within the model directory if they do not already exist.
        - Loads the appropriate model architecture based on the model choice and loads its pre-trained weights from 'best_model_f1.pth'.
        - Sets up the model for testing and moves it to the designated device.
        - Retrieves test datasets for multiple default days (ignoring the 'Anomaly' class) and runs comprehensive analysis on each.
        - Retrieves additional test datasets (using an alternative mapping for anomalies) to generate embeddings for anomaly detection.
        - Generates and saves performance plots, including loss history and metrics across different phases.
        - Merges and displays performance tables based on a predefined list of thresholds.

    Parameters:
        args (Namespace): An object containing the following attributes:
            1. batch_size (int): The batch size for data loading.
            2. max_n_per_class (int): The maximum number of samples per class for testing.
            3. dir (str): The directory path where the model and related files are stored.

    Returns:
        None

    Note:
        The function assumes that helper functions such as get_model, get_test_loaders, and model-specific methods 
        (e.g., setup_testing, run_all_analysis, make_embeddings_for_AD, create_loss_history_plot, create_metric_phase_plots,
        merge_performance_tables) are defined elsewhere in the codebase.
    """

    # Assign the arguments to variables
    batch_size = args.batch_size
    max_n_per_class = args.max_n_per_class
    model_dir = args.dir

    # Get the model choice
    model_choice = pd.read_csv(f'{model_dir}/train_args.csv')['model'][0]

    # Create the model directory if it does not exist   
    Path(f"{model_dir}/plots").mkdir(parents=True, exist_ok=True)
    Path(f"{model_dir}/reports").mkdir(parents=True, exist_ok=True)

    # Get the correct architecture and load weights
    model = get_model(model_choice)
    model.load_state_dict(torch.load(f'{model_dir}/best_model_f1.pth', map_location=device), strict=False)

    # Set up testing
    model = model.to(device)
    model.setup_testing(model_dir, device)

    # Get the test dataset augmented to different days
    test_loaders = get_test_loaders(model_choice, batch_size, max_n_per_class, defaults_days_list, excluded_classes=['Anomaly'])
    for d, test_dataloader in zip(defaults_days_list, test_loaders):
        model.run_all_analysis(test_dataloader, d)

    # Include the anomalies to see the clustering
    test_loaders = get_test_loaders(model_choice, batch_size, max_n_per_class, defaults_days_list, mapper=BTS_to_Astrophysical_mappings_AD)
    for d, test_dataloader in zip(defaults_days_list, test_loaders):
        model.make_embeddings_for_AD(test_dataloader, d)

    model.create_loss_history_plot()
    model.create_metric_phase_plots()
    model.merge_performance_tables([1, 2, 4, 8, 16, 32, 64, 128, 512, 1024])

def main():
    args = parse_args()
    run_testing_loop(args)

if __name__=='__main__':

    main()