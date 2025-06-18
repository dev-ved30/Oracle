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
from oracle.custom_datasets.ZTF_sims import *

# <----- Defaults for training the models ----->
default_batch_size = 1024
default_max_n_per_class = int(1e7)
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
    parser.add_argument('--dir', type=Path, required=True, help='Directory for saved model.')
    parser.add_argument('--batch_size', type=int, default=default_batch_size, help='Batch size used for test.')
    parser.add_argument('--max_n_per_class', type=int, default=default_max_n_per_class, help='Maximum number of samples for any class. This allows for balancing of datasets.')
    args = parser.parse_args()
    return args 

def run_testing_loop(args):

    # Assign the arguments to variables
    batch_size = args.batch_size
    max_n_per_class = args.max_n_per_class
    model_dir = args.dir

    # Get the model choice
    model_choice = pd.read_csv(f'{model_dir}/train_args.csv')['model'][0]

    # Create the model directory if it does not exist   
    Path(f"{model_dir}/plots").mkdir(parents=True, exist_ok=True)
    Path(f"{model_dir}/reports").mkdir(parents=True, exist_ok=True)

    # Keep generator on the CPU
    generator = torch.Generator(device=device)

    # Assign the taxonomy based on the choice
    if model_choice == "ORACLE1_ELAsTiCC":
        
        # Define the model taxonomy and architecture and load model weights
        taxonomy = ORACLE_Taxonomy()
        model = ORACLE1(taxonomy)
        model.load_state_dict(torch.load(f'{model_dir}/best_model.pth', map_location=device))

        # Set up testing
        model = model.to(device)
        model.setup_testing(model_dir, device)

        # Load the dataset
        test_dataset = ELAsTiCC_LC_Dataset(ELAsTiCC_test_parquet_path, include_lc_plots=False, max_n_per_class=max_n_per_class)

        for d in defaults_days_list:
            
            # Set the transform and create dataloader
            test_dataset.transform = partial(truncate_ELAsTiCC_light_curve_by_days_since_trigger, d=d)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_ELAsTiCC, generator=generator)

            model.run_all_analysis(test_dataloader, d)

    elif model_choice == "ORACLE1-lite_ELAsTiCC":

        taxonomy = ORACLE_Taxonomy()
        model = ORACLE1_lite(taxonomy)
        model.load_state_dict(torch.load(f'{model_dir}/best_model.pth', map_location=device))

        # Set up testing
        model = model.to(device)
        model.setup_testing(model_dir, device)

        # Load the dataset
        test_dataset = ELAsTiCC_LC_Dataset(ELAsTiCC_test_parquet_path, include_lc_plots=False, max_n_per_class=max_n_per_class)

        for d in defaults_days_list:
            
            # Set the transform and create dataloader
            test_dataset.transform = partial(truncate_ELAsTiCC_light_curve_by_days_since_trigger, d=d)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_ELAsTiCC, generator=generator)

            model.run_all_analysis(test_dataloader, d)

    elif model_choice == "ORACLE1-lite_BTS":

        # Define the model taxonomy and architecture
        taxonomy = BTS_Taxonomy()
        model = ORACLE1_lite(taxonomy)
        model.load_state_dict(torch.load(f'{model_dir}/best_model.pth', map_location=device))

        # Set up testing
        model = model.to(device)
        model.setup_testing(model_dir, device)

        # Load the training set
        test_dataset = BTS_LC_Dataset(BTS_test_parquet_path, include_lc_plots=False, max_n_per_class=max_n_per_class)

        for d in defaults_days_list:
            
            # Set the custom transform and reate dataloader
            test_dataset.transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_BTS, generator=generator)

            model.run_all_analysis(test_dataloader, d)

    elif model_choice == "ORACLE1-lite_ZTFSims":

        # Define the model taxonomy and architecture
        taxonomy = BTS_Taxonomy()
        model = ORACLE1_lite(taxonomy)

        # Define the model taxonomy and architecture
        taxonomy = BTS_Taxonomy()
        model = ORACLE1_lite(taxonomy)
        model.load_state_dict(torch.load(f'{model_dir}/best_model.pth', map_location=device))

        # Set up testing
        model = model.to(device)
        model.setup_testing(model_dir, device)

        # Load the training set
        test_dataset = ZTF_SIM_LC_Dataset(ZTF_sim_test_parquet_path, include_lc_plots=False, max_n_per_class=max_n_per_class)

        for d in defaults_days_list:
            
            # Set the custom transform and reate dataloader
            test_dataset.transform = partial(truncate_BTS_light_curve_by_days_since_trigger, d=d)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_ZTF_SIM, generator=generator)

            model.run_all_analysis(test_dataloader, d)

    model.create_loss_history_plot()
    model.create_metric_phase_plots()


def main():
    args = parse_args()
    run_testing_loop(args)

if __name__=='__main__':

    main()