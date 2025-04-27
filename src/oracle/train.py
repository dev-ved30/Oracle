import torch
import argparse
from tqdm import tqdm

from pathlib import Path    
from torch.utils.data import DataLoader

from oracle.loss import WHXE_Loss
from oracle.taxonomies import ORACLE_Taxonomy, BTS_Taxonomy
from oracle.models import Light_curve_classifier, Multi_modal_classifier, ORACLE_1
from oracle.datasets.ELAsTiCC import get_ELAsTiCC_dataset, collate_ELAsTiCC_lc_data, truncate_lcs_fractionally

# <----- Defaults for training the models ----->
default_num_epochs = 10
default_batch_size = 256
default_learning_rate = 1e-3
default_alpha = 0.5
default_model_dir = Path('./base_model')

# <----- Config for the model ----->
model_choices = ["LC", "MM", "ORACLE_1"]
default_model_type = "LC"
config = {
    "layer1_neurons": 512,
    "layer1_dropout": 0.3,
    "layer2_neurons": 128,
    "layer2_dropout": 0.2,
}

# <----- Config for the taxonomy ----->
taxonomy_choices = ["ORACLE", "BTS"]
default_taxonomy_type = "BTS"


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
    parser.add_argument('--model_type',choices=model_choices, default=default_model_type, help='Type of model to train.')
    parser.add_argument('--taxonomy',choices=taxonomy_choices, default=default_taxonomy_type, help='Taxonomies for training the hierarchical model.')
    parser.add_argument('--num_epochs', type=int, default=default_num_epochs, help='Number of epochs to train the model for.')
    parser.add_argument('--batch_size', type=int, default=default_batch_size, help='Batch size used for training.')
    parser.add_argument('--lr', type=float, default=default_learning_rate, help='Learning rate used for training.')
    parser.add_argument('--alpha', type=float, default=default_alpha, help='Alpha value used for the loss function. See Villar et al. (2024) for more information. [https://arxiv.org/abs/2312.02266]')
    parser.add_argument('--dir', type=Path, default=default_model_dir, help='Directory for saving the models and best model during training.')

    args = parser.parse_args()
    return args

def run_training_loop(args):

    # Assign the arguments to variables
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    alpha = args.alpha
    model_dir = args.dir

    taxonomy_choice = args.taxonomy
    model_choice = args.model_type

    # Assign the taxonomy based on the choice
    if taxonomy_choice == "ORACLE":

        taxonomy = ORACLE_Taxonomy()
        dataset = get_ELAsTiCC_dataset('test')
        dataset = dataset.with_transform(truncate_lcs_fractionally)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_ELAsTiCC_lc_data, shuffle=True)

    # elif taxonomy_choice == "BTS":

    #     taxonomy = BTS_Taxonomy()
    #     dataset = BTS_LC_Image_Dataset('data/BTS/bts_lc_images')

    # Assign the model based on the choice
    if model_choice == "LC":
        model = Light_curve_classifier(config, taxonomy)
    elif model_choice == "MM":
        model = Multi_modal_classifier(config, taxonomy)
    elif model_choice == "ORACLE_1":
        model = ORACLE_1(taxonomy)

    model = model.to(device)

    # Assign the loss function and optimizer
    loss_fn = WHXE_Loss(taxonomy, alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create the model directory if it does not exist   
    model_dir.mkdir(parents=True, exist_ok=True)      


    # Training loop
    for epoch in range(num_epochs):

        print(f"Epoch {epoch+1}/{num_epochs} started")

        # Loop over all the batches in the data set
        for i, instance in enumerate(tqdm(dataloader)):

            # Get the label encodings
            label_encodings = torch.from_numpy(taxonomy.get_hierarchical_one_hot_encoding(instance['labels']))           

            # Forward pass
            logits = model(instance)
            loss = loss_fn(logits, label_encodings)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Training progress
        print(f"Loss: {loss.item()}\n=======")

        # TODO: Log the value of the loss on the training set

        # TODO: Add validation loop here and log the value of the loss

        # TODO: Save the model as checkpoint. We can load the best model based on the validation loss for inference.

def main():
    args = parse_args()
    run_training_loop(args)

if __name__=='__main__':

    main()