import torch
import argparse

from pathlib import Path    

from loss import WHXE_Loss
from taxonomies import ORACLE_Taxonomy, BTS_Taxonomy
from models import Light_curve_classifier, Multi_modal_classifier

# <----- Defaults for training the models ----->
default_num_epochs = 10
default_batch_size = 32
default_learning_rate = 1e-3
default_alpha = 0.5
default_model_dir = Path('./base_model')

# <----- Config for the model ----->
model_choices = ["LC", "MM"]
default_model_type = "LC"
config = {
    "layer1_neurons": 512,
    "layer1_dropout": 0.3,
    "layer2_neurons": 128,
    "layer2_dropout": 0.2,
}

# <----- Config for the taxonomy ----->
taxonomy_choices = ["ORACLE", "BTS"]
default_taxonomy_type = "ORACLE"


# Switch device to GPU if available
device = "cpu"
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
    elif taxonomy_choice == "BTS":
        taxonomy = BTS_Taxonomy()

    # Assign the model based on the choice
    if model_choice == "LC":
        model = Light_curve_classifier(config, taxonomy)
    elif model_choice == "MM":
        model = Multi_modal_classifier(config, taxonomy)

    # Assign the loss function and optimizer
    loss_fn = WHXE_Loss(taxonomy, alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create the model directory if it does not exist   
    model_dir.mkdir(parents=True, exist_ok=True)      


    # TODO: Initialize the dataloader here
    data = torch.rand(10, 3, 256, 256)
    labels = torch.rand(10, len(taxonomy.nodes))

    # Training loop
    for i in range(num_epochs):

        # Forward pass
        logits = model(data)
        loss = loss_fn(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Training progress
        print(f"Epoch {i+1}/{num_epochs}, Loss: {loss.item()}")

        # TODO: Add validation loop here

        # TODO: Save the model


if __name__=='__main__':

    args = parse_args()
    run_training_loop(args)