"""
Top level module for defining the Weighted Hierarchical Cross Entropy Loss function for hierarchical classification tasks.
"""
import torch

import torch.nn as nn
import numpy as np

from oracle.taxonomies import Taxonomy, ORACLE_Taxonomy

device = "cuda" if torch.cuda.is_available() else "cpu"

# Implementation of Weighted Hierarchical Cross Entropy loss function by Villar et. al. 2023 (https://arxiv.org/abs/2312.02266) based on the Hierarchical Cross Entropy loss function by Bertinetto et. al. 2019 (https://arxiv.org/abs/1912.09393)
class WHXE_Loss(nn.Module):
    """
    Implementation of the Weighted Hierarchical Cross Entropy Loss function."""

    def __init__(self, taxonomy:Taxonomy, labels, alpha=0.5, beta=1):
        """
        Initializes an instance of the WHXE_Loss class.

        Parameters:
            taxonomy (Taxonomy): An instance providing hierarchical structure, level order traversal,
                                 depths, and sibling mask information.
            labels (array-like): A collection of labels used to generate the weights for the loss function.
            alpha (float, optional): A weighting parameter used within the loss function (default is 0.5).
            beta (float, optional): An exponent to scale the computed class weights (default is 1).

        Attributes:
            taxonomy (Taxonomy): The provided taxonomy for hierarchical structures.
            alpha (float): The alpha value used in the loss computation.
            beta (float): The beta value used to scale class weights.
            level_order_nodes (list): List of nodes in level order as obtained from the taxonomy.
            depths (dict): A dictionary mapping each node to its depth in the taxonomy.
            masks (dict): Sibling masks computed from the taxonomy, used in hierarchical loss calculations.
            N_nodes (int): The total number of nodes in the taxonomy.
            class_weights (Tensor): The precomputed and beta-scaled class weights based on hierarchical one-hot encodings.
        """

        super(WHXE_Loss, self).__init__()

        # Set the parameters
        self.taxonomy = taxonomy
        self.alpha = alpha
        self.beta = beta

        # Use taxonomy to get other useful information
        self.level_order_nodes = self.taxonomy.get_level_order_traversal()
        self.depths = self.taxonomy.get_depths()
        self.masks = self.taxonomy.get_sibling_masks()

        # Count the number of nodes in the taxonomy
        self.N_nodes = len(self.level_order_nodes)

        # Pre-compute the class weights
        true_encodings = self.taxonomy.get_hierarchical_one_hot_encoding(labels)
        self.class_weights = self.get_class_weights(torch.from_numpy(true_encodings).to(device))**self.beta

        print(self.taxonomy.get_level_order_traversal(), self.class_weights)

        # Pre-compute additional terms used in the loss function
        self.compute_lambda_term()

    def compute_lambda_term(self):
        """
        Compute the lambda term for node weighting.

        This method calculates the secondary weighting term using an exponential decay based on the node depths.
        The decay is controlled by the attribute 'alpha'. The resulting lambda term emphasizes different
        levels of the tree according to their depth.
        
        Returns:
            None

        Side Effects:
            - Sets self.lambda_term to a PyTorch tensor of shape (N_nodes) containing the computed values.
        """

        # Compute the secondary weight term, which emphasizes different levels of the tree. See paper for more details.

        # Lambda term of size (N_nodes), ordered in level order traversal
        self.lambda_term = torch.from_numpy(np.exp(-self.alpha * self.depths)).to(device=device)

    def get_class_weights(self, true):
        """
        Computes the class weights for each node in the taxonomy based on the true label data, using inverse frequency weighting.

        Parameters:
            true (torch.Tensor): A binary tensor of shape (N_samples, N_nodes) where each row represents a sample and each column corresponds to a node in the taxonomy. An element should be 1 if the sample belongs to the class represented by the node, and 0 otherwise.
        
        Returns:
            torch.Tensor: A 1D tensor of shape (N_nodes,) containing the computed class weights.
        """
        
        # Total number of samples
        N_samples = true.shape[0]

        # Count the number of samples for each node in the taxonomy. Should have shape (N_nodes)
        N_counts = torch.sum(true, dim=0)

        # Find the class weights and replace inf with 0
        class_weights = N_samples / (self.N_nodes * N_counts)
        class_weights[torch.isinf(class_weights)] = 0

        return class_weights

    def forward(self, logits, true, epsilon=1e-10):
        """
        Compute the hierarchical loss using the pseudo probabilities from masked softmaxes based on a taxonomy structure.

        Parameters:
            logits (torch.Tensor): The raw output logits from the model for a batch.
            true (torch.Tensor): A tensor containing the indicator values for the true class labels.
            epsilon (float, optional): A small constant to prevent logarithm of zero; defaults to 1e-10.

        Returns:
            torch.Tensor: A scalar tensor representing the averaged hierarchical loss over the batch.
        """

        #TODO: This could also be sped up....potentially.

        # Apply softmax to sets of siblings for the logits in order to get the pseudo conditional probabilities 
        conditional_probabilities = self.taxonomy.get_conditional_probabilities(logits)

        # At this point we have the masked soft maxes i.e. the pseudo probabilities. We can take the log of these values
        log_p = torch.log(conditional_probabilities)

        # Weight them by the level at which the corresponding node appears in the hierarchy
        log_p = log_p * self.lambda_term

        # Weight them by the class weight after using the target_probabilities as indicators. Then sum them up for each batch
        v1 = torch.sum(self.class_weights * (log_p * true), dim=1)

        # Finally, find the mean over all batches. Since we are taking logs of numbers <1 (the pseudo probabilities), we have to multiply by -1 to get a +ve loss value.
        v2 = -1 * torch.mean(v1)

        return v2

if __name__ == '__main__':

    # <--- Example usage of the loss function --->
    taxonomy = ORACLE_Taxonomy()
    n_nodes = len(list(taxonomy.nodes()))

    # Compute the loss
    loss = WHXE_Loss(taxonomy)
    true = taxonomy.get_hierarchical_one_hot_encoding(taxonomy.get_leaf_nodes())

    print(loss(torch.from_numpy(true * 10**2), torch.from_numpy(true)))


