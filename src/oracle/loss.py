import torch

import torch.nn as nn
import numpy as np

from oracle.taxonomies import Taxonomy, ORACLE_Taxonomy

# Implementation of Weighted Hierarchical Cross Entropy loss function by Villar et. al. 2023 (https://arxiv.org/abs/2312.02266) based on the Hierarchical Cross Entropy loss function by Bertinetto et. al. 2019 (https://arxiv.org/abs/1912.09393)
class WHXE_Loss(nn.Module):

    def __init__(self, taxonomy:Taxonomy, alpha=0.5):

        super(WHXE_Loss, self).__init__()

        # Set the parameters
        self.taxonomy = taxonomy
        self.alpha = alpha

        # Use taxonomy to get other useful information
        self.level_order_nodes = self.taxonomy.get_level_order_traversal()
        self.depths = self.taxonomy.get_depths()
        self.masks = self.taxonomy.get_sibling_masks()

        # Count the number of nodes in the taxonomy
        self.N_nodes = len(self.level_order_nodes)

        # Pre-compute additional terms used in the loss function
        self.compute_lambda_term()

    def compute_lambda_term(self):

        # Compute the secondary weight term, which emphasizes different levels of the tree. See paper for more details.

        # Lambda term of size (N_nodes), ordered in level order traversal
        self.lambda_term = torch.from_numpy(np.exp(-self.alpha * self.depths))

    def get_class_weights(self, true, epsilon=1e-10):
        
        # Total number of samples
        N_samples = true.shape[0]

        # Count the number of samples for each node in the taxonomy. Should have shape (N_nodes)
        N_counts = torch.sum(true, dim=0) + epsilon

        class_weights = N_samples / (self.N_nodes * N_counts)

        return class_weights

    def forward(self, logits, true, epsilon=1e-10):

        #TODO: This could also be sped up....potentially.

        # Apply softmax to sets of siblings for the logits in order to get the pseudo conditional probabilities 
        conditional_probabilities = self.taxonomy.get_conditional_probabilities(logits)

        # Class weights of size (N_nodes), ordered in level order traversal
        class_weights = self.get_class_weights(true)

        # At this point we have the masked soft maxes i.e. the pseudo probabilities. We can take the log of these values
        log_p = torch.log(conditional_probabilities)

        # Weight them by the level at which the corresponding node appears in the hierarchy
        log_p = log_p * self.lambda_term

        # Weight them by the class weight after using the target_probabilities as indicators. Then sum them up for each batch
        v1 = torch.sum(class_weights * (log_p * true), dim=1)

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


