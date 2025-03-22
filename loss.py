import torch

import torch.nn as nn
import numpy as np

from taxonomies import Taxonomy, ORACLE_Taxonomy



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

        # Class weights of size (N_nodes), ordered in level order traversal
        class_weights = self.get_class_weights(true)

        # Go through set of siblings
        for mask in self.masks:

            mask = torch.from_numpy(mask)

            # Get the e^logits
            exps = torch.exp(logits)

            # Multiply (dot product) the e^logits with the mask to maintain just the e^logits values that belong to this mask. All other values will be zeros.
            masked_exps = torch.multiply(exps, mask)

            # Find the sum of the e^logits values that belong to the mask. Do this for each element in the batch separately. Add a small value to avoid numerical problems with floating point numbers.
            masked_sums = torch.sum(masked_exps, dim=1, keepdim=True) + epsilon

            # Compute the softmax by dividing the e^logits with the sum (e^logits)
            softmax = masked_exps/masked_sums

            # (1 - mask) * y_pred gets the logits for all the values not in this mask and zeros out the values in the mask. Add those back so that we can repeat the process for other masks.
            logits = softmax + ((1 - mask) * logits)

        # At this point we have the masked soft maxes i.e. the pseudo probabilities. We can take the log of these values
        logits = torch.log(logits)

        # Weight them by the level at which the corresponding node appears in the hierarchy
        logits = logits * self.lambda_term

        # Weight them by the class weight after using the target_probabilities as indicators. Then sum them up for each batch
        v1 = torch.sum(class_weights * (logits * true), dim=1)

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


