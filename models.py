import torch

import torch.nn as nn
import pandas as pd
import numpy as np

from torchvision.models import swin_v2_b

from taxonomies import Taxonomy, ORACLE_Taxonomy

swin_v2_b_output_dim = 1000

# Template for the Hierarchical Classifier
class Hierarchical_classifier(nn.Module):

    def __init__(self, taxonomy: Taxonomy):
        super(Hierarchical_classifier, self).__init__()

        self.taxonomy = taxonomy
        self.n_nodes = len(taxonomy.get_level_order_traversal())

    def get_conditional_probabilities(self, x):
        
        logits = self.forward(x)
        conditional_probabilities = self.taxonomy.get_conditional_probabilities(logits).detach().numpy()
        return conditional_probabilities

    def get_class_probabilities(self, x):

        conditional_probabilities = self.get_conditional_probabilities(x)
        class_probabilities = self.taxonomy.get_class_probabilities(conditional_probabilities)
        return class_probabilities
    
    def get_conditional_probabilities_df(self, x):

        level_order_nodes = self.taxonomy.get_level_order_traversal()
        conditional_probabilities = self.get_conditional_probabilities(x)
        df = pd.DataFrame(conditional_probabilities, columns=level_order_nodes)
        return df
    
    def get_class_probabilities_df(self, x):

        level_order_nodes = self.taxonomy.get_level_order_traversal()
        class_probabilities = self.get_class_probabilities(x)
        df = pd.DataFrame(class_probabilities, columns=level_order_nodes)
        return df
    
# TODO: Come up with a catchier name for this class.
    
# Base version of the classifier which only uses the Light curve image
class Light_curve_classifier(Hierarchical_classifier):
        
    def __init__(self, config, taxonomy: Taxonomy):

        super(Light_curve_classifier, self).__init__(taxonomy)

        # TODO: Think about what weights we want to initialize the transformer with.
        self.swin = swin_v2_b()

        # Additional layers for classification
        self.fc = nn.Sequential(
            nn.Linear(swin_v2_b_output_dim, config['layer1_neurons']),
            nn.ReLU(True),
            nn.Dropout(config['layer1_dropout']),
            nn.Linear(config['layer1_neurons'], config['layer2_neurons']),
            nn.ReLU(True),
            nn.Dropout(config['layer2_dropout']),
            nn.Linear(config['layer2_neurons'], self.n_nodes),
        )
    
    def forward(self, x):
        
        transformer_output = self.swin(x)
        logits = self.fc(transformer_output)
        return logits
    
# TODO: Think about multi modal implementation.
class Multi_modal_classifier(Hierarchical_classifier):

    def __init__(self, config, taxonomy: Taxonomy):

        super(Multi_modal_classifier, self).__init__(taxonomy)


    def forward(self, x):
        
        raise NotImplementedError

if __name__ == '__main__':

    taxonomy = ORACLE_Taxonomy()
    config = {
        "layer1_neurons": 512,
        "layer1_dropout": 0.3,
        "layer2_neurons": 128,
        "layer2_dropout": 0.2,
    }
    
    model = Light_curve_classifier(config, taxonomy)
    model.eval()

    x = torch.rand(10, 3, 256, 256)

    print(model.get_conditional_probabilities_df(x))
    print(model.get_class_probabilities_df(x))